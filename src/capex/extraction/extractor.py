"""Low-cost LLM project extraction (spec §7) with adaptive splitting and justified escalation.

Per chunk:

1. cache lookup (chunk hash + model + prompt version + document metadata) — repeat runs cost nothing;
   identical in-flight requests are coalesced,
2. the cheap extraction model (default Claude Haiku 5.5) with strict JSON-schema output,
3. truncated/invalid output -> split the chunk at a row boundary and retry each half (cheap model),
4. dense table returning far fewer records than its deterministic row estimate -> split once and keep
   the better result (recall protection for long tables),
5. records failing grounding en masse (hallucination signal) or still-invalid output -> escalate to the
   stronger model, capped per job (``extraction.max_escalations_per_job``).
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from capex.config import AppConfig, ModelRef
from capex.costs import CostLedger
from capex.extraction.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from capex.extraction.schema import EXTRACTION_SCHEMA
from capex.extraction.validation import ChunkText, build_record
from capex.llm import LLMError, LLMRegistry
from capex.models import Chunk, DocumentCandidate, ExtractedRecord
from capex.normalize import estimate_tokens, find_id_tokens, sha256_text
from capex.storage.cache import FileCache

log = logging.getLogger(__name__)

_ROW_HINT_CHARS = ("$",)


@dataclass
class ChunkExtraction:
    chunk_id: str
    records: list[ExtractedRecord] = field(default_factory=list)
    raw_count: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    split: bool = False
    escalated: bool = False
    models: set[str] = field(default_factory=set)
    error: str | None = None


def split_chunk(chunk: Chunk) -> list[Chunk]:
    """Split a chunk in two at a line boundary near the middle, preferring page markers and row starts."""
    lines = chunk.text.splitlines()
    if len(lines) < 6:
        return [chunk]
    mid = len(lines) // 2
    best, best_score = mid, -1.0
    for i in range(max(2, mid - len(lines) // 4), min(len(lines) - 2, mid + len(lines) // 4)):
        line = lines[i]
        score = 0.0
        if line.startswith("[[PDF page"):
            score = 3.0
        elif line and not line[0].isspace() and (find_id_tokens(line[:40]) or any(c in line for c in _ROW_HINT_CHARS)):
            score = 2.0
        elif line and not line[0].isspace():
            score = 1.0
        score -= abs(i - mid) / len(lines)
        if score > best_score:
            best, best_score = i, score
    first, second = lines[:best], lines[best:]
    if second and not second[0].startswith("[[PDF page"):
        marker = next((ln for ln in reversed(first) if ln.startswith("[[PDF page")), None)
        if marker:
            second = [marker, *second]

    def make(part: list[str], suffix: str) -> Chunk:
        text = "\n".join(part)
        pages = [p for p in chunk.pages if f"[[PDF page {p}]]" in text or f"[[PDF page {p} |" in text]
        share = len(part) / max(1, len(lines))
        return chunk.model_copy(
            update={
                "chunk_id": f"{chunk.chunk_id}{suffix}",
                "text": text,
                "pages": pages or chunk.pages,
                "content_hash": sha256_text(chunk.content_hash + suffix + text),
                "est_tokens": estimate_tokens(text),
                "est_rows": round(chunk.est_rows * share),
            }
        )

    return [make(first, "a"), make(second, "b")]


class Extractor:
    def __init__(self, config: AppConfig, registry: LLMRegistry, ledger: CostLedger, cache: FileCache) -> None:
        self.config = config
        self.cfg = config.extraction
        self.registry = registry
        self.ledger = ledger
        self.cache = cache
        self.primary: ModelRef = config.models.extraction
        self.escalation: ModelRef = config.models.escalation
        self._sem = asyncio.Semaphore(self.cfg.concurrency)
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self.escalations = 0
        self.splits = 0

    # -- single LLM call (cached, coalesced) ------------------------------------------------------

    def _key(self, chunk: Chunk, document: DocumentCandidate, model: str, extra: str) -> str:
        meta = f"{document.doc_status.value}|{document.doc_type.value}|{document.coverage_years}|{document.title}"
        return sha256_text(f"{chunk.content_hash}|{model}|{PROMPT_VERSION}|{meta}|{sha256_text(extra)}")

    async def _call(
        self,
        chunk: Chunk,
        document: DocumentCandidate,
        organization: str,
        extra: str,
        ref: ModelRef,
        task: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Return (parsed output or None, truncated)."""
        key = self._key(chunk, document, ref.model, extra)
        cached = self.cache.get_json("extraction", key)
        if cached is not None:
            self.ledger.cache_hit(task)
            return cached, False
        if key in self._inflight:
            return await self._inflight[key]
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            client = self.registry.get(ref.provider)
            async with self._sem:
                response = await client.json_completion(
                    model=ref.model,
                    system=SYSTEM_PROMPT,
                    user=build_user_message(chunk, document, organization, extra),
                    schema=EXTRACTION_SCHEMA,
                    max_tokens=ref.max_tokens,
                    stage="extraction",
                    task=task,
                    ledger=self.ledger,
                    schema_name="capital_projects",
                )
            data = response.data if isinstance(response.data, dict) and "projects" in response.data else None
            if data is not None:
                self.cache.put_json("extraction", key, data)
            result = (data, response.truncated)
            future.set_result(result)
            return result
        except LLMError as exc:
            log.warning("extraction call failed for %s: %s", chunk.chunk_id, exc)
            future.set_result((None, False))
            return None, False
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            self._inflight.pop(key, None)

    # -- raw extraction with splitting/escalation ------------------------------------------------

    async def _raw(
        self,
        chunk: Chunk,
        document: DocumentCandidate,
        organization: str,
        extra: str,
        depth: int,
        ref: ModelRef,
        outcome: ChunkExtraction,
    ) -> list[tuple[Chunk, dict[str, Any], str]]:
        task = "extraction" if ref is self.primary else "extraction_escalation"
        data, truncated = await self._call(chunk, document, organization, extra, ref, task)
        if data is not None:
            outcome.models.add(ref.model)
            unit = str(data.get("amount_unit") or "unknown")
            return [(chunk, project, unit) for project in data.get("projects") or [] if isinstance(project, dict)]
        # truncated or invalid output
        parts = split_chunk(chunk) if depth < self.cfg.max_split_depth else [chunk]
        if len(parts) > 1:
            outcome.split = True
            self.splits += 1
            results = await asyncio.gather(
                *(self._raw(p, document, organization, extra, depth + 1, ref, outcome) for p in parts)
            )
            return [item for part in results for item in part]
        if ref is self.primary and self._can_escalate():
            self.escalations += 1
            outcome.escalated = True
            return await self._raw(chunk, document, organization, extra, self.cfg.max_split_depth, self.escalation, outcome)
        outcome.error = "extraction failed (invalid or truncated output after splitting)" + (
            "" if not truncated else "; output truncated"
        )
        return []

    def _can_escalate(self) -> bool:
        return self.registry.available(self.escalation.provider) and self.escalations < self.cfg.max_escalations_per_job

    def _build(
        self,
        raw: list[tuple[Chunk, dict[str, Any], str]],
        document: DocumentCandidate,
        organization: str,
        model: str,
        outcome: ChunkExtraction,
    ) -> list[ExtractedRecord]:
        records: list[ExtractedRecord] = []
        texts: dict[str, ChunkText] = {}
        for index, (piece, project, unit) in enumerate(raw):
            ct = texts.get(piece.chunk_id) or texts.setdefault(piece.chunk_id, ChunkText.build(piece))
            result = build_record(
                project,
                index=index,
                chunk=piece,
                chunk_text=ct,
                document=document,
                organization=organization,
                amount_unit=unit,
                model=model,
                config=self.cfg,
            )
            if result.record is None:
                outcome.rejected[result.reason or "rejected"] += 1
            else:
                records.append(result.record)
        return records

    async def extract(
        self, chunk: Chunk, document: DocumentCandidate, organization: str, extra_context: str = ""
    ) -> ChunkExtraction:
        outcome = ChunkExtraction(chunk_id=chunk.chunk_id)
        raw = await self._raw(chunk, document, organization, extra_context, 0, self.primary, outcome)
        records = self._build(raw, document, organization, self.primary.model, outcome)

        # recall protection for dense tables: fewer records than the deterministic row estimate
        if (
            chunk.kind in ("table", "mixed")
            and chunk.est_rows >= 8
            and len(records) < self.cfg.escalate_row_recall_ratio * chunk.est_rows
            and not outcome.split
        ):
            parts = split_chunk(chunk)
            if len(parts) > 1:
                retry = ChunkExtraction(chunk_id=chunk.chunk_id)
                raw_parts = await asyncio.gather(
                    *(self._raw(p, document, organization, extra_context, 1, self.primary, retry) for p in parts)
                )
                retry_records = self._build(
                    [item for part in raw_parts for item in part], document, organization, self.primary.model, retry
                )
                self.splits += 1
                if len(retry_records) > len(records):
                    records, outcome.rejected = retry_records, retry.rejected
                    raw = [item for part in raw_parts for item in part]
                    outcome.split = True

        # hallucination signal: many records whose names are not in the source -> stronger model
        hallucinated = outcome.rejected.get("name_not_in_source", 0)
        if raw and hallucinated >= 3 and hallucinated / len(raw) > self.cfg.escalate_unverified_ratio and self._can_escalate():
            self.escalations += 1
            strong = ChunkExtraction(chunk_id=chunk.chunk_id)
            raw_strong = await self._raw(chunk, document, organization, extra_context, self.cfg.max_split_depth, self.escalation, strong)
            strong_records = self._build(raw_strong, document, organization, self.escalation.model, strong)
            outcome.escalated = True
            if len(strong_records) >= len(records):
                records, outcome.rejected, raw = strong_records, strong.rejected, raw_strong
                outcome.models |= strong.models

        outcome.records = records
        outcome.raw_count = len(raw)
        return outcome
