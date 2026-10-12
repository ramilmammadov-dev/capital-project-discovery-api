"""End-to-end pipeline (spec §2):

Organization name -> document discovery -> ranking -> top-3 selection -> PDF/HTML parsing ->
layout-aware chunking -> Jev screening -> low-cost LLM extraction -> project validation ->
deduplication -> Excel + JSON.

The pipeline is a plain async object with no web-framework dependency: the API job runner, the CLI
and the tests all drive the same code.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capex.chunking import CHUNKER_VERSION, chunk_document, neighbour_context
from capex.config import AppConfig
from capex.consolidation import consolidate
from capex.costs import BudgetExceededError, CostLedger
from capex.discovery import discover
from capex.extraction import Extractor
from capex.fetch import Fetcher
from capex.llm import LLMRegistry
from capex.models import (
    Chunk,
    ExtractedRecord,
    JobMetrics,
    JobResult,
    ParsedDocument,
    ScreeningResult,
)
from capex.normalize import sha256_text, slugify
from capex.parsing import UnsupportedDocumentError, parse_document
from capex.screening import build_screener
from capex.storage.cache import FileCache
from capex.storage.db import Database

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, dict[str, Any]], None]

STAGES = [
    "discovering",
    "ranking",
    "downloading",
    "parsing",
    "chunking",
    "screening",
    "extracting",
    "validating",
    "deduplicating",
    "exporting",
]


@dataclass
class PipelineOptions:
    max_documents: int = 3
    screening: str | None = None  # jev | heuristic | none (default from config)
    force_refresh: bool = False


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        cache: FileCache,
        db: Database | None = None,
        registry: LLMRegistry | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.db = db
        self.registry = registry or LLMRegistry(config)

    # ------------------------------------------------------------------------------------------

    def _chunks(self, parsed: ParsedDocument) -> tuple[list[Chunk], bool]:
        cfg = self.config.chunking
        key = sha256_text(f"{parsed.content_hash}|{parsed.parser_version}|{CHUNKER_VERSION}|{cfg.model_dump_json()}")
        cached = self.cache.get_json("chunks", key)
        if cached is not None:
            chunks = [Chunk.model_validate(c) for c in cached]
            if chunks and chunks[0].doc_id != parsed.doc_id:  # same content reached via another URL
                for c in chunks:
                    c.chunk_id = c.chunk_id.replace(c.doc_id, parsed.doc_id, 1)
                    c.doc_id = parsed.doc_id
            return chunks, True
        chunks = chunk_document(parsed, cfg)
        self.cache.put_json("chunks", key, [c.model_dump(mode="json") for c in chunks])
        return chunks, False

    async def run(
        self,
        organization: str,
        options: PipelineOptions,
        job_id: str,
        ledger: CostLedger,
        progress: ProgressFn | None = None,
    ) -> JobResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        metrics = JobMetrics()
        counters: dict[str, Any] = {}

        def report(stage: str, **values: Any) -> None:
            counters.update(values)
            if progress:
                progress(stage, dict(counters))

        def timed(stage: str, since: float) -> float:
            now = time.perf_counter()
            timings[stage] = round(timings.get(stage, 0.0) + now - since, 3)
            return now

        fetcher = Fetcher(self.config, self.cache, self.db)
        try:
            # 1-3. discovery, ranking, selection -------------------------------------------------
            report("discovering")
            t = time.perf_counter()
            discovery = await discover(
                organization,
                self.config,
                self.registry,
                ledger,
                self.cache,
                fetcher,
                max_documents=options.max_documents,
                progress=lambda stage, values: report(stage, **values),
            )
            t = timed("discovery_and_ranking", t)
            metrics.warnings.extend(discovery.warnings)
            metrics.documents_discovered = len(discovery.candidates)
            metrics.documents_probed = sum(1 for c in discovery.candidates if c.probe)
            metrics.documents_selected = len(discovery.selected)
            profile = discovery.profile
            org_name = profile.canonical_name
            org_key = profile.official_domains[0] if profile.official_domains else slugify(org_name)
            documents = {c.doc_id: c for c in discovery.selected}
            ledger.check_budget()

            # 4. download + parse ----------------------------------------------------------------
            report("downloading")
            parsed_docs: dict[str, ParsedDocument] = {}
            for doc in discovery.selected:
                fetched = await fetcher.fetch_document(doc.url, force=options.force_refresh)
                if not fetched.ok:
                    metrics.errors.append(f"download failed for {doc.url}: {fetched.error}")
                    continue
                t = timed("downloading", t)
                report("parsing")
                try:
                    parsed, _from_cache = await asyncio.get_running_loop().run_in_executor(
                        None, parse_document, fetched, doc.doc_id, self.cache, self.config.parsing, doc.title
                    )
                except (UnsupportedDocumentError, Exception) as exc:
                    metrics.errors.append(f"parsing failed for {doc.title or doc.url}: {exc}")
                    continue
                t = timed("parsing", t)
                parsed_docs[doc.doc_id] = parsed
                metrics.warnings.extend(f"{doc.title[:60]}: {w}" for w in parsed.warnings)
                metrics.pages_processed += len(parsed.pages)
                report("parsing", pages_processed=metrics.pages_processed)

            # 5. chunking ------------------------------------------------------------------------
            report("chunking")
            chunks_by_doc: dict[str, list[Chunk]] = {}
            for doc_id, parsed in parsed_docs.items():
                chunks, _ = self._chunks(parsed)
                chunks_by_doc[doc_id] = chunks
            metrics.chunks_produced = sum(len(c) for c in chunks_by_doc.values())
            t = timed("chunking", t)
            report("chunking", chunks_produced=metrics.chunks_produced)

            # 6. screening -----------------------------------------------------------------------
            report("screening")
            screener, screen_warnings = build_screener(self.config, ledger, self.cache, options.screening)
            metrics.warnings.extend(screen_warnings)
            metrics.screening_provider = f"{screener.name}" + (f" ({screener.model})" if screener.model else "")
            screening: dict[str, ScreeningResult] = {}
            try:
                results = await asyncio.gather(
                    *(screener.screen(chunks_by_doc[d], documents[d], org_name) for d in chunks_by_doc)
                )
            finally:
                await screener.aclose()
            for result_list in results:
                for r in result_list:
                    screening[r.chunk_id] = r
            metrics.chunks_screened = len(screening) if screener.name != "none" else 0
            metrics.chunks_accepted = sum(1 for r in screening.values() if r.decision != "skip")
            metrics.chunks_rejected = sum(1 for r in screening.values() if r.decision == "skip")
            metrics.chunks_recall_guard = sum(1 for r in screening.values() if r.guard)
            if any(r.error for r in screening.values()):
                metrics.warnings.append(
                    f"{sum(1 for r in screening.values() if r.error)} chunk(s) screened by the heuristic fallback after Jev errors"
                )
            t = timed("screening", t)
            report(
                "screening",
                chunks_screened=metrics.chunks_screened,
                chunks_accepted=metrics.chunks_accepted,
                chunks_rejected=metrics.chunks_rejected,
            )
            ledger.check_budget()

            # 7-8. extraction + validation -------------------------------------------------------
            report("extracting")
            extractor = Extractor(self.config, self.registry, ledger, self.cache)
            metrics.extraction_model = self.config.models.extraction.model
            records: list[ExtractedRecord] = []
            rejected: Counter[str] = Counter()
            done = 0
            budget_hit = False

            async def run_chunk(doc_id: str, index: int, chunk: Chunk) -> None:
                nonlocal done, budget_hit
                decision = screening.get(chunk.chunk_id)
                if decision is None or decision.decision == "skip":
                    return
                if budget_hit:
                    return
                try:
                    ledger.check_budget()
                except BudgetExceededError:
                    budget_hit = True
                    return
                extra = neighbour_context(chunks_by_doc[doc_id], index) if decision.decision == "extract_with_context" else ""
                outcome = await extractor.extract(chunk, documents[doc_id], org_name, extra)
                records.extend(outcome.records)
                rejected.update(outcome.rejected)
                metrics.chunks_extracted += 1
                metrics.chunks_split += 1 if outcome.split else 0
                metrics.chunks_escalated += 1 if outcome.escalated else 0
                if outcome.error:
                    metrics.errors.append(f"{chunk.chunk_id}: {outcome.error}")
                done += 1
                if done % 5 == 0:
                    report("extracting", chunks_extracted=done, projects_extracted_raw=len(records))

            await asyncio.gather(
                *(
                    run_chunk(doc_id, index, chunk)
                    for doc_id, chunks in chunks_by_doc.items()
                    for index, chunk in enumerate(chunks)
                )
            )
            if budget_hit:
                metrics.warnings.append(
                    f"cost cap (${self.config.limits.max_job_cost_usd:.2f}) reached — remaining chunks were not extracted"
                )
            records.sort(key=lambda r: r.record_id)
            metrics.projects_extracted_raw = len(records)
            metrics.records_rejected_validation = sum(rejected.values())
            metrics.rejection_reasons = dict(rejected.most_common())
            t = timed("extraction_and_validation", t)
            report("validating", projects_extracted_raw=len(records))

            # 9. deduplication -------------------------------------------------------------------
            report("deduplicating")

            def resolve_uid(alias_keys: list[str], default_uid: str) -> str:
                if self.db is None:
                    return default_uid
                existing = self.db.registry_lookup(org_key, alias_keys)
                uid = existing or default_uid
                self.db.registry_register(org_key, alias_keys, uid)
                return uid

            consolidation = consolidate(records, documents, self.config.dedup, org_key, resolve_uid)
            metrics.duplicates_merged = consolidation.duplicates_merged
            metrics.unique_projects = len(consolidation.projects)
            metrics.possible_duplicates = consolidation.possible_duplicates
            metrics.projects_needing_review = sum(1 for p in consolidation.projects if p.needs_review)
            t = timed("deduplication", t)
            report("deduplicating", unique_projects=metrics.unique_projects)

            # selected-document contribution counts (for the Sources sheet)
            per_doc = Counter(r.doc_id for r in records)
            unique_per_doc = Counter(
                {d: sum(1 for p in consolidation.projects if p.references and p.references[0].doc_id == d) for d in documents}
            )
            for doc in discovery.selected:
                doc.notes.append(f"records extracted: {per_doc.get(doc.doc_id, 0)}")
                doc.notes.append(f"unique projects led by this document: {unique_per_doc.get(doc.doc_id, 0)}")
        finally:
            await fetcher.aclose()

        metrics.execution_time_s = round(time.perf_counter() - started, 2)
        metrics.stage_timings_s = timings
        self._finish_costs(metrics, ledger)
        return JobResult(
            job_id=job_id,
            organization_name=org_name,
            org_profile=profile,
            selected_documents=discovery.selected,
            discovered_documents=discovery.candidates,
            projects=consolidation.projects,
            records=records,
            screening=list(screening.values()),
            merge_audit=consolidation.audit,
            usage=ledger.records,
            metrics=metrics,
        )

    @staticmethod
    def _finish_costs(metrics: JobMetrics, ledger: CostLedger) -> None:
        metrics.token_usage_by_model = ledger.usage_by_model()
        metrics.cost_by_stage = ledger.cost_by_stage()
        metrics.total_cost_usd = ledger.total_cost
        metrics.cost_per_unique_project_usd = (
            round(ledger.total_cost / metrics.unique_projects, 6) if metrics.unique_projects else None
        )
        metrics.cost_includes_estimates = ledger.has_estimates
        metrics.paid_requests = len(ledger.records)
        metrics.cached_requests = sum(ledger.cache_hits.values())


def result_dir(data_dir: Path, job_id: str) -> Path:
    path = data_dir / "jobs" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path
