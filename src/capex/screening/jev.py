"""Jev (TypeSafe) — the primary low-cost screening layer (spec §6).

Jev is a non-generative decision model: for each chunk it returns a calibrated probability for
every allowed label of three questions (presence / content type / need), in one request, for
$0.042 per million input tokens and no output-token cost. Routing uses those probabilities
(``screening.base.route``) instead of a single label, so uncertain-but-plausible chunks are kept.

Requests run concurrently under a semaphore with paced starts (Jev limit: 80 req/s); results are
cached by chunk hash + model + question version, so re-runs never pay twice.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from capex.config import AppConfig
from capex.costs import CostLedger
from capex.models import Chunk, DocumentCandidate, ScreeningResult
from capex.normalize import sha256_text
from capex.screening.base import CONTENT_TYPES, NEED, PRESENCE, QUESTION_VERSION, Screener, argmax, route
from capex.storage.cache import FileCache

log = logging.getLogger(__name__)


def build_questions() -> dict[str, Any]:
    from typesafe_sdk import Choice

    return {
        "presence": Choice(
            instructions=(
                "Does `text` (an excerpt of `document`, a public-sector planning or budget document) contain "
                "information about specific capital projects? Capital projects may be called improvements, "
                "rehabilitation, replacement, expansion, modernization, construction, upgrades, design or "
                "infrastructure investments."
            ),
            criteria=PRESENCE,
        ),
        "content_type": Choice(instructions="What kind of content is `text` mainly?", criteria=CONTENT_TYPES),
        "need": Choice(
            instructions="Should `text` be sent to a project-extraction step?",
            criteria=NEED,
        ),
    }


class JevScreener(Screener):
    name = "jev"

    def __init__(
        self,
        config: AppConfig,
        api_key: str,
        ledger: CostLedger,
        cache: FileCache,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.cfg = config.screening
        self.model = self.cfg.jev_model
        self.ledger = ledger
        self.cache = cache
        if client is None:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

            kwargs: dict[str, Any] = {"api_key": api_key, "model": self.model, "timeout": 30.0}
            if config.providers.typesafe_base_url:
                kwargs["base_url"] = config.providers.typesafe_base_url
            # retries are handled here so that every retry is counted in the ledger
            kwargs["retry"] = RetryPolicy(max_retries=0)
            client = AsyncTypeSafeClient(**kwargs)
        self.client = client
        self.questions = build_questions()
        self._sem = asyncio.Semaphore(self.cfg.concurrency)
        self._starts_per_second = 40.0

    def _cache_key(self, chunk: Chunk) -> str:
        return sha256_text(f"{chunk.content_hash}|{self.model}|{QUESTION_VERSION}")

    async def _call(self, state: dict[str, Any]) -> tuple[Any, int, int]:
        from typesafe_sdk import (
            TypeSafeAPIConnectionError,
            TypeSafeInternalServerError,
            TypeSafeRateLimitError,
        )

        attempt = 0
        started = time.perf_counter()
        while True:
            try:
                response = await self.client.system_one(state=state, questions=self.questions, model=self.model)
                return response, attempt, int((time.perf_counter() - started) * 1000)
            except (TypeSafeRateLimitError, TypeSafeInternalServerError, TypeSafeAPIConnectionError) as exc:
                if attempt >= 4:
                    raise
                attempt += 1
                retry_after = getattr(exc, "retry_after_ms", None)
                delay = (retry_after / 1000) if retry_after else min(10.0, 0.5 * 2**attempt)
                await asyncio.sleep(delay)

    async def _screen_one(self, index: int, chunk: Chunk, document: DocumentCandidate, organization: str) -> ScreeningResult:
        key = self._cache_key(chunk)
        cached = self.cache.get_json("screening", key)
        if cached:
            self.ledger.cache_hit("screening")
            result = ScreeningResult.model_validate(cached)
            # probabilities are cached; the routing decision is recomputed with the current thresholds
            decision, reason, guard, p_project = route(
                result.presence_probs, result.need_probs, result.content_type, chunk, self.cfg
            )
            return result.model_copy(
                update={
                    "chunk_id": chunk.chunk_id,
                    "cached": True,
                    "decision": decision,
                    "reason": reason,
                    "guard": guard,
                    "p_project": round(p_project, 4),
                }
            )

        text = chunk.text
        if chunk.context:
            text = chunk.context[:1500] + "\n---\n" + text
        state = {
            "document": f"{document.title or document.url} — {organization}",
            "text": text[: self.cfg.max_state_chars],
        }
        await asyncio.sleep(min(index / self._starts_per_second, 60.0))  # pace request starts
        async with self._sem:
            try:
                response, retries, duration_ms = await self._call(state)
            except Exception as exc:
                self.ledger.record(
                    stage="screening",
                    task="jev_screening",
                    provider="jev",
                    model=self.model,
                    success=False,
                    error=f"{type(exc).__name__}: {str(exc)[:200]}",
                )
                raise
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        self.ledger.record(
            stage="screening",
            task="jev_screening",
            provider="jev",
            model=getattr(response, "model", None) or self.model,
            input_tokens=int(input_tokens if input_tokens is not None else len(state["text"]) // 4),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            duration_ms=duration_ms,
            retries=retries,
            usage_reported=input_tokens is not None,
        )
        choices = response.choices
        presence = dict(choices["presence"].probabilities)
        content = dict(choices["content_type"].probabilities)
        need = dict(choices["need"].probabilities)
        content_type = argmax(content)
        decision, reason, guard, p_project = route(presence, need, content_type, chunk, self.cfg)
        result = ScreeningResult(
            chunk_id=chunk.chunk_id,
            provider="jev",
            model=getattr(response, "model", None) or self.model,
            presence=argmax(presence),
            presence_probs=presence,
            content_type=content_type,
            content_type_probs=content,
            need=argmax(need),
            need_probs=need,
            p_project=round(p_project, 4),
            decision=decision,  # type: ignore[arg-type]
            reason=reason,
            guard=guard,
        )
        self.cache.put_json("screening", key, result.model_dump(mode="json"))
        return result

    async def screen(self, chunks: list[Chunk], document: DocumentCandidate, organization: str) -> list[ScreeningResult]:
        from capex.screening.heuristic import HeuristicScreener

        results = await asyncio.gather(
            *(self._screen_one(i, chunk, document, organization) for i, chunk in enumerate(chunks)),
            return_exceptions=True,
        )
        failed = [chunks[i] for i, r in enumerate(results) if isinstance(r, BaseException)]
        fallback: dict[str, ScreeningResult] = {}
        if failed:
            # never lose recall because the classifier is unavailable: fall back per chunk
            first_error = next(r for r in results if isinstance(r, BaseException))
            log.warning("Jev failed on %d chunk(s), using heuristic fallback: %s", len(failed), first_error)
            for r in await HeuristicScreener(self.cfg).screen(failed, document, organization):
                r.error = f"jev error: {type(first_error).__name__}: {str(first_error)[:160]}"
                r.reason = f"heuristic fallback after Jev error; {r.reason}"
                fallback[r.chunk_id] = r
        return [
            fallback[chunks[i].chunk_id] if isinstance(r, BaseException) else r
            for i, r in enumerate(results)
        ]

    async def aclose(self) -> None:
        close = getattr(self.client, "aclose", None)
        if callable(close):
            await close()
