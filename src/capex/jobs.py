"""Asynchronous job execution for the API (spec §11).

* idempotent submission: an ``Idempotency-Key`` header returns the job created with that key, and an
  identical request (same organization + parameters) re-uses a queued/running job or a job completed
  within ``limits.job_reuse_hours`` — repeat requests never create duplicate records or paid work;
* a bounded worker pool (``limits.max_concurrent_jobs``) consumes a queue persisted in the database;
  jobs left ``running`` by a crash are re-queued on start-up;
* per-job timeout, job-level retry for transient failures (on top of request-level retries in the
  HTTP/LLM/Jev clients), cost cap, structured error codes;
* every measured model/search request is written to the usage ledger as it happens, so
  ``GET /jobs/{id}`` reports accumulated cost live.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from capex.config import AppConfig, Settings
from capex.costs import BudgetExceededError, CostLedger
from capex.export import save_outputs
from capex.llm import LLMError, LLMRegistry, LLMUnavailableError
from capex.models import JobResult
from capex.normalize import clean_text, sha256_text
from capex.pipeline import Pipeline, PipelineOptions, result_dir
from capex.storage.cache import FileCache
from capex.storage.db import Database, JobRow

log = logging.getLogger(__name__)

TRANSIENT = (httpx.TransportError, LLMError, asyncio.TimeoutError, ConnectionError)


class IdempotencyConflictError(ValueError):
    pass


def request_key(organization: str, params: dict[str, Any]) -> str:
    relevant = {k: params.get(k) for k in ("max_documents", "screening")}
    return sha256_text(f"{clean_text(organization).lower()}|{sorted(relevant.items())}")


class JobManager:
    def __init__(
        self,
        settings: Settings,
        config: AppConfig,
        db: Database,
        cache: FileCache,
        registry: LLMRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.db = db
        self.cache = cache
        self.registry = registry or LLMRegistry(config)
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.workers: list[asyncio.Task[None]] = []
        self._results: dict[str, JobResult] = {}

    # -- lifecycle ------------------------------------------------------------------------------

    async def start(self) -> None:
        for row in self.db.jobs_with_status("running"):
            self.db.update_job(row.id, status="queued", stage="queued", warnings=[*row.warnings, "re-queued after restart"])
        for row in sorted(self.db.jobs_with_status("queued"), key=lambda r: r.created_at):
            self.queue.put_nowait(row.id)
        self.workers = [
            asyncio.create_task(self._worker(i), name=f"capex-worker-{i}")
            for i in range(self.config.limits.max_concurrent_jobs)
        ]

    async def stop(self) -> None:
        for task in self.workers:
            task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        await self.registry.aclose()

    # -- submission -----------------------------------------------------------------------------

    def submit(
        self,
        organization: str,
        params: dict[str, Any],
        idempotency_key: str | None = None,
        force_refresh: bool = False,
    ) -> tuple[JobRow, bool]:
        """Return (job, created)."""
        key = request_key(organization, params)
        if idempotency_key:
            existing = self.db.find_job_by_idempotency_key(idempotency_key)
            if existing:
                if existing.request_key != key:
                    raise IdempotencyConflictError("Idempotency-Key was already used for a different request")
                return existing, False
        if not force_refresh:
            reusable = self.db.find_reusable_job(key, self.config.limits.job_reuse_hours)
            if reusable:
                return reusable, False
        row = JobRow(
            id=f"job_{uuid.uuid4().hex[:20]}",
            request_key=key,
            idempotency_key=idempotency_key,
            organization_name=clean_text(organization),
            params={**params, "force_refresh": force_refresh},
            status="queued",
            stage="queued",
            progress={},
            warnings=[],
            errors=[],
        )
        self.db.create_job(row)
        self.queue.put_nowait(row.id)
        return row, True

    # -- execution ------------------------------------------------------------------------------

    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self.run_job(job_id)
            except Exception:  # pragma: no cover - run_job records its own failures
                log.exception("worker %d crashed on %s", index, job_id)
            finally:
                self.queue.task_done()

    async def run_job(self, job_id: str) -> None:
        row = self.db.get_job(job_id)
        if row is None or row.status not in ("queued",):
            return
        attempts = row.attempts + 1
        self.db.update_job(job_id, status="running", stage="discovering", started_at=datetime.now(UTC), attempts=attempts)

        def on_usage(record: Any) -> None:
            self.db.add_usage(job_id, record)

        ledger = CostLedger(self.config, max_cost_usd=self.config.limits.max_job_cost_usd, on_record=on_usage)

        def progress(stage: str, counters: dict[str, Any]) -> None:
            self.db.update_job(job_id, stage=stage, progress=counters, cost_usd=ledger.total_cost)

        params = row.params or {}
        options = PipelineOptions(
            max_documents=int(params.get("max_documents") or self.config.discovery.select_top),
            screening=params.get("screening"),
            force_refresh=bool(params.get("force_refresh")),
        )
        pipeline = Pipeline(self.config, self.cache, self.db, self.registry)
        try:
            result = await asyncio.wait_for(
                pipeline.run(row.organization_name, options, job_id, ledger, progress),
                timeout=self.config.limits.job_timeout_s,
            )
            self.db.update_job(job_id, stage="exporting", cost_usd=ledger.total_cost)
            out = result_dir(self.settings.data_dir, job_id)
            formats = params.get("output_format") or ["json", "xlsx"]
            await asyncio.get_running_loop().run_in_executor(None, save_outputs, result, out, formats)
            self._results[job_id] = result
            m = result.metrics
            self.db.update_job(
                job_id,
                status="completed",
                stage="completed",
                result_dir=str(out),
                finished_at=datetime.now(UTC),
                cost_usd=m.total_cost_usd,
                warnings=m.warnings,
                errors=[{"code": "stage_error", "message": e} for e in m.errors],
                progress={
                    "documents_discovered": m.documents_discovered,
                    "documents_selected": m.documents_selected,
                    "pages_processed": m.pages_processed,
                    "chunks_produced": m.chunks_produced,
                    "chunks_screened": m.chunks_screened,
                    "chunks_accepted": m.chunks_accepted,
                    "chunks_extracted": m.chunks_extracted,
                    "projects_extracted_raw": m.projects_extracted_raw,
                    "unique_projects": m.unique_projects,
                },
            )
        except TimeoutError:
            self._fail(job_id, "timeout", f"job exceeded {self.config.limits.job_timeout_s}s", ledger)
        except BudgetExceededError as exc:
            self._fail(job_id, "cost_cap_exceeded", str(exc), ledger)
        except LLMUnavailableError as exc:
            self._fail(job_id, "provider_not_configured", str(exc), ledger)
        except TRANSIENT as exc:
            if attempts < self.config.limits.job_max_attempts:
                log.warning("job %s transient failure (attempt %d), re-queueing: %s", job_id, attempts, exc)
                current = self.db.get_job(job_id)
                self.db.update_job(
                    job_id,
                    status="queued",
                    stage="queued",
                    warnings=[*(current.warnings if current else []), f"attempt {attempts} failed: {exc}; retrying"],
                )
                await asyncio.sleep(5)
                self.queue.put_nowait(job_id)
            else:
                self._fail(job_id, "upstream_unavailable", f"{type(exc).__name__}: {exc}", ledger)
        except Exception as exc:
            log.exception("job %s failed", job_id)
            self._fail(job_id, "internal_error", f"{type(exc).__name__}: {exc}", ledger)

    def _fail(self, job_id: str, code: str, message: str, ledger: CostLedger) -> None:
        current = self.db.get_job(job_id)
        self.db.update_job(
            job_id,
            status="failed",
            stage="failed",
            finished_at=datetime.now(UTC),
            cost_usd=ledger.total_cost,
            errors=[*(current.errors if current else []), {"code": code, "message": message}],
        )

    # -- results --------------------------------------------------------------------------------

    def load_result(self, job_id: str) -> JobResult | None:
        if job_id in self._results:
            return self._results[job_id]
        row = self.db.get_job(job_id)
        if not row or not row.result_dir:
            return None
        path = Path(row.result_dir) / "result.json"
        if not path.is_file():
            return None
        result = JobResult.model_validate_json(path.read_text("utf-8"))
        if len(self._results) > 16:
            self._results.pop(next(iter(self._results)))
        self._results[job_id] = result
        return result

    def export_path(self, job_id: str) -> Path | None:
        row = self.db.get_job(job_id)
        if not row or not row.result_dir:
            return None
        path = Path(row.result_dir) / "export.xlsx"
        if not path.is_file():
            result = self.load_result(job_id)
            if result is None:
                return None
            save_outputs(result, Path(row.result_dir), ["xlsx"])
        return path
