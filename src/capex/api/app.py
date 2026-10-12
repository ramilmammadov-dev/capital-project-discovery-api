"""REST API (spec §11). OpenAPI/Swagger UI at /docs, ReDoc at /redoc."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException

from capex import __version__
from capex.api.schemas import (
    ErrorResponse,
    ExtractRequest,
    JobLinks,
    JobStatus,
    JobSubmitted,
    MetricsResponse,
    ProjectsPage,
    RecordsPage,
    SourcesResponse,
)
from capex.config import AppConfig, Settings, get_app_config, get_settings
from capex.jobs import IdempotencyConflictError, JobManager
from capex.models import JobResult
from capex.storage.cache import FileCache
from capex.storage.db import Database, JobRow

log = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Service API key")


class APIError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str, details: Any = None) -> None:
        super().__init__(status_code=status_code, detail={"code": code, "message": message, "details": details})


def require_api_key(request: Request, api_key: Annotated[str | None, Depends(_api_key_header)]) -> str:
    accepted = request.app.state.settings.accepted_api_keys
    if not api_key or not any(secrets.compare_digest(api_key, k) for k in accepted):
        raise APIError(401, "unauthorized", "missing or invalid X-API-Key header")
    return api_key


def get_manager(request: Request) -> JobManager:
    return request.app.state.manager


Auth = Annotated[str, Depends(require_api_key)]
Manager = Annotated[JobManager, Depends(get_manager)]


def _links(job_id: str) -> JobLinks:
    base = f"{API_PREFIX}/jobs/{job_id}"
    return JobLinks(
        self=base,
        projects=f"{base}/projects",
        sources=f"{base}/sources",
        metrics=f"{base}/metrics",
        export_xlsx=f"{base}/export.xlsx",
    )


def _status(row: JobRow) -> JobStatus:
    return JobStatus(
        job_id=row.id,
        organization_name=row.organization_name,
        status=row.status,  # type: ignore[arg-type]
        stage=row.stage,
        progress=row.progress or {},
        warnings=row.warnings or [],
        errors=row.errors or [],
        accumulated_cost_usd=round(row.cost_usd or 0.0, 6),
        attempts=row.attempts,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        links=_links(row.id),
    )


def create_app(
    settings: Settings | None = None,
    config: AppConfig | None = None,
    manager: JobManager | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    config = config or get_app_config()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal manager
        if manager is None:
            db = Database(settings.resolved_database_url)
            cache = FileCache(settings.data_dir / "cache")
            manager = JobManager(settings, config, db, cache)
        app.state.manager = manager
        if not settings.accepted_api_keys:
            log.warning("CAPEX_API_KEYS is empty: every authenticated endpoint will return 401")
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(
        title="Public-Sector Capital Project Discovery API",
        version=__version__,
        description=(
            "Discovers official planning/budget documents for a public organization, selects the three most "
            "useful ones, extracts and consolidates capital projects (Jev screening + low-cost LLM extraction), "
            "and returns JSON and an Excel workbook with page-level source references and a full cost breakdown."
        ),
        lifespan=lifespan,
        responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    app.state.settings = settings

    # -- structured errors ------------------------------------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        default_code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": default_code, "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": detail}, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": "request validation failed", "details": exc.errors()}},
        )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - defensive
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": str(exc)}})

    # -- helpers ----------------------------------------------------------------------------------

    def job_or_404(manager: JobManager, job_id: str) -> JobRow:
        row = manager.db.get_job(job_id)
        if row is None:
            raise APIError(404, "job_not_found", f"job {job_id} does not exist")
        return row

    def completed_result(manager: JobManager, job_id: str) -> JobResult:
        row = job_or_404(manager, job_id)
        if row.status != "completed":
            raise APIError(
                409,
                "job_not_completed",
                f"job is {row.status} (stage: {row.stage})",
                {"status": row.status, "stage": row.stage},
            )
        result = manager.load_result(job_id)
        if result is None:
            raise APIError(410, "result_unavailable", "job output is no longer available")
        return result

    # -- routes -----------------------------------------------------------------------------------

    @app.get("/health", tags=["service"])
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.post(
        f"{API_PREFIX}/organizations/extract",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=JobSubmitted,
        tags=["extraction"],
        summary="Start an extraction job for a public organization",
        responses={200: {"model": JobSubmitted, "description": "Existing job returned (idempotent request)"}},
    )
    async def start_extraction(
        body: ExtractRequest,
        _: Auth,
        manager: Manager,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=200)] = None,
    ) -> JSONResponse:
        params = {
            "max_documents": body.max_documents,
            "output_format": body.output_format,
            "screening": body.screening,
        }
        try:
            row, created = manager.submit(body.organization_name, params, idempotency_key, body.force_refresh)
        except IdempotencyConflictError as exc:
            raise APIError(409, "idempotency_conflict", str(exc)) from exc
        payload = JobSubmitted(job_id=row.id, status=row.status, created=created, links=_links(row.id))
        return JSONResponse(status_code=202 if created else 200, content=payload.model_dump(mode="json"))

    @app.get(f"{API_PREFIX}/jobs", tags=["jobs"], response_model=list[JobStatus], summary="List recent jobs")
    async def list_jobs(_: Auth, manager: Manager, limit: int = Query(20, ge=1, le=200)) -> list[JobStatus]:
        return [_status(row) for row in manager.db.list_jobs(limit)]

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}", tags=["jobs"], response_model=JobStatus, summary="Job status")
    async def job_status(job_id: str, _: Auth, manager: Manager) -> JobStatus:
        return _status(job_or_404(manager, job_id))

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}/projects",
        tags=["results"],
        response_model=ProjectsPage,
        summary="Consolidated projects (paginated, filterable)",
    )
    async def job_projects(
        job_id: str,
        _: Auth,
        manager: Manager,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=500),
        domain: str | None = None,
        phase: str | None = Query(None, description="e.g. Funded, Design, 'Construction / In Progress'"),
        funding_status: str | None = None,
        min_budget: float | None = Query(None, ge=0),
        q: str | None = Query(None, description="case-insensitive text search in name/description/location"),
    ) -> ProjectsPage:
        result = completed_result(manager, job_id)
        items = result.projects
        if domain:
            items = [p for p in items if p.domain.lower() == domain.lower()]
        if phase:
            items = [p for p in items if p.current_phase.value.lower() == phase.lower()]
        if funding_status:
            items = [p for p in items if p.funding_status.value.lower() == funding_status.lower()]
        if min_budget is not None:
            items = [p for p in items if (p.estimated_budget or 0) >= min_budget]
        if q:
            needle = q.lower()
            items = [
                p
                for p in items
                if needle in f"{p.project_name} {p.project_description or ''} {p.location or ''}".lower()
            ]
        total = len(items)
        start = (page - 1) * page_size
        return ProjectsPage(
            job_id=job_id,
            organization_name=result.organization_name,
            total=total,
            page=page,
            page_size=page_size,
            next_page=page + 1 if start + page_size < total else None,
            items=items[start : start + page_size],
        )

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}/records",
        tags=["results"],
        response_model=RecordsPage,
        summary="Raw extracted records (before deduplication), paginated",
    )
    async def job_records(
        job_id: str,
        _: Auth,
        manager: Manager,
        page: int = Query(1, ge=1),
        page_size: int = Query(200, ge=1, le=1000),
    ) -> RecordsPage:
        result = completed_result(manager, job_id)
        total = len(result.records)
        start = (page - 1) * page_size
        return RecordsPage(
            job_id=job_id,
            organization_name=result.organization_name,
            total=total,
            page=page,
            page_size=page_size,
            next_page=page + 1 if start + page_size < total else None,
            items=result.records[start : start + page_size],
        )

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}/export.xlsx",
        tags=["results"],
        summary="Download the Excel workbook",
        response_class=FileResponse,
    )
    async def job_export(job_id: str, _: Auth, manager: Manager) -> FileResponse:
        result = completed_result(manager, job_id)
        path = manager.export_path(job_id)
        if path is None:
            raise APIError(410, "result_unavailable", "workbook is no longer available")
        slug = "".join(c if c.isalnum() else "-" for c in result.organization_name.lower()).strip("-")
        return FileResponse(
            path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"capital-projects-{slug}.xlsx",
        )

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}/sources",
        tags=["results"],
        response_model=SourcesResponse,
        summary="Selected documents (with rationale) and the full discovered inventory",
    )
    async def job_sources(job_id: str, _: Auth, manager: Manager) -> SourcesResponse:
        result = completed_result(manager, job_id)
        return SourcesResponse(
            job_id=job_id,
            organization_name=result.organization_name,
            selected=result.selected_documents,
            discovered=result.discovered_documents,
        )

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}/metrics",
        tags=["results"],
        response_model=MetricsResponse,
        summary="Extraction, deduplication and cost metrics",
    )
    async def job_metrics(
        job_id: str,
        _: Auth,
        manager: Manager,
        include_requests: bool = Query(False, description="include every measured model/search request"),
    ) -> MetricsResponse:
        result = completed_result(manager, job_id)
        return MetricsResponse(
            job_id=job_id,
            organization_name=result.organization_name,
            metrics=result.metrics,
            usage_records=[u.model_dump(mode="json") for u in result.usage] if include_requests else None,
        )

    return app
