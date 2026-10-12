"""Request/response models of the public REST API (shown in the OpenAPI / Swagger docs)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from capex.models import ConsolidatedProject, DocumentCandidate, ExtractedRecord, JobMetrics


class ExtractRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "organization_name": "Arizona Department of Transportation",
                    "max_documents": 3,
                    "output_format": ["json", "xlsx"],
                }
            ]
        }
    )

    organization_name: str = Field(min_length=3, max_length=300, description="Public organization name.")
    max_documents: int = Field(3, ge=1, le=5, description="Documents selected for detailed extraction (spec: 3).")
    output_format: list[Literal["json", "xlsx"]] = Field(default=["json", "xlsx"])
    screening: Literal["jev", "heuristic", "none"] | None = Field(
        None, description="Override the screening layer (default: Jev, with heuristic fallback)."
    )
    force_refresh: bool = Field(
        False, description="Start a new job even if an identical recent job exists (caches still prevent paid re-work)."
    )


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class JobLinks(BaseModel):
    self: str
    projects: str
    sources: str
    metrics: str
    export_xlsx: str


class JobSubmitted(BaseModel):
    job_id: str
    status: str
    created: bool = Field(description="false when an existing job was returned (idempotent request)")
    links: JobLinks


class JobStatus(BaseModel):
    job_id: str
    organization_name: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    progress: dict[str, Any]
    warnings: list[str]
    errors: list[dict[str, Any]]
    accumulated_cost_usd: float
    attempts: int
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    links: JobLinks


class ProjectsPage(BaseModel):
    job_id: str
    organization_name: str
    total: int
    page: int
    page_size: int
    next_page: int | None
    items: list[ConsolidatedProject]


class SourcesResponse(BaseModel):
    job_id: str
    organization_name: str
    selected: list[DocumentCandidate]
    discovered: list[DocumentCandidate]


class MetricsResponse(BaseModel):
    job_id: str
    organization_name: str
    metrics: JobMetrics
    usage_records: list[dict[str, Any]] | None = None


class RecordsPage(BaseModel):
    """Raw structured extraction output: one record per project per chunk, before deduplication."""

    job_id: str
    organization_name: str
    total: int
    page: int
    page_size: int
    next_page: int | None
    items: list[ExtractedRecord]
