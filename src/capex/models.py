"""Domain models shared by every pipeline stage.

The vocabulary (field names, lifecycle stages, funding statuses) follows the technical specification
so the JSON API, the Excel workbook and the code read the same way.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------------------------------


class Phase(StrEnum):
    """Normalized lifecycle stage (spec §8)."""

    IDENTIFIED = "Identified"
    PROPOSED = "Proposed"
    FUNDED = "Funded"
    DESIGN = "Design"
    PROCUREMENT = "RFP / Procurement"
    CONSTRUCTION = "Construction / In Progress"
    COMPLETED = "Completed"
    DEFERRED = "Deferred"
    UNKNOWN = "Unknown"


# Codes used inside LLM schemas (short, unambiguous tokens) -> display values
PHASE_CODES: dict[str, Phase] = {
    "identified": Phase.IDENTIFIED,
    "proposed": Phase.PROPOSED,
    "funded": Phase.FUNDED,
    "design": Phase.DESIGN,
    "rfp_procurement": Phase.PROCUREMENT,
    "construction_in_progress": Phase.CONSTRUCTION,
    "completed": Phase.COMPLETED,
    "deferred": Phase.DEFERRED,
    "unknown": Phase.UNKNOWN,
}

# Order used when comparing how far a project has advanced
PHASE_ORDER: dict[Phase, int] = {
    Phase.UNKNOWN: 0,
    Phase.IDENTIFIED: 1,
    Phase.PROPOSED: 2,
    Phase.FUNDED: 3,
    Phase.DESIGN: 4,
    Phase.PROCUREMENT: 5,
    Phase.CONSTRUCTION: 6,
    Phase.COMPLETED: 7,
    Phase.DEFERRED: 1,
}


class FundingStatus(StrEnum):
    UNFUNDED = "Unfunded"
    REQUESTED = "Requested"
    PARTIALLY_FUNDED = "Partially funded"
    FUNDED = "Funded"
    UNKNOWN = "Unknown"


FUNDING_CODES: dict[str, FundingStatus] = {
    "unfunded": FundingStatus.UNFUNDED,
    "requested": FundingStatus.REQUESTED,
    "partially_funded": FundingStatus.PARTIALLY_FUNDED,
    "funded": FundingStatus.FUNDED,
    "unknown": FundingStatus.UNKNOWN,
}

DOMAINS: list[str] = [
    "transportation",
    "transit",
    "aviation",
    "water",
    "wastewater",
    "stormwater",
    "solid_waste",
    "energy_utilities",
    "facilities",
    "parks_recreation",
    "public_safety",
    "technology",
    "housing_community",
    "economic_development",
    "environmental",
    "other",
]

BUDGET_TYPES: list[str] = [
    "requested",
    "proposed",
    "approved",
    "adopted",
    "allocated",
    "appropriated",
    "programmed",
    "estimated",
    "unknown",
]


class DocType(StrEnum):
    """Document categories from spec §3."""

    CIP_PLAN = "Capital Improvement Plan"
    CIP_PROGRAM = "Capital Improvement Program"
    MULTI_YEAR_PROGRAM = "Multi-Year Capital Program"
    TIP = "Transportation Improvement Program (TIP/STIP)"
    PROPOSED_BUDGET = "Proposed Budget"
    ADOPTED_BUDGET = "Adopted Budget"
    CAPITAL_BUDGET = "Capital Expenditure Budget"
    INFRASTRUCTURE_PLAN = "Infrastructure Investment Plan"
    CAPITAL_REPORT = "Capital Planning Report"
    BOARD_SCHEDULE = "Board-Approved Project/Funding Schedule"
    FACILITY_PROGRAM = "Facility Improvement/Construction Program"
    OTHER = "Other"


class DocStatus(StrEnum):
    ADOPTED = "Adopted/Approved"
    PROPOSED = "Proposed/Tentative"
    UNKNOWN = "Unknown"


# --------------------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------------------


class OrgProfile(BaseModel):
    """What the planner knows about the requested organization."""

    input_name: str
    canonical_name: str
    org_type: str = "other"
    state: str | None = None
    official_domains: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    document_terms: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    planner: str = "template"


class SearchHit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    query: str = ""
    provider: str = ""
    rank: int = 0
    page_age: str | None = None


class ProbeResult(BaseModel):
    """Cheap, deterministic profile of a candidate document (no LLM extraction)."""

    ok: bool = False
    error: str | None = None
    content_hash: str | None = None
    size_bytes: int | None = None
    page_count: int | None = None
    pdf_title: str | None = None
    pdf_date: str | None = None
    text_page_ratio: float = 0.0
    sampled_pages: list[int] = Field(default_factory=list)
    itemized_page_ratio: float = 0.0
    project_rows_per_page: float = 0.0
    est_projects: int = 0
    has_budgets: bool = False
    has_years: bool = False
    has_descriptions: bool = False
    aggregate_only: bool = False
    org_mentions: int = 0
    fiscal_years: list[int] = Field(default_factory=list)
    status_hint: str | None = None
    id_tokens: list[str] = Field(default_factory=list)
    name_tokens: list[str] = Field(default_factory=list)
    first_page_text: str = ""
    llm_profile: dict[str, Any] | None = None


class DocumentCandidate(BaseModel):
    doc_id: str
    url: str
    final_url: str | None = None
    title: str = ""
    file_type: Literal["pdf", "html", "xlsx", "other"] = "other"
    domain: str = ""
    is_official_domain: bool = False
    # a copy of the organization's own document hosted elsewhere (e.g. when the official host blocks downloads)
    is_mirror_of_official: bool = False
    archive_url: str | None = None
    discovered_via: list[str] = Field(default_factory=list)
    snippet: str = ""
    publisher: str | None = None
    publication_date: str | None = None
    coverage_start_year: int | None = None
    coverage_end_year: int | None = None
    document_fiscal_year: int | None = None
    doc_type: DocType = DocType.OTHER
    doc_status: DocStatus = DocStatus.UNKNOWN
    contains_itemized_projects: bool | None = None
    probe: ProbeResult | None = None
    prelim_score: float = 0.0
    score: float | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    selected: bool = False
    selection_rank: int | None = None
    selection_rationale: str | None = None
    rejection_reason: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def coverage_years(self) -> str | None:
        if self.coverage_start_year and self.coverage_end_year:
            if self.coverage_start_year == self.coverage_end_year:
                return f"FY{self.coverage_start_year}"
            return f"FY{self.coverage_start_year}-FY{self.coverage_end_year}"
        if self.coverage_start_year or self.coverage_end_year:
            return f"FY{self.coverage_start_year or self.coverage_end_year}"
        return None


# --------------------------------------------------------------------------------------------------
# Parsing & chunking
# --------------------------------------------------------------------------------------------------


class ParsedLine(BaseModel):
    text: str
    top: float = 0.0
    x0: float = 0.0
    size: float = 0.0
    bold: bool = False
    heading: bool = False
    table_id: int | None = None
    tabular: bool = False


class ParsedTable(BaseModel):
    table_id: int
    rows: list[list[str | None]]
    bbox: tuple[float, float, float, float] | None = None


class ParsedPage(BaseModel):
    pdf_page: int  # 1-based physical page number in the PDF
    printed_page: str | None = None  # page label printed on the page, when different
    width: float = 0.0
    height: float = 0.0
    lines: list[ParsedLine] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    ocr: bool = False

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class ParsedDocument(BaseModel):
    doc_id: str
    content_hash: str
    file_type: str
    title: str = ""
    page_count: int = 0
    pages: list[ParsedPage] = Field(default_factory=list)
    parser: str = ""
    parser_version: str = ""
    warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    seq: int
    pages: list[int]
    printed_pages: dict[int, str] = Field(default_factory=dict)
    section: str = ""
    kind: Literal["table", "text", "mixed"] = "text"
    context: str = ""  # headings / repeated table header / preceding lines (not to be extracted)
    text: str  # content with [[PDF page N]] markers
    content_hash: str
    est_tokens: int = 0
    est_rows: int = 0  # deterministic estimate of project-like table rows


class ScreeningResult(BaseModel):
    chunk_id: str
    provider: str
    model: str
    presence: str
    presence_probs: dict[str, float] = Field(default_factory=dict)
    content_type: str
    content_type_probs: dict[str, float] = Field(default_factory=dict)
    need: str
    need_probs: dict[str, float] = Field(default_factory=dict)
    p_project: float = 0.0
    decision: Literal["extract", "extract_with_context", "skip"] = "extract"
    reason: str = ""
    guard: bool = False
    cached: bool = False
    error: str | None = None


# --------------------------------------------------------------------------------------------------
# Extraction & consolidation
# --------------------------------------------------------------------------------------------------


class FundingYear(BaseModel):
    fiscal_year: int | None = None
    label: str = ""
    amount: float | None = None
    amount_text: str = ""


class ExtractedRecord(BaseModel):
    """One project record as extracted from one chunk (before deduplication)."""

    record_id: str
    organization_name: str
    project_id: str | None = None
    project_name: str
    project_description: str | None = None
    domain: str = "other"
    project_type: str | None = None
    location: str | None = None
    estimated_budget: float | None = None
    budget_text: str | None = None
    budget_currency: str = "USD"
    budget_type: str = "unknown"
    budget_basis: str | None = None
    funding_by_year: list[FundingYear] = Field(default_factory=list)
    funding_sources: str | None = None
    document_fiscal_year: int | None = None
    anticipated_start_year: int | None = None
    anticipated_construction_year: int | None = None
    anticipated_completion_year: int | None = None
    current_phase: Phase = Phase.UNKNOWN
    phase_evidence: str | None = None
    phase_basis: Literal["explicit", "inferred"] = "inferred"
    status_text: str | None = None
    funding_status: FundingStatus = FundingStatus.UNKNOWN
    funding_evidence: str | None = None
    source_document: str
    source_url: str
    source_pages: list[int] = Field(default_factory=list)
    printed_pages: list[str] = Field(default_factory=list)
    source_section: str | None = None
    source_excerpt: str = ""
    extraction_confidence: float = 0.0
    # provenance & validation
    doc_id: str
    chunk_id: str
    model: str = ""
    excerpt_verified: bool = False
    name_verified: bool = False
    flags: list[str] = Field(default_factory=list)
    needs_review: bool = False


class SourceReference(BaseModel):
    record_id: str
    doc_id: str
    document_title: str
    url: str
    page_url: str
    pdf_pages: list[int]
    printed_pages: list[str] = Field(default_factory=list)
    section: str | None = None
    excerpt: str = ""
    original_project_name: str
    original_project_id: str | None = None
    document_fiscal_year: int | None = None
    estimated_budget: float | None = None
    current_phase: Phase = Phase.UNKNOWN
    verified: bool = False


class ConsolidatedProject(BaseModel):
    project_uid: str
    organization_name: str
    project_id: str | None = None
    project_name: str
    aliases: list[str] = Field(default_factory=list)
    project_description: str | None = None
    domain: str = "other"
    project_type: str | None = None
    location: str | None = None
    estimated_budget: float | None = None
    budget_currency: str = "USD"
    budget_type: str = "unknown"
    budget_basis: str | None = None
    budget_source: str | None = None
    budget_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    funding_by_year: list[FundingYear] = Field(default_factory=list)
    funding_sources: str | None = None
    document_fiscal_year: int | None = None
    anticipated_start_year: int | None = None
    anticipated_construction_year: int | None = None
    anticipated_completion_year: int | None = None
    current_phase: Phase = Phase.UNKNOWN
    phase_evidence: str | None = None
    phase_basis: Literal["explicit", "inferred"] = "inferred"
    phase_history: list[dict[str, Any]] = Field(default_factory=list)
    funding_status: FundingStatus = FundingStatus.UNKNOWN
    funding_evidence: str | None = None
    source_document: str = ""
    source_url: str = ""
    source_pages: list[int] = Field(default_factory=list)
    source_section: str | None = None
    source_excerpt: str = ""
    references: list[SourceReference] = Field(default_factory=list)
    extraction_confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    merged_record_ids: list[str] = Field(default_factory=list)
    possible_duplicates: list[str] = Field(default_factory=list)


class MergeAuditEntry(BaseModel):
    decision: Literal["merged", "possible_duplicate"]
    project_uid: str | None = None
    other_project_uid: str | None = None
    record_ids: list[str]
    names: list[str]
    rule: str
    score: float
    signals: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------------------
# Metrics & results
# --------------------------------------------------------------------------------------------------


class UsageRecord(BaseModel):
    """One measured model/search request (spec §12: every model request must be measured)."""

    ts: datetime = Field(default_factory=utcnow)
    stage: str
    task: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    search_requests: int = 0
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None
    request_fee_usd: float = 0.0
    cost_usd: float = 0.0
    duration_ms: int = 0
    retries: int = 0
    success: bool = True
    error: str | None = None
    estimated: bool = False
    cached: bool = False


class JobMetrics(BaseModel):
    documents_discovered: int = 0
    documents_probed: int = 0
    documents_selected: int = 0
    pages_processed: int = 0
    chunks_produced: int = 0
    chunks_screened: int = 0
    chunks_accepted: int = 0
    chunks_rejected: int = 0
    chunks_recall_guard: int = 0
    chunks_extracted: int = 0
    chunks_split: int = 0
    chunks_escalated: int = 0
    projects_extracted_raw: int = 0
    records_rejected_validation: int = 0
    duplicates_merged: int = 0
    unique_projects: int = 0
    possible_duplicates: int = 0
    projects_needing_review: int = 0
    screening_provider: str = ""
    extraction_model: str = ""
    token_usage_by_model: list[dict[str, Any]] = Field(default_factory=list)
    cost_by_stage: dict[str, float] = Field(default_factory=dict)
    total_cost_usd: float = 0.0
    cost_per_unique_project_usd: float | None = None
    cost_includes_estimates: bool = False
    paid_requests: int = 0
    cached_requests: int = 0
    execution_time_s: float = 0.0
    stage_timings_s: dict[str, float] = Field(default_factory=dict)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class JobResult(BaseModel):
    job_id: str
    organization_name: str
    org_profile: OrgProfile | None = None
    generated_at: datetime = Field(default_factory=utcnow)
    selected_documents: list[DocumentCandidate] = Field(default_factory=list)
    discovered_documents: list[DocumentCandidate] = Field(default_factory=list)
    projects: list[ConsolidatedProject] = Field(default_factory=list)
    records: list[ExtractedRecord] = Field(default_factory=list)
    screening: list[ScreeningResult] = Field(default_factory=list)
    merge_audit: list[MergeAuditEntry] = Field(default_factory=list)
    usage: list[UsageRecord] = Field(default_factory=list)
    metrics: JobMetrics = Field(default_factory=JobMetrics)
