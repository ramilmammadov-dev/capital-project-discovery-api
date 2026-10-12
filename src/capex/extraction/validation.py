"""Project validation: grounding, normalization and lifecycle rules (spec §7, §8, §14).

Every record returned by the extraction model is checked against the chunk it came from:

* the project name must occur in the source text (fuzzy, whitespace/punctuation-insensitive) —
  otherwise the record is rejected (no invented project names),
* identifiers, budgets, yearly amounts and years must occur in the text — otherwise the value is
  dropped and the record flagged (no invented budgets or dates),
* cited pages are verified by locating the excerpt/name on the page and corrected when the quote
  sits on a different page of the chunk (reference accuracy),
* totals/subtotals and other aggregate lines are rejected,
* stage and funding status are post-processed by the deterministic lifecycle rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from capex.config import ExtractionConfig
from capex.lifecycle import apply_funding_rules, apply_phase_rules
from capex.models import (
    DOMAINS,
    FUNDING_CODES,
    PHASE_CODES,
    Chunk,
    DocumentCandidate,
    ExtractedRecord,
    FundingStatus,
    FundingYear,
    Phase,
)
from capex.normalize import (
    clean_text,
    digits_only,
    norm_for_match,
    normalize_project_id,
    parse_money,
    parse_year,
    year_tokens,
)

_MARKER = re.compile(r"^\[\[PDF page (\d+)(?: \| printed page ([^\]]+))?\]\]$")
_AGGREGATE = re.compile(
    r"^(grand\s+)?(sub[- ]?)?totals?\b(?!\s+maximum\s+daily)|^(program|fund|department|division|category|section|capital)\s+totals?\b|"
    r"\btotal (capital|program|expenditures?|funding|sources|uses|cost of all)\b|^all projects\b|^various\b$",
    re.IGNORECASE,
)


@dataclass
class ChunkText:
    """Chunk text indexed by page for grounding checks."""

    full: str
    full_norm: str
    pages: dict[int, str]
    pages_norm: dict[int, str]
    printed: dict[int, str]
    numbers: set[str]
    id_text: str

    @classmethod
    def build(cls, chunk: Chunk) -> ChunkText:
        pages: dict[int, list[str]] = {}
        printed: dict[int, str] = {}
        current = chunk.pages[0] if chunk.pages else 1
        for line in chunk.text.splitlines():
            m = _MARKER.match(line.strip())
            if m:
                current = int(m.group(1))
                if m.group(2):
                    printed[current] = m.group(2).strip()
                continue
            pages.setdefault(current, []).append(line)
        page_text = {p: "\n".join(lines) for p, lines in pages.items()}
        full = "\n".join(page_text.values())
        numbers = {digits_only(m.group(0)) for m in re.finditer(r"\d[\d,.]*", full)}
        numbers |= {n.lstrip("0") for n in numbers if n.lstrip("0")}
        return cls(
            full=full,
            full_norm=norm_for_match(full),
            pages=page_text,
            pages_norm={p: norm_for_match(t) for p, t in page_text.items()},
            printed=printed or dict(chunk.printed_pages),
            numbers=numbers,
            id_text=re.sub(r"[^A-Z0-9]", "", full.upper()),
        )


@dataclass
class ValidationOutcome:
    record: ExtractedRecord | None
    reason: str | None = None
    flags: list[str] = field(default_factory=list)


def _match(needle: str, haystack: str) -> float:
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 100.0
    return fuzz.partial_ratio(needle, haystack)


def _best_page(needle_norm: str, ct: ChunkText) -> tuple[int | None, float]:
    best_page, best = None, 0.0
    for page, text in ct.pages_norm.items():
        score = _match(needle_norm, text)
        if score > best:
            best_page, best = page, score
    return best_page, best


def _amount_grounded(text: str, ct: ChunkText) -> bool:
    digits = digits_only(text)
    if not digits:
        return False
    return digits in ct.numbers or digits.lstrip("0") in ct.numbers


def _year_grounded(text: str, ct: ChunkText) -> bool:
    years = {y for _, y in year_tokens(text)}
    if not years:
        return False
    found = {y for _, y in year_tokens(ct.full)}
    return bool(years & found)


def _excerpt_line(name_norm: str, ct: ChunkText, page: int | None) -> str:
    """Fallback excerpt: the source line that best matches the project name (always verbatim)."""
    best_line, best = "", 0.0
    candidates = ct.pages.get(page, ct.full) if page else ct.full
    for line in candidates.splitlines():
        score = fuzz.partial_ratio(name_norm, norm_for_match(line)) if line.strip() else 0
        if score > best:
            best_line, best = line, score
    return re.sub(r"\s{2,}", "  ", best_line.strip())[:240]


def build_record(
    raw: dict[str, Any],
    *,
    index: int,
    chunk: Chunk,
    chunk_text: ChunkText,
    document: DocumentCandidate,
    organization: str,
    amount_unit: str,
    model: str,
    config: ExtractionConfig,
) -> ValidationOutcome:
    flags: list[str] = []
    name = clean_text(raw.get("project_name"))
    if not name or len(name) < 3:
        return ValidationOutcome(None, "empty_name")
    if _AGGREGATE.search(name):
        return ValidationOutcome(None, "aggregate_row")

    ct = chunk_text
    name_norm = norm_for_match(name)
    name_score = _match(name_norm, ct.full_norm)
    if name_score < config.name_match_threshold:
        token_score = fuzz.token_set_ratio(name_norm, ct.full_norm)
        if name_score < 60 and token_score < 90:
            return ValidationOutcome(None, "name_not_in_source")
        flags.append("name_partially_matched")
    name_verified = name_score >= config.name_match_threshold

    # identifier
    project_id = clean_text(raw.get("project_id")) or None
    if project_id:
        norm_id = normalize_project_id(project_id)
        if not norm_id or norm_id not in ct.id_text:
            flags.append("id_not_in_source")
            project_id = None

    # pages & excerpt
    excerpt = clean_text(raw.get("source_excerpt"))[:300]
    excerpt_norm = norm_for_match(excerpt)
    cited = [p for p in (raw.get("source_pages") or []) if isinstance(p, int) and p in ct.pages]
    excerpt_page, excerpt_score = _best_page(excerpt_norm, ct) if excerpt_norm else (None, 0.0)
    name_page, _name_page_score = _best_page(name_norm, ct)
    excerpt_verified = excerpt_score >= config.excerpt_match_threshold
    if not excerpt_verified:
        excerpt = _excerpt_line(name_norm, ct, cited[0] if cited else name_page)
        excerpt_verified = bool(excerpt)
        excerpt_page = name_page
        flags.append("excerpt_replaced_with_source_line")
    anchor_page = excerpt_page if excerpt_score >= config.excerpt_match_threshold else name_page
    if cited:
        if anchor_page and anchor_page not in cited:
            on_cited = max(_match(excerpt_norm or name_norm, ct.pages_norm[p]) for p in cited)
            if on_cited < config.excerpt_match_threshold:
                cited = [anchor_page]
                flags.append("page_corrected")
        pages = sorted(set(cited))
    elif anchor_page:
        pages = [anchor_page]
        flags.append("page_located")
    else:
        pages = chunk.pages[:1]
        flags.append("page_inferred")

    # money
    budget_text = clean_text(raw.get("budget_text")) or None
    estimated_budget = None
    if budget_text:
        if _amount_grounded(budget_text, ct):
            estimated_budget = parse_money(budget_text, amount_unit)
        else:
            flags.append("budget_not_in_source")
            budget_text = None
    funding_years: list[FundingYear] = []
    for item in raw.get("funding_by_year") or []:
        amount_text = clean_text(item.get("amount_text"))
        label = clean_text(item.get("year_label"))
        if not amount_text or not _amount_grounded(amount_text, ct):
            continue
        amount = parse_money(amount_text, amount_unit)
        if amount is None:
            continue
        funding_years.append(
            FundingYear(fiscal_year=parse_year(label), label=label, amount=amount, amount_text=amount_text)
        )
    if estimated_budget is None and funding_years and not budget_text:
        flags.append("budget_missing_yearly_amounts_only")

    # years (must be printed in the chunk)
    years: dict[str, int | None] = {}
    for key, raw_key in (("start", "start_year_text"), ("construction", "construction_year_text"), ("completion", "completion_year_text")):
        text = clean_text(raw.get(raw_key))
        if text and _year_grounded(text, ct):
            years[key] = parse_year(text, prefer="last" if key == "completion" else "first")
        else:
            if text:
                flags.append(f"{key}_year_not_in_source")
            years[key] = None

    # lifecycle & funding (deterministic rules over the model's proposal)
    has_allocation = bool(estimated_budget or funding_years)
    doc_fy = document.document_fiscal_year or document.coverage_start_year
    phase_code = str(raw.get("current_phase") or "unknown")
    phase = PHASE_CODES.get(phase_code, Phase.UNKNOWN)
    phase_evidence = clean_text(raw.get("phase_evidence")) or None
    status_text = clean_text(raw.get("status_text")) or None
    phase, basis, notes = apply_phase_rules(
        phase,
        bool(raw.get("phase_is_explicit")),
        phase_evidence or "",
        status_text or "",
        document.doc_status,
        has_allocation,
        doc_fy,
    )
    flags.extend(f"phase_rule: {n}" for n in notes)
    funding_status = FUNDING_CODES.get(str(raw.get("funding_status") or "unknown"), FundingStatus.UNKNOWN)
    funding_evidence = clean_text(raw.get("funding_evidence")) or None
    funding_status = apply_funding_rules(
        funding_status, funding_evidence or "", status_text or "", phase, document.doc_status, has_allocation
    )

    # confidence & review
    confidence = float(raw.get("extraction_confidence") or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    penalties = {
        "name_partially_matched": 0.15,
        "page_inferred": 0.2,
        "budget_not_in_source": 0.1,
        "id_not_in_source": 0.05,
        "excerpt_replaced_with_source_line": 0.05,
    }
    for flag in flags:
        confidence -= penalties.get(flag, 0.0)
    confidence = round(max(0.0, confidence), 3)
    critical = {"name_partially_matched", "page_inferred", "budget_not_in_source"}
    needs_review = confidence < config.review_confidence_below or bool(critical & set(flags))

    domain = str(raw.get("domain")) if raw.get("domain") in DOMAINS else "other"
    record = ExtractedRecord(
        record_id=f"{chunk.chunk_id}#{index:03d}",
        organization_name=organization,
        project_id=project_id,
        project_name=name,
        project_description=clean_text(raw.get("project_description")) or None,
        domain=domain,
        project_type=clean_text(raw.get("project_type")) or None,
        location=clean_text(raw.get("location")) or None,
        estimated_budget=estimated_budget,
        budget_text=budget_text,
        budget_type=str(raw.get("budget_type") or "unknown") if budget_text or funding_years else "unknown",
        budget_basis=clean_text(raw.get("budget_basis")) or None,
        funding_by_year=funding_years,
        funding_sources=clean_text(raw.get("funding_sources")) or None,
        document_fiscal_year=doc_fy,
        anticipated_start_year=years["start"],
        anticipated_construction_year=years["construction"],
        anticipated_completion_year=years["completion"],
        current_phase=phase,
        phase_evidence=phase_evidence,
        phase_basis=basis,  # type: ignore[arg-type]
        status_text=status_text,
        funding_status=funding_status,
        funding_evidence=funding_evidence,
        source_document=document.title or document.url,
        source_url=document.final_url or document.url,
        source_pages=pages,
        printed_pages=[ct.printed[p] for p in pages if p in ct.printed],
        source_section=clean_text(raw.get("source_section")) or chunk.section or None,
        source_excerpt=excerpt,
        extraction_confidence=confidence,
        doc_id=document.doc_id,
        chunk_id=chunk.chunk_id,
        model=model,
        excerpt_verified=excerpt_verified,
        name_verified=name_verified,
        flags=flags,
        needs_review=needs_review,
    )
    return ValidationOutcome(record, None, flags)
