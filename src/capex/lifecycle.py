"""Deterministic lifecycle & funding-status rules (spec §8).

The extraction model proposes a stage with evidence; these rules then enforce the specification:

* advanced stages (design, procurement, construction, completed, deferred) need explicit evidence words,
* a future construction year alone is not construction-stage,
* a capital-budget approval alone is not evidence that design/procurement/construction started,
* document status supplies the default (adopted program allocation -> Funded; proposed/tentative -> Proposed),
* ``current_phase`` and ``funding_status`` are kept separate.
"""

from __future__ import annotations

import re

from capex.models import DocStatus, FundingStatus, Phase
from capex.normalize import year_tokens

_EVIDENCE: dict[Phase, re.Pattern[str]] = {
    Phase.DESIGN: re.compile(
        r"\b(design|engineering|feasibility|stud(y|ies)|architect\w*|preliminary|scoping|environmental "
        r"(review|clearance|assessment)|planning phase|\d{2}\s?% design|pre-?design|concept\w*)\b",
        re.IGNORECASE,
    ),
    Phase.PROCUREMENT: re.compile(
        r"\b(rfp|rfq|ifb|itb|bids?|bidding|solicit\w*|procure\w*|advertis\w*|tender\w*|award\w*|"
        r"request for (proposals?|qualifications))\b",
        re.IGNORECASE,
    ),
    Phase.CONSTRUCTION: re.compile(
        r"(under construction|construction (is |was )?(underway|in progress|ongoing|began|started|commenced)|"
        r"\bin progress\b|\bunderway\b|\bongoing\b|being (built|installed|constructed)|currently (being )?"
        r"(constructed|installed|implemented)|implementation (is )?underway|\bwip\b)",
        re.IGNORECASE,
    ),
    Phase.COMPLETED: re.compile(
        r"\b(complete[d]?|substantially complete|placed (in|into) service|in service|opened|closed[- ]out|finished)\b",
        re.IGNORECASE,
    ),
    Phase.DEFERRED: re.compile(
        r"\b(defer\w*|postpon\w*|suspend\w*|cancel\w*|on hold|removed|delay\w*|eliminat\w*|unprogrammed)\b",
        re.IGNORECASE,
    ),
}
_ADVANCED = {Phase.DESIGN, Phase.PROCUREMENT, Phase.CONSTRUCTION, Phase.COMPLETED, Phase.DEFERRED}
_UNFUNDED = re.compile(r"\b(unfunded|not funded|no funding|unprogrammed|future need|unmet need)\b", re.IGNORECASE)
_PARTIAL = re.compile(r"\b(partial(ly)? funded|partial funding|funding gap|additional funding (is )?needed)\b", re.IGNORECASE)


def _default_phase(doc_status: DocStatus, has_allocation: bool) -> Phase:
    if doc_status == DocStatus.ADOPTED and has_allocation:
        return Phase.FUNDED
    if doc_status == DocStatus.PROPOSED:
        return Phase.PROPOSED
    if has_allocation:
        return Phase.FUNDED
    return Phase.UNKNOWN


def apply_phase_rules(
    phase: Phase,
    explicit: bool,
    evidence: str,
    status_text: str,
    doc_status: DocStatus,
    has_allocation: bool,
    document_fiscal_year: int | None,
) -> tuple[Phase, str, list[str]]:
    """Return (phase, basis 'explicit'|'inferred', notes)."""
    text = f"{evidence or ''} {status_text or ''}".strip()
    notes: list[str] = []

    if phase in _ADVANCED:
        pattern = _EVIDENCE[phase]
        if not pattern.search(text):
            fallback = _default_phase(doc_status, has_allocation)
            notes.append(f"stage '{phase.value}' not supported by explicit evidence -> '{fallback.value}'")
            return fallback, "inferred", notes
        if phase in (Phase.CONSTRUCTION, Phase.DESIGN, Phase.PROCUREMENT) and document_fiscal_year:
            # "Construction FY2029" in a program table = programmed, not underway
            in_progress = _EVIDENCE[Phase.CONSTRUCTION].search(text) or re.search(
                r"\b(underway|in progress|ongoing|currently|began|started|initiated|issued)\b", text, re.IGNORECASE
            )
            future = [y for _, y in year_tokens(text) if y > document_fiscal_year]
            if future and not in_progress:
                fallback = _default_phase(doc_status, has_allocation)
                notes.append(
                    f"'{phase.value}' evidence refers to a future year ({future[0]}) -> '{fallback.value}'"
                )
                return fallback, "inferred", notes
        return phase, "explicit" if explicit else "inferred", notes

    if phase in (Phase.UNKNOWN, Phase.IDENTIFIED) and has_allocation and doc_status == DocStatus.ADOPTED:
        if not _UNFUNDED.search(text):
            notes.append("adopted program allocation -> 'Funded'")
            return Phase.FUNDED, "inferred", notes
    if phase == Phase.UNKNOWN and _UNFUNDED.search(text):
        notes.append("unfunded wording -> 'Identified'")
        return Phase.IDENTIFIED, "inferred", notes
    if phase == Phase.UNKNOWN:
        fallback = _default_phase(doc_status, has_allocation)
        if fallback != Phase.UNKNOWN:
            notes.append(f"no stage stated; document status -> '{fallback.value}'")
            return fallback, "inferred", notes
    if phase == Phase.FUNDED and doc_status == DocStatus.PROPOSED and not explicit:
        notes.append("document is proposed/tentative -> 'Proposed'")
        return Phase.PROPOSED, "inferred", notes
    return phase, "explicit" if explicit else "inferred", notes


def apply_funding_rules(
    status: FundingStatus,
    evidence: str,
    status_text: str,
    phase: Phase,
    doc_status: DocStatus,
    has_allocation: bool,
) -> FundingStatus:
    text = f"{evidence or ''} {status_text or ''}"
    if _UNFUNDED.search(text):
        return FundingStatus.UNFUNDED
    if _PARTIAL.search(text):
        return FundingStatus.PARTIALLY_FUNDED
    if status != FundingStatus.UNKNOWN:
        if status == FundingStatus.FUNDED and doc_status == DocStatus.PROPOSED and not evidence:
            return FundingStatus.REQUESTED
        return status
    if has_allocation:
        if doc_status == DocStatus.ADOPTED:
            return FundingStatus.FUNDED
        if doc_status == DocStatus.PROPOSED:
            return FundingStatus.REQUESTED
    if phase == Phase.IDENTIFIED:
        return FundingStatus.UNFUNDED if not has_allocation else FundingStatus.UNKNOWN
    return FundingStatus.UNKNOWN
