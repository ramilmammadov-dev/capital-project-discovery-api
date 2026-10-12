"""Extraction prompt. Static instructions go in the (cached) system prompt; the per-chunk message
contains only document metadata, context and the chunk text — token-efficient by construction."""

from __future__ import annotations

from capex.models import Chunk, DocumentCandidate

PROMPT_VERSION = "extract-v3"

SYSTEM_PROMPT = """\
You extract capital projects from excerpts of official public-sector planning and budget documents \
(capital improvement plans and programs, transportation improvement programs, capital budgets, facility \
programs). Return JSON that matches the schema.

EXHAUSTIVENESS
- Return EVERY distinct capital project that appears in CONTENT. Consider every row of every project \
table and every project description. Never return only the "most important" projects or examples: if a \
table lists 30 projects, return 30 records.
- One record per project entry. Phases, segments or components that the document lists as separate \
rows stay separate records.
- Capital projects may be described as improvements, rehabilitation, replacement, expansion, \
modernization, construction, upgrades, design, acquisition, equipment or infrastructure investments, even \
when the words "capital project" are not used.
- Do NOT return: totals, subtotals, fund/department/program summaries, revenue or financing lines, \
operating items, or generic narrative. A recurring program line (e.g. "Pavement Preservation — various \
locations") counts only if it is its own line item with its own budget.
- CONTEXT is only for interpretation (section names, table headers, units, preceding lines). Never return \
a project that appears only in CONTEXT.

FIDELITY — documented facts only
- Copy values exactly as printed: project_id, project_name, amounts (budget_text, amount_text), year \
labels. Do not compute, sum, convert or round. Never invent a name, budget, date or identifier; use an \
empty string (or empty list) when the text does not state it.
- amount_unit: report "thousands"/"millions" when a header or note says so (e.g. "($000)", "in thousands").
- source_pages: the number(s) N from the [[PDF page N]] marker(s) above the lines of the project's entry.
- source_excerpt: a short exact snippet (<= 200 characters) copied from CONTENT that contains the project's \
name or identifier.
- Interpretations (phase, funding status, domain, type) must be supported by the text; set \
phase_is_explicit=true only when the text itself states the stage.

LIFECYCLE STAGE (current_phase)
- identified: need or potential project recognized, no formal commitment (unfunded needs, future/candidate \
projects, master-plan items).
- proposed: in a draft, proposed, tentative or recommended plan/budget, or a funding request.
- funded: approved or adopted allocation, appropriation or documented authorization; procurement not \
necessarily started.
- design: feasibility, study, engineering, architecture, preliminary or detailed design underway.
- rfp_procurement: solicitation, bidding, RFQ/RFP, tender, advertisement or award activity initiated.
- construction_in_progress: physical construction, installation or implementation underway.
- completed: work completed or placed into service.
- deferred: postponed, suspended, cancelled or removed from the active program (keep the wording in \
status_text).
- unknown: no basis in the text.
Rules: a future construction year alone is NOT construction-stage. Budget approval alone does NOT show that \
design, procurement or construction has started. For an entry in an ADOPTED/APPROVED program or budget \
that shows funding, use funded unless the text states a later stage; in a PROPOSED/TENTATIVE/DRAFT document \
use proposed unless the text states otherwise.

FUNDING STATUS (separate from the stage)
- funded: adopted/approved funding covers the project. partially_funded: only part is funded or the text \
says so. requested: proposed budget or funding request. unfunded: explicitly unfunded/unprogrammed. \
unknown: no basis.

DOMAIN: transportation (roads, bridges, highways, traffic), transit, aviation (airports), water, \
wastewater, stormwater (drainage, flood control), solid_waste, energy_utilities (electric, gas, energy), \
facilities (buildings, HVAC, roofs, fire/police stations, libraries), parks_recreation, public_safety \
(equipment/systems), technology, housing_community, economic_development, environmental, other.

extraction_confidence: high (>=0.85) for a clear table row or project sheet; lower when the entry is \
fragmentary, ambiguous, or may not be a capital project.
"""


def build_user_message(
    chunk: Chunk,
    document: DocumentCandidate,
    organization: str,
    extra_context: str = "",
) -> str:
    coverage = document.coverage_years or "unknown"
    parts = [
        f"ORGANIZATION: {organization}",
        f"DOCUMENT: {document.title or document.url}",
        f"DOCUMENT TYPE: {document.doc_type.value}; STATUS: {document.doc_status.value}; "
        f"FISCAL YEARS COVERED: {coverage}",
    ]
    context = "\n\n".join(p for p in (chunk.context, extra_context) if p)
    if context:
        parts.append(f"CONTEXT (do not extract from here):\n{context}")
    parts.append(f"CONTENT:\n{chunk.text}")
    return "\n\n".join(parts)
