"""Screening taxonomy (spec §6) and the routing policy shared by every screener."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from capex.config import ScreeningConfig
from capex.models import Chunk, DocumentCandidate, ScreeningResult
from capex.normalize import find_id_tokens

QUESTION_VERSION = "screen-v2"

# A. Project presence
PRESENCE: dict[str, str] = {
    "explicit": "Names one or more specific capital projects (construction, improvement, rehabilitation, "
    "replacement, expansion, upgrade, design, acquisition of infrastructure, facilities or equipment), "
    "e.g. rows of a CIP/program table or a project description.",
    "possible": "Probably mentions specific projects or capital work items, but they are not clearly "
    "labelled as capital projects (e.g. 'improvements', 'rehabilitation', 'replacement', a budget line "
    "for a named facility or road segment).",
    "context_only": "Discusses capital programs, funding policy or project-related context without naming "
    "any specific project.",
    "none": "Contains no capital-project information (cover, table of contents, glossary, staff lists, "
    "operating budget narrative, revenue tables).",
}

# B. Content type
CONTENT_TYPES: dict[str, str] = {
    "project_listing": "A table or list with one row/entry per project.",
    "project_description": "A narrative or data sheet describing an individual project.",
    "budget_line_item": "Capital budget line items that name specific projects or facilities.",
    "project_schedule": "Schedule/timeline of projects (years, phases, milestones).",
    "funding_approval": "Approval/authorization of funding for named projects (resolutions, awards).",
    "status_update": "Progress or status reporting on named projects.",
    "financial_summary": "Aggregate financial summary (fund totals, department totals) without itemized projects.",
    "administrative": "Administrative or irrelevant content.",
}

# C. Need for deeper extraction
NEED: dict[str, str] = {
    "extract": "The text itself contains enough information to extract project records.",
    "extract_with_context": "Project information is present but cut off or depends on neighbouring text "
    "(e.g. a table continuing from the previous page, a description whose budget is on the next page).",
    "skip": "Nothing to extract.",
}

_CAPITAL = re.compile(
    r"\b(capital|cip|project|improvements?|rehabilitat\w*|replace\w*|expan\w*|moderni\w*|construct\w*|"
    r"upgrade\w*|design|renovat\w*|widen\w*|reconstruct\w*|install\w*|acqui\w*|infrastructure|facilit\w*|"
    r"pavement|bridge|interchange|pipeline|main|reservoir|pump|station|plant|park|building|roof|hvac|"
    r"chiller|boiler|repair|resurfac\w*|lane|road|street|well|sewer|drainage|signal)\b",
    re.IGNORECASE,
)
_AMOUNT = re.compile(r"\$\s?\d[\d,.]*|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")


def recall_guard(chunk: Chunk) -> bool:
    """Deterministic 'never skip' rule for chunks with strong project-table signals."""
    text = chunk.text
    ids = len(set(find_id_tokens(text)))
    amounts = len(_AMOUNT.findall(text))
    capital = len(_CAPITAL.findall(text))
    return chunk.est_rows >= 4 or (ids >= 2 and amounts >= 2) or (capital >= 6 and amounts >= 4)


def route(
    presence_probs: dict[str, float],
    need_probs: dict[str, float],
    content_type: str,
    chunk: Chunk,
    config: ScreeningConfig,
) -> tuple[str, str, bool, float]:
    """Return (decision, reason, guard_triggered, p_project).

    High recall: a chunk is skipped only when the screener is confident it holds no project
    information *and* the deterministic recall guard does not fire.
    """
    p_project = presence_probs.get("explicit", 0.0) + presence_probs.get("possible", 0.0)
    p_context = need_probs.get("extract_with_context", 0.0)
    p_skip = need_probs.get("skip", 0.0)

    if p_project >= config.extract_threshold or p_skip < 0.5:
        decision = "extract_with_context" if p_context >= config.context_threshold else "extract"
        return decision, f"p_project={p_project:.2f}", False, p_project
    if p_project >= config.plausible_threshold:
        decision = "extract_with_context" if p_context >= config.context_threshold else "extract"
        return decision, f"uncertain but plausible (p_project={p_project:.2f})", False, p_project
    if config.recall_guard and recall_guard(chunk):
        return "extract", f"recall guard (table/ID signals; p_project={p_project:.2f})", True, p_project
    return "skip", f"no project information (p_project={p_project:.2f}, type={content_type})", False, p_project


class Screener(ABC):
    name: str = "abstract"
    model: str = ""

    @abstractmethod
    async def screen(self, chunks: list[Chunk], document: DocumentCandidate, organization: str) -> list[ScreeningResult]:
        ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


def argmax(probs: dict[str, float]) -> str:
    return max(probs.items(), key=lambda kv: kv[1])[0] if probs else ""
