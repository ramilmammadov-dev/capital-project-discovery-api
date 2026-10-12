"""Zero-cost fallback screeners.

* ``HeuristicScreener`` — lexical/structural features -> pseudo-probabilities in the same shape Jev
  returns, routed by the same policy. Used automatically when no Jev key is configured (and
  reported as such in the metrics), and as a free baseline in the screening evaluation.
* ``PassThroughScreener`` — sends every chunk to extraction ("process all chunks directly"); used
  to measure the recall/cost trade-off of screening (spec §12).
"""

from __future__ import annotations

import math
import re

from capex.config import ScreeningConfig
from capex.models import Chunk, DocumentCandidate, ScreeningResult
from capex.normalize import find_id_tokens
from capex.screening.base import Screener, route

_AMOUNT = re.compile(r"\$\s?\d[\d,.]*|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_CAPITAL = re.compile(
    r"\b(capital|cip|project|improvements?|rehabilitat\w*|replace\w*|expan\w*|moderni\w*|construct\w*|"
    r"upgrade\w*|design|renovat\w*|widen\w*|reconstruct\w*|install\w*|acqui\w*|infrastructure|facilit\w*|"
    r"pavement|bridge|interchange|pipeline|reservoir|pump|station|plant|park|building|roof|hvac|"
    r"chiller|boiler|repair|resurfac\w*|sewer|drainage|signal|well|main)\b",
    re.IGNORECASE,
)
_ADMIN = re.compile(
    r"\b(table of contents|glossary|acronyms?|mission statement|vision|organizational chart|acknowledg\w*|"
    r"message from|letter from|index|staff directory|definitions|elected officials|council members|"
    r"distinguished budget|gfoa|award)\b",
    re.IGNORECASE,
)
_SUMMARY = re.compile(
    r"\b(revenues?|fund balance|personnel|full[- ]time equivalent|fte|operating budget|debt service|"
    r"transfers? (?:in|out)|property tax|sales tax|expenditures by)\b",
    re.IGNORECASE,
)
_PROJECT_LABELS = re.compile(
    r"\b(project (?:name|number|no\.?|id|description|title|location|scope)|location:|scope:|description:|"
    r"total project cost|estimated cost|construction start|completion)\b",
    re.IGNORECASE,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def heuristic_probabilities(chunk: Chunk) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    text = chunk.text
    words = max(1, len(text.split()))
    ids = len(set(find_id_tokens(text)))
    amounts = len(_AMOUNT.findall(text))
    capital = len(_CAPITAL.findall(text))
    admin = len(_ADMIN.findall(text))
    summary = len(_SUMMARY.findall(text))
    labels = len(_PROJECT_LABELS.findall(text))
    rows = chunk.est_rows

    z = (
        -2.2
        + 0.35 * min(rows, 12)
        + 0.30 * min(ids, 10)
        + 0.06 * min(amounts, 30)
        + 0.9 * min(capital / max(1.0, words / 120), 4)
        + 0.6 * min(labels, 4)
        - 0.9 * min(admin, 4)
        - 0.25 * min(summary, 8)
    )
    p_explicit = _sigmoid(z)
    p_possible = (1 - p_explicit) * _sigmoid(z + 1.6) * 0.8
    p_context = (1 - p_explicit - p_possible) * (0.7 if capital else 0.3)
    p_none = max(0.0, 1 - p_explicit - p_possible - p_context)
    presence = {"explicit": p_explicit, "possible": p_possible, "context_only": p_context, "none": p_none}

    if rows >= 3 or (ids >= 3 and amounts >= 3):
        ctype = "project_listing"
    elif labels >= 2:
        ctype = "project_description"
    elif admin >= 2:
        ctype = "administrative"
    elif summary >= 3 and ids == 0:
        ctype = "financial_summary"
    elif amounts >= 2 and capital >= 2:
        ctype = "budget_line_item"
    else:
        ctype = "administrative" if capital == 0 else "status_update"
    content = {k: (0.7 if k == ctype else 0.3 / 7) for k in (
        "project_listing",
        "project_description",
        "budget_line_item",
        "project_schedule",
        "funding_approval",
        "status_update",
        "financial_summary",
        "administrative",
    )}
    p_project = p_explicit + p_possible
    need = {
        "extract": p_project * 0.85,
        "extract_with_context": p_project * 0.15,
        "skip": 1 - p_project,
    }
    return presence, content, need


class HeuristicScreener(Screener):
    name = "heuristic"
    model = "heuristic-v1"

    def __init__(self, config: ScreeningConfig) -> None:
        self.cfg = config

    async def screen(self, chunks: list[Chunk], document: DocumentCandidate, organization: str) -> list[ScreeningResult]:
        out = []
        for chunk in chunks:
            presence, content, need = heuristic_probabilities(chunk)
            ctype = max(content.items(), key=lambda kv: kv[1])[0]
            decision, reason, guard, p_project = route(presence, need, ctype, chunk, self.cfg)
            out.append(
                ScreeningResult(
                    chunk_id=chunk.chunk_id,
                    provider=self.name,
                    model=self.model,
                    presence=max(presence.items(), key=lambda kv: kv[1])[0],
                    presence_probs={k: round(v, 4) for k, v in presence.items()},
                    content_type=ctype,
                    content_type_probs=content,
                    need=max(need.items(), key=lambda kv: kv[1])[0],
                    need_probs={k: round(v, 4) for k, v in need.items()},
                    p_project=round(p_project, 4),
                    decision=decision,  # type: ignore[arg-type]
                    reason=reason,
                    guard=guard,
                )
            )
        return out


class PassThroughScreener(Screener):
    name = "none"
    model = "none"

    async def screen(self, chunks: list[Chunk], document: DocumentCandidate, organization: str) -> list[ScreeningResult]:
        return [
            ScreeningResult(
                chunk_id=chunk.chunk_id,
                provider=self.name,
                model=self.model,
                presence="unscreened",
                content_type="unscreened",
                need="extract",
                p_project=1.0,
                decision="extract",
                reason="screening disabled (all chunks extracted)",
            )
            for chunk in chunks
        ]
