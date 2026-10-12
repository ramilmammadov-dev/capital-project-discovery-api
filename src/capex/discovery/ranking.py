"""Document scoring and top-N selection (spec §4).

Each probed document gets an explainable score (weighted features, all 0-1):

* itemization  — share of sampled pages that list individual projects (0 for aggregate-only reports)
* volume       — estimated number of identifiable projects (log scale)
* authority    — official organization domain > other government domain > third party
* recency      — fiscal-year coverage relative to the current fiscal year
* detail       — budgets, years/dates and project descriptions present
* doc_type     — CIP / capital program / TIP > capital budget > budget book > reports
* relevance    — organization named on the cover pages
* readability  — share of pages with a text layer (reliable page references)

Selection is greedy on *marginal* value: score x (1 - overlap_penalty x estimated overlap with the
documents already selected), so the three picks maximize combined unique project coverage instead
of being three near-identical versions of one program.
"""

from __future__ import annotations

import math

from capex.config import DiscoveryConfig
from capex.discovery.classify import current_fiscal_year
from capex.discovery.probe import overlap_estimate
from capex.models import DocStatus, DocType, DocumentCandidate

_DOC_TYPE_VALUE = {
    DocType.CIP_PROGRAM: 1.0,
    DocType.CIP_PLAN: 1.0,
    DocType.MULTI_YEAR_PROGRAM: 1.0,
    DocType.TIP: 0.95,
    DocType.CAPITAL_BUDGET: 0.85,
    DocType.FACILITY_PROGRAM: 0.85,
    DocType.INFRASTRUCTURE_PLAN: 0.8,
    DocType.ADOPTED_BUDGET: 0.65,
    DocType.PROPOSED_BUDGET: 0.6,
    DocType.BOARD_SCHEDULE: 0.6,
    DocType.CAPITAL_REPORT: 0.4,
    DocType.OTHER: 0.2,
}


def _recency(candidate: DocumentCandidate, fiscal_year: int) -> float:
    years = [y for y in (candidate.coverage_end_year, candidate.coverage_start_year) if y]
    if not years and candidate.probe and candidate.probe.fiscal_years:
        years = [max(candidate.probe.fiscal_years)]
    if not years:
        return 0.3
    end = max(years)
    start = min(years)
    if start <= fiscal_year <= end or start == fiscal_year + 1:
        return 1.0
    if end >= fiscal_year:
        return 0.9
    return max(0.0, 0.75 - 0.25 * (fiscal_year - end - 1))


def score_candidate(candidate: DocumentCandidate, config: DiscoveryConfig, fiscal_year: int | None = None) -> float:
    fiscal_year = fiscal_year or current_fiscal_year()
    probe = candidate.probe
    features: dict[str, float] = {}
    if probe is None or not probe.ok:
        candidate.score = 0.0
        candidate.score_breakdown = {}
        return 0.0
    features["itemization"] = 0.0 if probe.aggregate_only else min(1.0, probe.itemized_page_ratio / 0.5)
    features["volume"] = min(1.0, math.log10(1 + probe.est_projects) / math.log10(1 + 400))
    if candidate.is_official_domain:
        features["authority"] = 1.0
    elif candidate.is_mirror_of_official:
        features["authority"] = 0.8
    else:
        features["authority"] = 0.6 if candidate.domain.endswith((".gov", ".us")) else 0.2
    features["recency"] = _recency(candidate, fiscal_year)
    features["detail"] = 0.4 * probe.has_budgets + 0.3 * probe.has_years + 0.3 * probe.has_descriptions
    features["doc_type"] = _DOC_TYPE_VALUE.get(candidate.doc_type, 0.2)
    features["relevance"] = 1.0 if candidate.is_official_domain else min(1.0, probe.org_mentions / 3)
    features["readability"] = probe.text_page_ratio
    weights = config.weights
    total = sum(weights.get(k, 0.0) * v for k, v in features.items())
    total /= max(1e-9, sum(weights.get(k, 0.0) for k in features))
    if "cover suggests a different organization" in candidate.notes:
        total *= 0.5
    candidate.score_breakdown = {k: round(v, 3) for k, v in features.items()}
    candidate.score = round(total, 4)
    return candidate.score


def _rationale(candidate: DocumentCandidate, overlap: float, rank: int) -> str:
    probe = candidate.probe
    assert probe is not None
    parts = [
        f"#{rank}: score {candidate.score:.2f}",
        f"{candidate.doc_type.value}",
        f"status {candidate.doc_status.value}",
    ]
    if candidate.coverage_years:
        parts.append(f"covers {candidate.coverage_years}")
    parts.append(
        f"~{probe.est_projects} itemized projects estimated ({probe.itemized_page_ratio:.0%} of sampled pages list projects)"
    )
    detail = [n for n, ok in (("budgets", probe.has_budgets), ("years", probe.has_years), ("descriptions", probe.has_descriptions)) if ok]
    if detail:
        parts.append("has " + ", ".join(detail))
    if candidate.is_official_domain:
        parts.append("official source" + (" (read from Internet Archive capture)" if candidate.archive_url else ""))
    elif candidate.is_mirror_of_official:
        parts.append(f"copy of the organization's own document hosted on {candidate.domain}")
    else:
        parts.append(f"source domain {candidate.domain}")
    if rank > 1:
        parts.append(f"estimated overlap with documents already selected {overlap:.0%}")
    return "; ".join(parts)


def select_documents(
    candidates: list[DocumentCandidate], config: DiscoveryConfig, top_n: int | None = None
) -> list[DocumentCandidate]:
    top_n = top_n or config.select_top
    fiscal_year = current_fiscal_year()
    for candidate in candidates:
        score_candidate(candidate, config, fiscal_year)
    eligible = [
        c
        for c in candidates
        if c.probe and c.probe.ok and c.file_type in ("pdf", "html") and (c.score or 0) > 0
    ]
    itemized = [c for c in eligible if c.contains_itemized_projects]
    pool = itemized if len(itemized) >= top_n else eligible
    selected: list[DocumentCandidate] = []
    while pool and len(selected) < top_n:
        best, best_gain, best_overlap = None, -1.0, 0.0
        for c in pool:
            overlap = max((overlap_estimate(c, s) for s in selected), default=0.0)
            gain = (c.score or 0.0) * (1 - config.overlap_penalty * overlap)
            if gain > best_gain:
                best, best_gain, best_overlap = c, gain, overlap
        assert best is not None
        best.selected = True
        best.selection_rank = len(selected) + 1
        best.selection_rationale = _rationale(best, best_overlap, best.selection_rank)
        selected.append(best)
        pool.remove(best)

    for c in candidates:
        if c.selected:
            continue
        if not c.probe:
            c.rejection_reason = c.rejection_reason or "not probed (lower preliminary relevance score)"
        elif not c.probe.ok:
            c.rejection_reason = f"could not be processed: {c.probe.error}"
        elif c.file_type not in ("pdf", "html"):
            c.rejection_reason = f"unsupported format ({c.file_type})"
        elif c.probe.aggregate_only or not c.contains_itemized_projects:
            c.rejection_reason = "no itemized project listings detected (aggregate/summary content)"
        else:
            overlap = max((overlap_estimate(c, s) for s in selected), default=0.0)
            if overlap >= 0.6:
                twin = max(selected, key=lambda s: overlap_estimate(c, s))
                c.rejection_reason = f"overlaps ~{overlap:.0%} with selected #{twin.selection_rank} ({twin.title[:60]})"
            else:
                c.rejection_reason = f"lower marginal value (score {c.score:.2f})"
    return selected


def doc_status_label(candidate: DocumentCandidate) -> str:
    return candidate.doc_status.value if candidate.doc_status != DocStatus.UNKNOWN else "Unknown"
