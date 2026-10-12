"""Deduplication & consolidation (spec §9).

Signals: official identifiers, name similarity (after abbreviation normalization), numbers and
phase/segment markers inside names, locations, budgets, descriptions, domains and timeline.

Decision policy — deterministic and conservative (false merges are treated as the worst error):

* hard *cannot-link*: different identifiers of the same ID scheme, different numbers in the name
  (7th Ave vs 19th Ave, Phase 1 vs Phase 2), different phase/segment markers, different budgets for
  two rows of the same chunk;
* *merge*: same identifier with compatible names; or near-identical names (>= ``name_auto_merge``);
  or similar names (>= ``name_corroborated_merge``) corroborated by location/budget/description;
* *possible duplicate*: anything else above ``name_possible`` — flagged for review, never merged.

Clustering is a constrained union-find: two clusters are joined only if no cross pair is a
cannot-link, which prevents transitive chains from silently merging distinct projects.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz, process

from capex.config import DedupConfig
from capex.models import (
    PHASE_ORDER,
    ConsolidatedProject,
    DocumentCandidate,
    ExtractedRecord,
    FundingStatus,
    MergeAuditEntry,
    Phase,
    SourceReference,
)
from capex.normalize import (
    id_shape,
    name_key,
    name_numbers,
    normalize_project_id,
    page_link,
    phase_markers,
    short_hash,
)


@dataclass
class _Rec:
    index: int
    record: ExtractedRecord
    id_norm: str | None
    id_shape: str | None
    key: str
    numbers: frozenset[str]
    markers: frozenset[str]
    location: str
    recency: tuple[int, int, float]


@dataclass
class _Decision:
    decision: str  # merge | possible | distinct
    rule: str
    score: float
    hard: bool = False
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationResult:
    projects: list[ConsolidatedProject]
    audit: list[MergeAuditEntry]
    raw_records: int
    duplicates_merged: int
    possible_duplicates: int


def _doc_recency(doc: DocumentCandidate | None) -> tuple[int, int]:
    if doc is None:
        return (0, 99)
    year = doc.coverage_start_year or doc.document_fiscal_year or doc.coverage_end_year or 0
    return (year, -(doc.selection_rank or 99))


def _prepare(records: list[ExtractedRecord], documents: dict[str, DocumentCandidate]) -> list[_Rec]:
    out = []
    for i, r in enumerate(records):
        nid = normalize_project_id(r.project_id)
        doc_year, doc_rank = _doc_recency(documents.get(r.doc_id))
        out.append(
            _Rec(
                index=i,
                record=r,
                id_norm=nid,
                id_shape=id_shape(nid) if nid else None,
                key=name_key(r.project_name),
                numbers=frozenset(name_numbers(r.project_name)),
                markers=frozenset(phase_markers(r.project_name)),
                location=name_key(r.location) if r.location else "",
                recency=(doc_year, doc_rank, r.extraction_confidence),
            )
        )
    return out


def _budget_close(a: float | None, b: float | None, tolerance: float) -> bool | None:
    if not a or not b:
        return None
    return abs(a - b) / max(a, b) <= tolerance


def _cannot_link(a: _Rec, b: _Rec) -> str | None:
    if a.id_norm and b.id_norm:
        if a.id_norm == b.id_norm:
            return None
        if a.id_shape == b.id_shape:
            return "different_project_ids"
    if a.numbers and b.numbers and a.numbers != b.numbers:
        return "different_numbers_in_name"
    if a.markers != b.markers and (a.markers or b.markers):
        kinds_a = {m.split(":")[0] for m in a.markers}
        kinds_b = {m.split(":")[0] for m in b.markers}
        if kinds_a & kinds_b or (a.markers and b.markers):
            return "different_phase_or_component"
    if a.record.chunk_id == b.record.chunk_id and a.record.estimated_budget and b.record.estimated_budget:
        if a.record.estimated_budget != b.record.estimated_budget:
            return "separate_rows_same_table"
    return None


def _decide(a: _Rec, b: _Rec, cfg: DedupConfig) -> _Decision:
    sort_score = fuzz.token_sort_ratio(a.key, b.key)
    set_score = fuzz.token_set_ratio(a.key, b.key)
    name_score = round(0.6 * sort_score + 0.4 * set_score, 1)
    signals: dict[str, Any] = {"name_score": name_score}

    if a.id_norm and b.id_norm and a.id_norm == b.id_norm:
        signals["project_id"] = a.id_norm
        if name_score >= cfg.id_name_min or a.key in b.key or b.key in a.key:
            return _Decision("merge", "same_project_id", 100.0, signals=signals)
        return _Decision("possible", "same_project_id_but_different_names", name_score, signals=signals)

    hard = _cannot_link(a, b)
    if hard:
        signals["blocked_by"] = hard
        return _Decision("distinct", hard, name_score, hard=True, signals=signals)
    if a.markers != b.markers:
        # "Bridge X" vs "Bridge X Phase 2": a component vs the whole — never merged automatically
        signals["phase_marker_one_side"] = sorted(a.markers | b.markers)
        if name_score >= cfg.name_possible:
            return _Decision("possible", "component_vs_parent_project", name_score, signals=signals)
        return _Decision("distinct", "dissimilar", name_score, signals=signals)

    corroboration = 0
    negatives = 0
    if a.location and b.location:
        loc = fuzz.token_set_ratio(a.location, b.location)
        signals["location_score"] = loc
        if loc >= cfg.location_min:
            corroboration += 1
        elif loc < 40:
            negatives += 1
    budget = _budget_close(a.record.estimated_budget, b.record.estimated_budget, cfg.budget_tolerance)
    if budget is not None:
        signals["budget_close"] = budget
        corroboration += 1 if budget else 0
    da, db = a.record.project_description or "", b.record.project_description or ""
    if da and db:
        desc = fuzz.token_set_ratio(da.lower(), db.lower())
        signals["description_score"] = desc
        corroboration += 1 if desc >= 75 else 0
    if a.record.domain != b.record.domain and "other" not in (a.record.domain, b.record.domain):
        signals["domain_mismatch"] = True
        negatives += 1
    signals["corroboration"] = corroboration

    if name_score >= cfg.name_auto_merge and negatives == 0:
        return _Decision("merge", "near_identical_name", name_score, signals=signals)
    if name_score >= cfg.name_corroborated_merge and corroboration >= 1 and negatives == 0:
        return _Decision("merge", "similar_name_corroborated", name_score, signals=signals)
    if name_score >= cfg.name_possible:
        return _Decision("possible", "similar_name_uncorroborated", name_score, signals=signals)
    return _Decision("distinct", "dissimilar", name_score, signals=signals)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.members: dict[int, list[int]] = {i: [i] for i in range(n)}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].extend(self.members.pop(rb))
        return ra


def _candidate_pairs(recs: list[_Rec], cfg: DedupConfig) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    by_id: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        if r.id_norm:
            by_id[r.id_norm].append(r.index)
    for members in by_id.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                pairs.add((members[x], members[y]))
    keys = [r.key for r in recs]
    for r in recs:
        if not r.key:
            continue
        for _, _score, j in process.extract(
            r.key, keys, scorer=fuzz.token_set_ratio, score_cutoff=cfg.name_possible, limit=None
        ):
            if j > r.index:
                pairs.add((r.index, j))
    return pairs


def _phase_rank(phase: Phase) -> int:
    return PHASE_ORDER.get(phase, 0)


def consolidate(
    records: list[ExtractedRecord],
    documents: dict[str, DocumentCandidate],
    cfg: DedupConfig,
    org_key: str,
    uid_resolver: Callable[[list[str], str], str] | None = None,
) -> ConsolidationResult:
    recs = _prepare(records, documents)
    decisions: dict[tuple[int, int], _Decision] = {}
    for i, j in _candidate_pairs(recs, cfg):
        decisions[(i, j)] = _decide(recs[i], recs[j], cfg)

    uf = _UnionFind(len(recs))
    audit: list[MergeAuditEntry] = []
    merges = sorted(((d.score, pair) for pair, d in decisions.items() if d.decision == "merge"), reverse=True)
    blocked: list[tuple[tuple[int, int], str]] = []
    for _, (i, j) in merges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        conflict = None
        for x in uf.members[ri]:
            for y in uf.members[rj]:
                reason = _cannot_link(recs[x], recs[y])
                if reason:
                    conflict = reason
                    break
            if conflict:
                break
        if conflict:
            blocked.append(((i, j), conflict))
            continue
        uf.union(i, j)

    clusters = [sorted(members) for members in uf.members.values()]
    projects: list[ConsolidatedProject] = []
    cluster_of: dict[int, int] = {}
    for c_index, members in enumerate(clusters):
        for m in members:
            cluster_of[m] = c_index

    assigned: set[str] = set()
    for members in clusters:
        project = _merge_cluster([recs[m] for m in members], documents, org_key, uid_resolver, assigned)
        assigned.add(project.project_uid)
        projects.append(project)
        if len(members) > 1:
            pair_rules = [
                decisions[(min(x, y), max(x, y))].rule
                for x in members
                for y in members
                if x < y and (x, y) in decisions and decisions[(x, y)].decision == "merge"
            ]
            audit.append(
                MergeAuditEntry(
                    decision="merged",
                    project_uid=project.project_uid,
                    record_ids=[recs[m].record.record_id for m in members],
                    names=sorted({recs[m].record.project_name for m in members}),
                    rule=", ".join(sorted(set(pair_rules))) or "merged",
                    score=max(
                        (decisions[(x, y)].score for x in members for y in members if (x, y) in decisions),
                        default=100.0,
                    ),
                    signals={"records": len(members), "documents": len({recs[m].record.doc_id for m in members})},
                )
            )

    # uncertain possible duplicates between different final projects
    possible_pairs: set[tuple[int, int]] = set()
    for (i, j), d in decisions.items():
        ci, cj = cluster_of[i], cluster_of[j]
        if ci == cj or d.decision != "possible":
            continue
        key = (min(ci, cj), max(ci, cj))
        if key in possible_pairs:
            continue
        possible_pairs.add(key)
        pa, pb = projects[ci], projects[cj]
        pa.possible_duplicates.append(pb.project_uid)
        pb.possible_duplicates.append(pa.project_uid)
        for p in (pa, pb):
            if "possible duplicate" not in p.review_reasons:
                p.review_reasons.append("possible duplicate")
                p.needs_review = True
        audit.append(
            MergeAuditEntry(
                decision="possible_duplicate",
                project_uid=pa.project_uid,
                other_project_uid=pb.project_uid,
                record_ids=[recs[i].record.record_id, recs[j].record.record_id],
                names=[recs[i].record.project_name, recs[j].record.project_name],
                rule=d.rule,
                score=d.score,
                signals=d.signals,
            )
        )
    for (i, j), reason in blocked:
        ci, cj = cluster_of[i], cluster_of[j]
        if ci == cj:
            continue
        key = (min(ci, cj), max(ci, cj))
        if key in possible_pairs:
            continue
        possible_pairs.add(key)
        audit.append(
            MergeAuditEntry(
                decision="possible_duplicate",
                project_uid=projects[ci].project_uid,
                other_project_uid=projects[cj].project_uid,
                record_ids=[recs[i].record.record_id, recs[j].record.record_id],
                names=[recs[i].record.project_name, recs[j].record.project_name],
                rule=f"merge_blocked_by_{reason}",
                score=decisions[(i, j)].score,
                signals=decisions[(i, j)].signals,
            )
        )

    projects.sort(key=lambda p: (p.domain, p.project_name.lower()))
    return ConsolidationResult(
        projects=projects,
        audit=audit,
        raw_records=len(records),
        duplicates_merged=len(records) - len(projects),
        possible_duplicates=len(possible_pairs),
    )


def _merge_cluster(
    members: list[_Rec],
    documents: dict[str, DocumentCandidate],
    org_key: str,
    uid_resolver: Callable[[list[str], str], str] | None,
    assigned: set[str],
) -> ConsolidatedProject:
    # newest document first, then better-ranked document, then higher confidence
    ordered = sorted(members, key=lambda r: r.recency, reverse=True)
    recs = [r.record for r in ordered]
    primary = recs[0]

    def first(attr: str) -> Any:
        for r in recs:
            value = getattr(r, attr)
            if value not in (None, "", []):
                return value
        return None

    # identity & stable UID
    ids = [r.id_norm for r in ordered if r.id_norm]
    alias_keys = sorted({f"id:{i}" for i in ids}) or sorted({f"name:{r.key}|loc:{r.location}" for r in ordered})
    default_uid = "PRJ-" + short_hash(org_key, alias_keys[0], length=10).upper()
    uid = uid_resolver(alias_keys, default_uid) if uid_resolver else default_uid
    if uid in assigned:  # never let two projects of one run share an ID
        uid = "PRJ-" + short_hash(org_key, *sorted(r.record_id for r in recs), length=10).upper()

    names = []
    for r in recs:
        if r.project_name not in names:
            names.append(r.project_name)
    descriptions = [r.project_description for r in recs if r.project_description]
    domains = Counter(r.domain for r in recs if r.domain != "other")
    domain = domains.most_common(1)[0][0] if domains else primary.domain

    budget_rec = next((r for r in recs if r.estimated_budget), None)
    conflicts = []
    if budget_rec and budget_rec.estimated_budget:
        latest = budget_rec.estimated_budget
        for r in recs:
            if r.estimated_budget and r is not budget_rec and r.doc_id != budget_rec.doc_id:
                if abs(r.estimated_budget - latest) / max(r.estimated_budget, latest) > 0.05:
                    conflicts.append(
                        {
                            "document": r.source_document,
                            "estimated_budget": r.estimated_budget,
                            "budget_text": r.budget_text,
                            "budget_basis": r.budget_basis,
                            "pages": r.source_pages,
                        }
                    )

    # current phase: newest credible evidence; explicit statements beat inferred defaults
    phase_rec = next((r for r in recs if r.phase_basis == "explicit" and r.current_phase != Phase.UNKNOWN), None)
    if phase_rec is None:
        phase_rec = next((r for r in recs if r.current_phase != Phase.UNKNOWN), primary)
    history = []
    seen_docs = set()
    for r in recs:
        if r.doc_id in seen_docs:
            continue
        seen_docs.add(r.doc_id)
        history.append(
            {
                "document": r.source_document,
                "document_fiscal_year": r.document_fiscal_year,
                "phase": r.current_phase.value,
                "basis": r.phase_basis,
                "evidence": r.phase_evidence or r.status_text,
            }
        )
    funding_rec = next((r for r in recs if r.funding_status != FundingStatus.UNKNOWN), primary)

    references = []
    for r in recs:
        doc = documents.get(r.doc_id)
        is_pdf = (doc.file_type == "pdf") if doc else r.source_url.lower().split("?")[0].endswith(".pdf")
        references.append(
            SourceReference(
                record_id=r.record_id,
                doc_id=r.doc_id,
                document_title=r.source_document,
                url=r.source_url,
                page_url=page_link(r.source_url, r.source_pages[0] if r.source_pages else None, is_pdf),
                pdf_pages=r.source_pages,
                printed_pages=r.printed_pages,
                section=r.source_section,
                excerpt=r.source_excerpt,
                original_project_name=r.project_name,
                original_project_id=r.project_id,
                document_fiscal_year=r.document_fiscal_year,
                estimated_budget=r.estimated_budget,
                current_phase=r.current_phase,
                verified=r.excerpt_verified and r.name_verified,
            )
        )

    confidence = max(r.extraction_confidence for r in recs)
    review_reasons = []
    if all(r.needs_review for r in recs):
        review_reasons.extend(sorted({f for r in recs for f in r.flags if not f.startswith("phase_rule")})[:4] or ["low confidence"])
    if conflicts:
        review_reasons.append("conflicting budgets across sources")
    if len({r.id_norm for r in ordered if r.id_norm}) > 1:
        review_reasons.append("multiple identifier schemes")

    advanced = max(recs, key=lambda r: _phase_rank(r.current_phase))
    if _phase_rank(advanced.current_phase) > _phase_rank(phase_rec.current_phase) and advanced.phase_basis == "explicit":
        review_reasons.append(f"older source reports a later stage ({advanced.current_phase.value})")

    return ConsolidatedProject(
        project_uid=uid,
        organization_name=primary.organization_name,
        project_id=primary.project_id or first("project_id"),
        project_name=primary.project_name,
        aliases=names[1:],
        project_description=max(descriptions, key=len) if descriptions else None,
        domain=domain,
        project_type=first("project_type"),
        location=first("location"),
        estimated_budget=budget_rec.estimated_budget if budget_rec else None,
        budget_type=budget_rec.budget_type if budget_rec else "unknown",
        budget_basis=budget_rec.budget_basis if budget_rec else None,
        budget_source=budget_rec.source_document if budget_rec else None,
        budget_conflicts=conflicts,
        funding_by_year=(budget_rec.funding_by_year if budget_rec and budget_rec.funding_by_year else first("funding_by_year") or []),
        funding_sources=first("funding_sources"),
        document_fiscal_year=primary.document_fiscal_year,
        anticipated_start_year=first("anticipated_start_year"),
        anticipated_construction_year=first("anticipated_construction_year"),
        anticipated_completion_year=first("anticipated_completion_year"),
        current_phase=phase_rec.current_phase,
        phase_evidence=phase_rec.phase_evidence or phase_rec.status_text,
        phase_basis=phase_rec.phase_basis,
        phase_history=history if len(history) > 1 else [],
        funding_status=funding_rec.funding_status,
        funding_evidence=funding_rec.funding_evidence,
        source_document=primary.source_document,
        source_url=primary.source_url,
        source_pages=primary.source_pages,
        source_section=primary.source_section,
        source_excerpt=primary.source_excerpt,
        references=references,
        extraction_confidence=confidence,
        needs_review=bool(review_reasons),
        review_reasons=review_reasons,
        merged_record_ids=[r.record_id for r in recs],
    )
