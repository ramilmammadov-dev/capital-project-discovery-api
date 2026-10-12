"""Quality evaluation against manually verified samples (spec §14).

Gold file (YAML), one per organization::

    organization: City of Mesa, Arizona
    result: pilot/city-of-mesa-arizona/result.json
    pages:                                   # manually inventoried sample pages
      - document: "Capital Improvement Program"   # substring of the selected document title (or URL)
        pdf_page: 45
        complete: true                         # every project on the page is listed below
        projects:
          - name: "Fire Station 221 Replacement"
            id: "CP0123"                       # optional
            budget: 5300000                    # optional, USD as printed x unit
            phase: Funded                      # optional — only where the source clearly states the stage
    dedup:                                     # reviewed deduplication decisions
      - names: ["Signal Butte Rd Widening", "Signal Butte Road Widening"]
        same: true

Metrics
* recall            — gold projects found among all records of that document
* precision         — records citing a sample page that match a gold project on it (complete pages only)
* page accuracy     — matched records whose cited PDF page(s) include the gold page
* lifecycle accuracy— matched records whose normalized stage equals the gold stage (where given)
* budget accuracy   — matched records whose budget equals the gold budget (+-0.5%, where given)
* dedup precision/recall — over reviewed pairs: did the system put them in one consolidated project?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz

from capex.evaluation.loader import load_result
from capex.models import ExtractedRecord, JobResult
from capex.normalize import name_key, normalize_project_id


@dataclass
class Score:
    hits: int = 0
    total: int = 0

    @property
    def value(self) -> float | None:
        return round(self.hits / self.total, 4) if self.total else None

    def add(self, ok: bool) -> None:
        self.total += 1
        self.hits += 1 if ok else 0


@dataclass
class OrgEvaluation:
    organization: str
    recall: Score = field(default_factory=Score)
    precision: Score = field(default_factory=Score)
    page_accuracy: Score = field(default_factory=Score)
    lifecycle_accuracy: Score = field(default_factory=Score)
    budget_accuracy: Score = field(default_factory=Score)
    dedup_precision: Score = field(default_factory=Score)
    dedup_recall: Score = field(default_factory=Score)
    misses: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    page_errors: list[dict[str, Any]] = field(default_factory=list)
    phase_errors: list[dict[str, Any]] = field(default_factory=list)
    budget_errors: list[dict[str, Any]] = field(default_factory=list)
    dedup_errors: list[dict[str, Any]] = field(default_factory=list)
    sample_pages: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "organization": self.organization,
            "sample_pages": self.sample_pages,
            "gold_projects": self.recall.total,
            "recall": self.recall.value,
            "precision": self.precision.value,
            "page_reference_accuracy": self.page_accuracy.value,
            "lifecycle_accuracy": self.lifecycle_accuracy.value,
            "budget_accuracy": self.budget_accuracy.value,
            "dedup_precision": self.dedup_precision.value,
            "dedup_recall": self.dedup_recall.value,
            "counts": {
                k: {"hits": v.hits, "total": v.total}
                for k, v in (
                    ("recall", self.recall),
                    ("precision", self.precision),
                    ("page_accuracy", self.page_accuracy),
                    ("lifecycle_accuracy", self.lifecycle_accuracy),
                    ("budget_accuracy", self.budget_accuracy),
                    ("dedup_precision", self.dedup_precision),
                    ("dedup_recall", self.dedup_recall),
                )
            },
            "misses": self.misses,
            "false_positives": self.false_positives,
            "page_errors": self.page_errors,
            "phase_errors": self.phase_errors,
            "budget_errors": self.budget_errors,
            "dedup_errors": self.dedup_errors,
        }


def _matches(gold: dict[str, Any], record: ExtractedRecord) -> float:
    gid = normalize_project_id(str(gold.get("id") or ""))
    if gid and normalize_project_id(record.project_id) == gid:
        return 100.0
    return fuzz.token_set_ratio(name_key(str(gold.get("name", ""))), name_key(record.project_name)) * (
        0.6 + 0.4 * fuzz.token_sort_ratio(name_key(str(gold.get("name", ""))), name_key(record.project_name)) / 100
    )


def _doc_records(result: JobResult, document: str) -> list[ExtractedRecord]:
    needle = document.lower()
    doc_ids = {
        d.doc_id
        for d in result.selected_documents
        if needle in (d.title or "").lower() or needle in (d.final_url or d.url).lower()
    }
    return [r for r in result.records if r.doc_id in doc_ids]


def evaluate_org(gold_path: Path, base_dir: Path | None = None) -> OrgEvaluation:
    gold = yaml.safe_load(gold_path.read_text())
    base_dir = base_dir or gold_path.parent
    result_path = Path(gold["result"])
    if not result_path.is_absolute() and not result_path.exists():
        result_path = (base_dir / result_path).resolve()
    result = load_result(result_path)
    ev = OrgEvaluation(organization=gold.get("organization", result.organization_name))
    threshold = float(gold.get("match_threshold", 80))

    for page in gold.get("pages", []):
        ev.sample_pages += 1
        records = _doc_records(result, str(page["document"]))
        pdf_page = int(page["pdf_page"])
        projects = page.get("projects", [])
        used: set[str] = set()
        for g in projects:
            # prefer records citing this page (or a neighbour), then anywhere in the document
            ranked = sorted(
                records,
                key=lambda r: (_matches(g, r), -min((abs(p - pdf_page) for p in r.source_pages), default=99)),
                reverse=True,
            )
            match = next((r for r in ranked if r.record_id not in used and _matches(g, r) >= threshold), None)
            ev.recall.add(match is not None)
            if match is None:
                ev.misses.append({"page": pdf_page, "gold": g})
                continue
            used.add(match.record_id)
            ok_page = pdf_page in match.source_pages
            ev.page_accuracy.add(ok_page)
            if not ok_page:
                ev.page_errors.append({"gold": g, "gold_page": pdf_page, "cited": match.source_pages, "record": match.record_id})
            if g.get("phase"):
                ok_phase = match.current_phase.value.lower() == str(g["phase"]).lower()
                ev.lifecycle_accuracy.add(ok_phase)
                if not ok_phase:
                    ev.phase_errors.append(
                        {"gold": g, "predicted": match.current_phase.value, "evidence": match.phase_evidence, "record": match.record_id}
                    )
            if g.get("budget") is not None:
                expected = float(g["budget"])
                got = match.estimated_budget
                ok_budget = got is not None and abs(got - expected) <= max(1.0, expected * 0.005)
                ev.budget_accuracy.add(ok_budget)
                if not ok_budget:
                    ev.budget_errors.append({"gold": g, "predicted": got, "budget_text": match.budget_text, "record": match.record_id})
        if page.get("complete", True):
            for r in records:
                if pdf_page not in r.source_pages:
                    continue
                ok = any(_matches(g, r) >= threshold for g in projects)
                ev.precision.add(ok)
                if not ok:
                    ev.false_positives.append({"page": pdf_page, "record": r.record_id, "name": r.project_name})

    # deduplication: reviewed pairs
    project_of: dict[str, str] = {}
    for p in result.projects:
        for rid in p.merged_record_ids:
            project_of[rid] = p.project_uid

    def find(name: str) -> list[ExtractedRecord]:
        key = name_key(name)
        return [r for r in result.records if fuzz.ratio(name_key(r.project_name), key) >= 92]

    for pair in gold.get("dedup", []):
        a_records, b_records = find(pair["names"][0]), find(pair["names"][1])
        if not a_records or not b_records:
            continue
        together = any(project_of.get(a.record_id) == project_of.get(b.record_id) for a in a_records for b in b_records)
        if pair.get("same"):
            ev.dedup_recall.add(together)
        if together:
            ev.dedup_precision.add(bool(pair.get("same")))
        if together != bool(pair.get("same")):
            ev.dedup_errors.append({"names": pair["names"], "gold_same": pair.get("same"), "system_merged": together})
    return ev


def evaluate_all(gold_dir: Path) -> dict[str, Any]:
    orgs = [evaluate_org(p) for p in sorted(gold_dir.glob("*.yaml"))]
    total = OrgEvaluation(organization="ALL")
    for ev in orgs:
        for attr in (
            "recall",
            "precision",
            "page_accuracy",
            "lifecycle_accuracy",
            "budget_accuracy",
            "dedup_precision",
            "dedup_recall",
        ):
            agg: Score = getattr(total, attr)
            part: Score = getattr(ev, attr)
            agg.hits += part.hits
            agg.total += part.total
        total.sample_pages += ev.sample_pages
    summary = total.summary()
    for key in ("misses", "false_positives", "page_errors", "phase_errors", "budget_errors", "dedup_errors"):
        summary.pop(key)
    return {"overall": summary, "organizations": [ev.summary() for ev in orgs]}
