"""Consolidation / deduplication (src/capex/consolidation/dedup.py)."""

from __future__ import annotations

import random

import pytest

from capex.config import DedupConfig
from capex.consolidation.dedup import consolidate
from capex.models import DocumentCandidate, ExtractedRecord, FundingStatus, Phase

CFG = DedupConfig()
ORG = "examplecity"


def doc(doc_id: str, year: int, rank: int | None = None, pdf: bool = True) -> DocumentCandidate:
    return DocumentCandidate(
        doc_id=doc_id, url=f"https://www.examplecity.gov/{doc_id}.{'pdf' if pdf else 'html'}", title=f"CIP {year}",
        file_type="pdf" if pdf else "html", document_fiscal_year=year, coverage_start_year=year, selection_rank=rank,
    )


DOCS = {"doc_new": doc("doc_new", 2027, 1), "doc_old": doc("doc_old", 2024, 2), "doc_c": doc("doc_c", 2025, 3)}


def rec(n: int, name: str, *, pid: str | None = None, doc_id: str = "doc_new", budget: float | None = None,
        location: str | None = None, phase: Phase = Phase.FUNDED, explicit: bool = False, chunk: str | None = None,
        domain: str = "transportation", description: str | None = None, pages=(4,), **extra) -> ExtractedRecord:
    return ExtractedRecord(
        record_id=f"{doc_id}:{n:04d}#000", organization_name="City of Exampleville", project_id=pid,
        project_name=name, project_description=description, domain=domain, location=location,
        estimated_budget=budget, budget_text=f"${budget:,.0f}" if budget else None,
        current_phase=phase, phase_basis="explicit" if explicit else "inferred",
        source_document=DOCS[doc_id].title, source_url=DOCS[doc_id].url, source_pages=list(pages),
        source_excerpt=name, doc_id=doc_id, chunk_id=chunk or f"{doc_id}:{n:04d}", extraction_confidence=0.9,
        excerpt_verified=True, name_verified=True, **extra,
    )


def run(records, **kw):
    return consolidate(records, DOCS, CFG, ORG, **kw)


class TestSameIdAcrossDocuments:
    def test_same_project_id_merges_keeps_all_references_and_newest_phase_wins(self):
        old = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old", phase=Phase.DESIGN,
                  budget=900_000, pages=(12,))
        new = rec(1, "Harbor Rd Resurfacing Project", pid="ST85100001", doc_id="doc_new", phase=Phase.CONSTRUCTION,
                  budget=910_000, pages=(31,))
        result = run([old, new])
        assert len(result.projects) == 1 and result.duplicates_merged == 1 and result.possible_duplicates == 0
        project = result.projects[0]
        assert project.current_phase == Phase.CONSTRUCTION  # newest document wins
        assert project.project_name == "Harbor Rd Resurfacing Project"
        assert project.aliases == ["Harbor Road Resurfacing"]
        assert project.estimated_budget == 910_000
        assert {r.doc_id for r in project.references} == {"doc_old", "doc_new"}
        assert len(project.references) == 2
        by_doc = {r.doc_id: r for r in project.references}
        assert by_doc["doc_new"].page_url.endswith("/doc_new.pdf#page=31")
        assert by_doc["doc_old"].page_url.endswith("/doc_old.pdf#page=12")
        assert sorted(project.merged_record_ids) == sorted([old.record_id, new.record_id])
        assert [h["phase"] for h in project.phase_history] == ["Construction / In Progress", "Design"]
        assert result.audit[0].decision == "merged" and "same_project_id" in result.audit[0].rule
        assert result.raw_records == 2

    def test_explicit_older_statement_beats_an_inferred_newer_default(self):
        old = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old", phase=Phase.DESIGN, explicit=True)
        new = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_new", phase=Phase.FUNDED)
        project = run([old, new]).projects[0]
        assert project.current_phase == Phase.DESIGN and project.phase_basis == "explicit"

    def test_funding_status_comes_from_the_best_record(self):
        old = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old", funding_status=FundingStatus.FUNDED)
        new = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_new")
        assert run([old, new]).projects[0].funding_status == FundingStatus.FUNDED

    def test_same_id_with_unrelated_names_is_only_a_possible_duplicate(self):
        a = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old")
        b = rec(1, "Citywide Library Roof Replacement", pid="ST85100001", doc_id="doc_new")
        result = run([a, b])
        assert len(result.projects) == 2 and result.possible_duplicates == 1


class TestNeverMerge:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("7th Ave Widening", "19th Ave Widening"),
            ("Bridge X Phase 1", "Bridge X Phase 2"),
            ("Bridge X Phase I", "Bridge X Phase II"),
            ("Sewer Rehabilitation Segment A", "Sewer Rehabilitation Segment B"),
            ("Water Main Replacement Package 3", "Water Main Replacement Package 4"),
        ],
    )
    def test_distinct_numbers_or_components(self, a, b):
        records = [rec(1, a, doc_id="doc_old", budget=1_000_000, location="Zone A"),
                   rec(1, b, doc_id="doc_new", budget=1_000_000, location="Zone A")]
        result = run(records)
        assert len(result.projects) == 2
        assert result.duplicates_merged == 0
        assert not any(e.decision == "merged" for e in result.audit)

    def test_component_versus_parent_is_never_auto_merged(self):
        records = [rec(1, "Bridge X", doc_id="doc_old"), rec(1, "Bridge X Phase 2", doc_id="doc_new")]
        assert len(run(records).projects) == 2

    def test_different_ids_with_the_same_id_shape_never_merge(self):
        records = [rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old"),
                   rec(1, "Harbor Road Resurfacing", pid="ST85100002", doc_id="doc_new")]
        result = run(records)
        assert len(result.projects) == 2
        assert {p.project_id for p in result.projects} == {"ST85100001", "ST85100002"}

    def test_identical_names_with_a_different_id_scheme_merge_but_are_flagged(self):
        records = [rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old"),
                   rec(1, "Harbor Road Resurfacing", pid="H898001C", doc_id="doc_new")]
        project = run(records).projects
        assert len(project) == 1
        assert "multiple identifier schemes" in project[0].review_reasons and project[0].needs_review

    def test_two_rows_of_one_table_with_different_budgets_stay_separate(self):
        records = [rec(1, "Elm Street Repaving", budget=100_000, chunk="doc_new:0001"),
                   rec(2, "Elm Street Repaving", budget=250_000, chunk="doc_new:0001")]
        assert len(run(records).projects) == 2

    def test_transitive_chain_with_conflicting_ends_is_not_fully_merged(self):
        a = rec(1, "Cedar Avenue Widening", pid="ST85100001", doc_id="doc_old")
        b = rec(1, "Cedar Avenue Widening", doc_id="doc_c")
        c = rec(1, "Cedar Avenue Widening", pid="ST85100002", doc_id="doc_new")
        result = run([a, b, c])
        assert len(result.projects) == 2
        ids = sorted(p.project_id or "" for p in result.projects)
        assert ids[0] == "ST85100001" or ids[1] == "ST85100002"
        assert {p.project_id for p in result.projects} == {"ST85100001", "ST85100002"}
        assert any(e.rule.startswith("merge_blocked_by_") for e in result.audit)


class TestMerges:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Oak Rd Resurfacing", "Oak Road Resurfacing"),
            ("Main St. Rd Impr", "Main Street Road Improvements"),
            ("Community Ctr Bldg Rehab", "Community Center Building Rehabilitation"),
            ("N Main Ave Widening", "North Main Avenue Widening"),
        ],
    )
    def test_abbreviation_differences_merge(self, a, b):
        result = run([rec(1, a, doc_id="doc_old"), rec(1, b, doc_id="doc_new")])
        assert len(result.projects) == 1
        assert result.audit[0].decision == "merged" and result.audit[0].rule == "near_identical_name"

    def test_merged_project_keeps_every_source_and_the_longest_description(self):
        a = rec(1, "Oak Rd Resurfacing", doc_id="doc_old", description="Resurface Oak Rd.")
        b = rec(1, "Oak Road Resurfacing", doc_id="doc_new", description="Resurface Oak Road from 1st to 9th Street.")
        project = run([a, b]).projects[0]
        assert project.project_description == "Resurface Oak Road from 1st to 9th Street."
        assert len(project.references) == 2


class TestPossibleDuplicates:
    PAIR = ("Harbor Road Resurfacing", "Harbor Road Reconstruction")

    def test_similar_but_uncorroborated_names_are_flagged_not_merged(self):
        records = [rec(1, self.PAIR[0], doc_id="doc_old"), rec(1, self.PAIR[1], doc_id="doc_new")]
        result = run(records)
        assert len(result.projects) == 2 and result.duplicates_merged == 0
        assert result.possible_duplicates == 1
        entry = next(e for e in result.audit if e.decision == "possible_duplicate")
        assert entry.rule == "similar_name_uncorroborated" and 72 <= entry.score < 93
        a, b = result.projects
        assert a.possible_duplicates == [b.project_uid] and b.possible_duplicates == [a.project_uid]
        assert a.needs_review and "possible duplicate" in a.review_reasons

    def test_clearly_different_names_are_silent(self):
        records = [rec(1, "Harbor Road Resurfacing"), rec(2, "Library Roof Replacement")]
        result = run(records)
        assert len(result.projects) == 2 and result.possible_duplicates == 0 and result.audit == []


class TestBudgetConflicts:
    def test_conflicting_budgets_are_preserved_with_their_source(self):
        old = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old", budget=2_000_000, pages=(12,))
        new = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_new", budget=1_000_000, pages=(31,))
        project = run([old, new]).projects[0]
        assert project.estimated_budget == 1_000_000  # newest document
        assert len(project.budget_conflicts) == 1
        conflict = project.budget_conflicts[0]
        assert conflict["estimated_budget"] == 2_000_000 and conflict["pages"] == [12]
        assert conflict["document"] == "CIP 2024"
        assert "conflicting budgets across sources" in project.review_reasons and project.needs_review

    def test_budgets_within_five_percent_are_not_conflicts(self):
        old = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old", budget=1_020_000)
        new = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_new", budget=1_000_000)
        assert run([old, new]).projects[0].budget_conflicts == []

    def test_budget_from_an_older_record_fills_a_missing_one(self):
        old = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old", budget=900_000)
        new = rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_new")
        assert run([old, new]).projects[0].estimated_budget == 900_000


class TestStableIds:
    def records(self):
        return [rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_old"),
                rec(1, "Harbor Road Resurfacing", pid="ST85100001", doc_id="doc_new"),
                rec(2, "Library Roof Replacement", pid="ST85100002", doc_id="doc_new"),
                rec(3, "Oak Rd Resurfacing", doc_id="doc_new"), rec(3, "Oak Road Resurfacing", doc_id="doc_old")]

    def test_uids_are_deterministic_across_calls_and_input_order(self):
        first = sorted(p.project_uid for p in run(self.records()).projects)
        second = sorted(p.project_uid for p in run(self.records()).projects)
        shuffled = self.records()
        random.Random(7).shuffle(shuffled)
        third = sorted(p.project_uid for p in run(shuffled).projects)
        assert first == second == third
        assert len(set(first)) == 3 and all(u.startswith("PRJ-") and len(u) == 14 for u in first)

    def test_uid_depends_on_the_organization(self):
        a = consolidate(self.records(), DOCS, CFG, "orga").projects
        b = consolidate(self.records(), DOCS, CFG, "orgb").projects
        assert {p.project_uid for p in a}.isdisjoint({p.project_uid for p in b})

    def test_uid_resolver_receives_alias_keys_and_can_override(self):
        seen = []

        def resolver(alias_keys, default):
            seen.append((alias_keys, default))
            return "PRJ-FIXED" + str(len(seen))

        result = run(self.records(), uid_resolver=resolver)
        assert any(keys == ["id:ST85100001"] for keys, _ in seen)
        assert {p.project_uid for p in result.projects} == {"PRJ-FIXED1", "PRJ-FIXED2", "PRJ-FIXED3"}

    def test_duplicate_resolved_uids_are_made_unique_within_a_run(self):
        result = run(self.records(), uid_resolver=lambda keys, default: "PRJ-SAME")
        assert len({p.project_uid for p in result.projects}) == 3

    def test_empty_input(self):
        result = run([])
        assert result.projects == [] and result.audit == [] and result.raw_records == 0
