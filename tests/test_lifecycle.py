"""Deterministic lifecycle and funding-status rules (src/capex/lifecycle.py, spec section 8)."""

from __future__ import annotations

import pytest

from capex.lifecycle import apply_funding_rules, apply_phase_rules
from capex.models import DocStatus, FundingStatus, Phase

ADOPTED = DocStatus.ADOPTED
PROPOSED = DocStatus.PROPOSED
UNKNOWN_DOC = DocStatus.UNKNOWN


def phase_of(phase, evidence="", doc=ADOPTED, allocation=True, explicit=True, status_text="", fy=2026):
    return apply_phase_rules(phase, explicit, evidence, status_text, doc, allocation, fy)


class TestAdvancedStagesNeedEvidence:
    @pytest.mark.parametrize(
        ("doc", "expected"),
        [(ADOPTED, Phase.FUNDED), (PROPOSED, Phase.PROPOSED)],
    )
    def test_construction_with_only_a_future_year_is_downgraded(self, doc, expected):
        phase, basis, notes = phase_of(Phase.CONSTRUCTION, "Construction FY2029", doc=doc)
        assert phase == expected
        assert basis == "inferred"
        assert notes
        assert "Construction / In Progress" in notes[0]

    def test_future_year_downgrade_does_not_depend_on_document_year(self):
        phase, basis, _ = phase_of(Phase.CONSTRUCTION, "Construction FY2029", fy=None)
        assert (phase, basis) == (Phase.FUNDED, "inferred")

    @pytest.mark.parametrize("stage", [Phase.DESIGN, Phase.PROCUREMENT])
    def test_design_or_procurement_scheduled_for_a_future_year_is_downgraded(self, stage):
        evidence = "Design FY2028" if stage == Phase.DESIGN else "RFP FY2028"
        phase, basis, notes = phase_of(stage, evidence)
        assert phase == Phase.FUNDED
        assert basis == "inferred"
        assert "future year (2028)" in notes[0]

    @pytest.mark.parametrize(
        ("doc", "allocation", "expected"),
        [
            (ADOPTED, True, Phase.FUNDED),
            (PROPOSED, True, Phase.PROPOSED),
            (PROPOSED, False, Phase.PROPOSED),
            (ADOPTED, False, Phase.UNKNOWN),
            (UNKNOWN_DOC, False, Phase.UNKNOWN),
            (UNKNOWN_DOC, True, Phase.FUNDED),
        ],
    )
    def test_design_without_evidence_falls_back_to_document_status(self, doc, allocation, expected):
        phase, basis, notes = phase_of(Phase.DESIGN, evidence="", doc=doc, allocation=allocation)
        assert phase == expected
        assert basis == "inferred"
        assert "not supported by explicit evidence" in notes[0]

    @pytest.mark.parametrize(
        "stage",
        [Phase.DESIGN, Phase.PROCUREMENT, Phase.CONSTRUCTION, Phase.COMPLETED, Phase.DEFERRED],
    )
    def test_every_advanced_stage_is_dropped_when_the_evidence_is_blank(self, stage):
        phase, basis, _ = phase_of(stage, evidence="")
        assert phase == Phase.FUNDED
        assert basis == "inferred"

    def test_evidence_that_names_a_different_stage_does_not_support_the_claim(self):
        # "RFP issued" supports procurement, not construction
        phase, _, _ = phase_of(Phase.CONSTRUCTION, evidence="RFP issued in January 2026")
        assert phase == Phase.FUNDED


class TestExplicitEvidenceIsKept:
    @pytest.mark.parametrize(
        "evidence",
        [
            "Project is under construction",
            "under construction; expected completion FY2028",
            "Construction in progress, completion 2029",
            "construction underway",
        ],
    )
    def test_explicit_under_construction_kept(self, evidence):
        phase, basis, notes = phase_of(Phase.CONSTRUCTION, evidence)
        assert phase == Phase.CONSTRUCTION
        assert basis == "explicit"
        assert notes == []

    def test_inferred_basis_when_the_extractor_did_not_flag_it_explicit(self):
        phase, basis, _ = phase_of(Phase.CONSTRUCTION, "construction began in March 2025", explicit=False)
        assert (phase, basis) == (Phase.CONSTRUCTION, "inferred")

    def test_status_text_can_supply_the_evidence(self):
        phase, basis, _ = phase_of(Phase.CONSTRUCTION, evidence="", status_text="In progress")
        assert (phase, basis) == (Phase.CONSTRUCTION, "explicit")

    @pytest.mark.parametrize(
        ("stage", "evidence"),
        [
            (Phase.DESIGN, "Design underway"),
            (Phase.DESIGN, "Design 60% complete"),
            (Phase.DESIGN, "feasibility study in progress"),
            (Phase.PROCUREMENT, "RFP issued in Jan 2026"),
            (Phase.PROCUREMENT, "bids opened"),
            (Phase.COMPLETED, "Project completed in 2024"),
            (Phase.COMPLETED, "placed in service"),
            (Phase.DEFERRED, "Project deferred to 2030"),
            (Phase.DEFERRED, "postponed pending funding"),
        ],
    )
    def test_stage_with_supporting_evidence_is_kept(self, stage, evidence):
        phase, basis, notes = phase_of(stage, evidence)
        assert phase == stage
        assert basis == "explicit"
        assert notes == []


class TestDefaultsFromDocumentStatus:
    def test_unknown_adopted_with_allocation_becomes_funded(self):
        phase, basis, notes = phase_of(Phase.UNKNOWN, doc=ADOPTED, allocation=True, explicit=False)
        assert (phase, basis) == (Phase.FUNDED, "inferred")
        assert "adopted program allocation" in notes[0]

    def test_identified_adopted_with_allocation_becomes_funded(self):
        phase, basis, _ = phase_of(Phase.IDENTIFIED, doc=ADOPTED, allocation=True, explicit=False)
        assert (phase, basis) == (Phase.FUNDED, "inferred")

    def test_identified_with_unfunded_wording_stays_identified(self):
        phase, basis, notes = phase_of(
            Phase.IDENTIFIED, evidence="future need, unfunded", doc=ADOPTED, allocation=True, explicit=False
        )
        assert (phase, basis, notes) == (Phase.IDENTIFIED, "inferred", [])

    def test_unknown_in_a_proposed_document_becomes_proposed(self):
        phase, basis, notes = phase_of(Phase.UNKNOWN, doc=PROPOSED, allocation=True, explicit=False)
        assert (phase, basis) == (Phase.PROPOSED, "inferred")
        assert notes

    def test_unknown_with_nothing_to_go_on_stays_unknown(self):
        phase, basis, notes = phase_of(Phase.UNKNOWN, doc=UNKNOWN_DOC, allocation=False, explicit=False)
        assert (phase, basis, notes) == (Phase.UNKNOWN, "inferred", [])

    def test_unknown_with_allocation_in_unclassified_document_becomes_funded(self):
        phase, _, _ = phase_of(Phase.UNKNOWN, doc=UNKNOWN_DOC, allocation=True, explicit=False)
        assert phase == Phase.FUNDED

    def test_funded_claim_in_a_proposed_document_is_downgraded_unless_explicit(self):
        phase, basis, notes = phase_of(Phase.FUNDED, doc=PROPOSED, explicit=False)
        assert (phase, basis) == (Phase.PROPOSED, "inferred")
        assert "proposed" in notes[0]
        phase, basis, notes = phase_of(Phase.FUNDED, doc=PROPOSED, explicit=True)
        assert (phase, basis, notes) == (Phase.FUNDED, "explicit", [])

    def test_funded_in_adopted_document_unchanged(self):
        assert phase_of(Phase.FUNDED, doc=ADOPTED, explicit=False)[:2] == (Phase.FUNDED, "inferred")

    @pytest.mark.parametrize("stage", [Phase.PROPOSED, Phase.IDENTIFIED])
    def test_proposed_and_identified_pass_through_without_allocation(self, stage):
        phase, _, notes = phase_of(stage, doc=ADOPTED, allocation=False, explicit=True)
        assert phase == stage
        assert notes == []

    def test_unknown_with_unfunded_wording_becomes_identified(self):
        phase, _, _ = phase_of(
            Phase.UNKNOWN, evidence="unfunded future need", doc=ADOPTED, allocation=True, explicit=False
        )
        assert phase == Phase.IDENTIFIED


class TestFundingStatus:
    @pytest.mark.parametrize(
        "wording",
        ["unfunded", "not funded", "no funding identified", "unprogrammed", "future need", "unmet need"],
    )
    @pytest.mark.parametrize("given", [FundingStatus.UNKNOWN, FundingStatus.FUNDED])
    def test_unfunded_wording_wins(self, wording, given):
        assert apply_funding_rules(given, wording, "", Phase.FUNDED, ADOPTED, True) == FundingStatus.UNFUNDED

    @pytest.mark.parametrize(
        "wording",
        ["partially funded", "partial funding", "funding gap of $2M", "additional funding needed"],
    )
    def test_partially_funded_wording(self, wording):
        status = apply_funding_rules(FundingStatus.UNKNOWN, wording, "", Phase.FUNDED, ADOPTED, True)
        assert status == FundingStatus.PARTIALLY_FUNDED

    def test_unfunded_beats_partial_when_both_appear(self):
        status = apply_funding_rules(FundingStatus.UNKNOWN, "unfunded; funding gap", "", Phase.FUNDED, ADOPTED, True)
        assert status == FundingStatus.UNFUNDED

    def test_status_text_is_scanned_too(self):
        status = apply_funding_rules(FundingStatus.UNKNOWN, "", "Unprogrammed", Phase.UNKNOWN, ADOPTED, True)
        assert status == FundingStatus.UNFUNDED

    def test_proposed_document_downgrades_funded_without_evidence_to_requested(self):
        status = apply_funding_rules(FundingStatus.FUNDED, "", "", Phase.FUNDED, PROPOSED, True)
        assert status == FundingStatus.REQUESTED

    def test_proposed_document_keeps_funded_when_evidence_is_given(self):
        status = apply_funding_rules(FundingStatus.FUNDED, "Funded by Measure X", "", Phase.FUNDED, PROPOSED, True)
        assert status == FundingStatus.FUNDED

    @pytest.mark.parametrize(
        ("doc", "expected"),
        [(PROPOSED, FundingStatus.REQUESTED), (ADOPTED, FundingStatus.FUNDED), (UNKNOWN_DOC, FundingStatus.UNKNOWN)],
    )
    def test_unknown_status_with_an_allocation_follows_document_status(self, doc, expected):
        assert apply_funding_rules(FundingStatus.UNKNOWN, "", "", Phase.UNKNOWN, doc, True) == expected

    def test_identified_without_allocation_is_unfunded(self):
        status = apply_funding_rules(FundingStatus.UNKNOWN, "", "", Phase.IDENTIFIED, ADOPTED, False)
        assert status == FundingStatus.UNFUNDED

    def test_identified_with_allocation_in_unclassified_document_stays_unknown(self):
        status = apply_funding_rules(FundingStatus.UNKNOWN, "", "", Phase.IDENTIFIED, UNKNOWN_DOC, True)
        assert status == FundingStatus.UNKNOWN

    @pytest.mark.parametrize("given", [FundingStatus.REQUESTED, FundingStatus.PARTIALLY_FUNDED, FundingStatus.FUNDED])
    def test_stated_status_passes_through_in_adopted_documents(self, given):
        assert apply_funding_rules(given, "", "", Phase.UNKNOWN, ADOPTED, True) == given

    def test_nothing_to_go_on_is_unknown(self):
        status = apply_funding_rules(FundingStatus.UNKNOWN, "", "", Phase.UNKNOWN, UNKNOWN_DOC, False)
        assert status == FundingStatus.UNKNOWN

    def test_phase_and_funding_are_independent(self):
        phase, _, _ = phase_of(Phase.CONSTRUCTION, "under construction", doc=ADOPTED)
        status = apply_funding_rules(FundingStatus.UNKNOWN, "partially funded", "", phase, ADOPTED, True)
        assert phase == Phase.CONSTRUCTION
        assert status == FundingStatus.PARTIALLY_FUNDED
