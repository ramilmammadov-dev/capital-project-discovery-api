"""Grounding validation of extracted records (src/capex/extraction/validation.py)."""

from __future__ import annotations

import pytest

from capex.config import ExtractionConfig
from capex.extraction.validation import ChunkText, build_record
from capex.models import DocStatus, FundingStatus, Phase

from .helpers import make_chunk, make_document

ROW_1 = "ST85100001   Harbor Road Resurfacing         Zone A district  $5,300       $5,300"
ROW_3 = "ST85100003   Ridge Park Renovation           Zone C district  $900         $900"


def build(raw, *, unit="dollars", chunk=None, document=None, index=0, **kw):
    chunk = chunk or make_chunk()
    return build_record(
        raw,
        index=index,
        chunk=chunk,
        chunk_text=ChunkText.build(chunk),
        document=document or make_document(),
        organization="City of Exampleville",
        amount_unit=unit,
        model="fake-model",
        config=kw.pop("config", ExtractionConfig()),
        **kw,
    )


def good(**over):
    raw = {
        "project_name": "Harbor Road Resurfacing",
        "project_id": "ST85100001",
        "source_excerpt": ROW_1,
        "source_pages": [4],
        "budget_text": "$5,300",
        "current_phase": "funded",
        "extraction_confidence": 0.9,
        "domain": "transportation",
    }
    raw.update(over)
    return raw


class TestChunkText:
    def test_pages_numbers_and_printed_labels(self):
        ct = ChunkText.build(make_chunk())
        assert sorted(ct.pages) == [4, 5]
        assert "Harbor Road Resurfacing" in ct.pages[4] and "Ridge Park" in ct.pages[5]
        assert "[[PDF page" not in ct.full
        assert ct.printed == {5: "51"}
        assert {"5300", "1200", "6500", "900", "2025", "2028"} <= ct.numbers
        assert "ST85100001" in ct.id_text


class TestAcceptedRecord:
    def test_clean_row_is_accepted_and_fully_verified(self):
        outcome = build(good())
        rec = outcome.record
        assert outcome.reason is None and rec is not None
        assert rec.project_name == "Harbor Road Resurfacing"
        assert rec.project_id == "ST85100001"
        assert rec.estimated_budget == 5300
        assert rec.source_pages == [4]
        assert rec.name_verified and rec.excerpt_verified
        assert rec.flags == []
        assert rec.needs_review is False
        assert rec.record_id == "doc_x:0001#000"
        assert rec.source_url == "https://www.examplecity.gov/files/cip.pdf"
        assert rec.source_document == "FY2027-2031 CIP"
        assert rec.document_fiscal_year == 2027
        assert rec.domain == "transportation"
        assert rec.organization_name == "City of Exampleville"

    def test_unknown_domain_falls_back_to_other_and_confidence_is_clamped(self):
        rec = build(good(domain="space travel", extraction_confidence=7)).record
        assert rec.domain == "other"
        assert rec.extraction_confidence == 1.0
        assert build(good(extraction_confidence=-2)).record.extraction_confidence == 0.0

    def test_printed_page_is_reported_for_page_5(self):
        rec = build(good(project_name="Ridge Park Renovation", project_id="ST85100003", source_excerpt=ROW_3,
                         source_pages=[5], budget_text="$900")).record
        assert rec.source_pages == [5]
        assert rec.printed_pages == ["51"]
        assert rec.estimated_budget == 900


class TestRejections:
    def test_invented_project_name_is_rejected(self):
        outcome = build(good(project_name="Quantum Teleportation Hub"))
        assert outcome.record is None
        assert outcome.reason == "name_not_in_source"

    @pytest.mark.parametrize("name", ["", "  ", "ab", None])
    def test_empty_name(self, name):
        assert build(good(project_name=name)).reason == "empty_name"

    @pytest.mark.parametrize(
        "name",
        ["Total", "Subtotal Parks", "Sub-total Streets", "Grand Total", "Total Capital Program",
         "Department Totals", "Fund total", "All projects", "Various", "TOTAL CAPITAL BUDGET"],
    )
    def test_aggregate_rows_are_rejected(self, name):
        outcome = build(good(project_name=name))
        assert outcome.record is None
        assert outcome.reason == "aggregate_row"

    def test_totally_is_not_an_aggregate(self):
        chunk = make_chunk("[[PDF page 4]]\nST85100009   Totally Green Plaza Redesign   $400")
        outcome = build(good(project_name="Totally Green Plaza Redesign", project_id="ST85100009",
                             source_excerpt="ST85100009   Totally Green Plaza Redesign   $400", budget_text="$400"),
                        chunk=chunk)
        assert outcome.record is not None

    def test_real_project_starting_with_total_is_kept(self):
        row = "ST85100010   Total Maximum Daily Load Compliance   $750"
        chunk = make_chunk("[[PDF page 4]]\n" + row)
        outcome = build(good(project_name="Total Maximum Daily Load Compliance", project_id="ST85100010",
                             source_excerpt=row, budget_text="$750"), chunk=chunk)
        assert outcome.record is not None


class TestBudgets:
    def test_budget_not_in_text_is_dropped_and_flagged(self):
        outcome = build(good(budget_text="$7,777,000"))
        rec = outcome.record
        assert rec.estimated_budget is None
        assert rec.budget_text is None
        assert "budget_not_in_source" in rec.flags
        assert rec.needs_review is True
        assert rec.extraction_confidence == pytest.approx(0.8)  # 0.9 - 0.1 penalty

    def test_thousands_unit_scales_the_printed_amount(self):
        rec = build(good(), unit="thousands").record
        assert rec.estimated_budget == 5_300_000
        assert rec.budget_text == "$5,300"

    def test_millions_unit(self):
        assert build(good(), unit="millions").record.estimated_budget == 5_300_000_000

    def test_funding_by_year_must_be_grounded_too(self):
        raw = good(budget_text="", funding_by_year=[
            {"year_label": "FY 2027", "amount_text": "$5,300"},
            {"year_label": "FY 2028", "amount_text": "$9,999"},  # not printed anywhere
        ])
        rec = build(raw).record
        assert [(f.fiscal_year, f.amount) for f in rec.funding_by_year] == [(2027, 5300.0)]
        assert rec.estimated_budget is None
        assert "budget_missing_yearly_amounts_only" in rec.flags


class TestPages:
    def test_wrong_cited_page_is_corrected_to_the_page_holding_the_excerpt(self):
        outcome = build(good(source_pages=[5]))
        rec = outcome.record
        assert rec.source_pages == [4]
        assert "page_corrected" in rec.flags

    def test_correct_cited_page_is_kept_without_flag(self):
        rec = build(good(source_pages=[4])).record
        assert rec.source_pages == [4]
        assert "page_corrected" not in rec.flags

    def test_missing_citation_is_located_from_the_text(self):
        rec = build(good(source_pages=[])).record
        assert rec.source_pages == [4]
        assert "page_located" in rec.flags

    def test_citation_of_a_page_outside_the_chunk_is_ignored(self):
        rec = build(good(source_pages=[99])).record
        assert rec.source_pages == [4]
        assert "page_located" in rec.flags

    def test_unverifiable_excerpt_is_replaced_by_the_verbatim_source_line(self):
        rec = build(good(source_excerpt="something the model made up entirely")).record
        assert "excerpt_replaced_with_source_line" in rec.flags
        assert "Harbor Road Resurfacing" in rec.source_excerpt
        assert rec.excerpt_verified is True


class TestIdentifiersAndYears:
    def test_id_not_in_text_is_dropped(self):
        rec = build(good(project_id="ZZ99999999")).record
        assert rec.project_id is None
        assert "id_not_in_source" in rec.flags

    def test_id_with_different_punctuation_is_still_grounded(self):
        assert build(good(project_id="ST-85100001")).record.project_id == "ST-85100001"

    def test_years_not_in_text_are_dropped(self):
        raw = good(start_year_text="FY 2031", construction_year_text="FY 2028", completion_year_text="2040")
        rec = build(raw, chunk=make_chunk()).record
        assert rec.anticipated_start_year is None
        assert rec.anticipated_construction_year == 2028
        assert rec.anticipated_completion_year is None
        assert "start_year_not_in_source" in rec.flags
        assert "completion_year_not_in_source" in rec.flags
        assert "construction_year_not_in_source" not in rec.flags

    def test_year_followed_by_comma_in_source_counts_as_printed(self):
        chunk = make_chunk("[[PDF page 4]]\nST85100001   Harbor Road Resurfacing   $5,300\nDesign completed in 2025, bids due soon")
        rec = build(good(start_year_text="2025", source_excerpt="ST85100001   Harbor Road Resurfacing   $5,300"),
                    chunk=chunk).record
        assert rec.anticipated_start_year == 2025


class TestLifecycleThroughValidation:
    def test_future_year_construction_claim_is_downgraded_with_a_flag(self):
        raw = good(current_phase="construction_in_progress", phase_evidence="Construction FY 2028",
                   phase_is_explicit=True)
        rec = build(raw).record
        assert rec.current_phase == Phase.FUNDED
        assert rec.phase_basis == "inferred"
        assert any(f.startswith("phase_rule:") for f in rec.flags)

    def test_explicit_construction_is_kept(self):
        raw = good(current_phase="construction_in_progress", phase_evidence="under construction",
                   phase_is_explicit=True)
        rec = build(raw).record
        assert rec.current_phase == Phase.CONSTRUCTION
        assert rec.phase_basis == "explicit"

    def test_proposed_document_makes_funded_rows_proposed_and_requested(self):
        document = make_document(status=DocStatus.PROPOSED)
        rec = build(good(current_phase="funded", funding_status="funded"), document=document).record
        assert rec.current_phase == Phase.PROPOSED
        assert rec.funding_status == FundingStatus.REQUESTED

    def test_adopted_document_with_allocation_and_no_stage_is_funded(self):
        rec = build(good(current_phase="unknown")).record
        assert rec.current_phase == Phase.FUNDED
        assert rec.funding_status == FundingStatus.FUNDED
