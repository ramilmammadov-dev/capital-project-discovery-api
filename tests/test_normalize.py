"""Deterministic normalization helpers (src/capex/normalize.py)."""

from __future__ import annotations

import pytest

from capex import normalize as nz

# --------------------------------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------------------------------


class TestParseMoney:
    @pytest.mark.parametrize(
        ("text", "unit", "expected"),
        [
            ("$1,234,567", "dollars", 1_234_567),
            ("1,234,567", "dollars", 1_234_567),
            ("750", "dollars", 750),
            ("$12,000,000 (FY2026)", "dollars", 12_000_000),
            # table-level unit scales bare numbers
            ("$1,250", "thousands", 1_250_000),
            ("5,300", "thousands", 5_300_000),
            ("12.5", "millions", 12_500_000),
            ("$12.5", "millions", 12_500_000),
            ("1,250", "unknown", 1_250),
        ],
    )
    def test_units(self, text, unit, expected):
        assert nz.parse_money(text, unit) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("text", "unit", "expected"),
        [
            ("$1.2M", "dollars", 1_200_000),
            ("$1.2M", "thousands", 1_200_000),  # an explicit suffix beats the table unit
            ("3 million", "dollars", 3_000_000),
            ("$2.4 million", "thousands", 2_400_000),
            ("$2.5 billion", "dollars", 2_500_000_000),
            ("1,500K", "dollars", 1_500_000),
            ("$750k", "dollars", 750_000),
        ],
    )
    def test_suffix_wins_over_unit(self, text, unit, expected):
        assert nz.parse_money(text, unit) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["$5-10 million", "$5 - 10 million", "$5 to 10 million", "$5 through $10M"])
    def test_ranges_use_the_upper_bound(self, text):
        assert nz.parse_money(text) == pytest.approx(10_000_000)

    def test_range_suffix_ignores_table_unit(self):
        assert nz.parse_money("$5-10 million", "thousands") == pytest.approx(10_000_000)

    @pytest.mark.parametrize(
        "text",
        ["-", "--", "TBD", "tbd", "TBA", "N/A", "n/a", "", "   ", None, "$-", "$ -", "unknown", "varies", "none", "abc"],
    )
    def test_blank_dash_and_text_are_none(self, text):
        assert nz.parse_money(text) is None

    @pytest.mark.parametrize("text", ["$0", "0", "0.00"])
    def test_zero_is_none(self, text):
        assert nz.parse_money(text) is None

    @pytest.mark.parametrize("text", ["FY2026", "FY 2026", "2026", "fy2031", "FY 2027-2031"])
    def test_years_are_not_money(self, text):
        assert nz.parse_money(text) is None

    def test_amount_after_a_fiscal_year_label(self):
        assert nz.parse_money("FY2026 $5,000") == 5_000

    def test_parenthesised_amount_is_negative(self):
        assert nz.parse_money("(1,250)", "thousands") == pytest.approx(-1_250_000)

    def test_amounts_in_text_returns_digit_strings(self):
        found = nz.amounts_in_text("Budget $1,234,567 and 5,300 (FY 2026)")
        assert "1234567" in found
        assert "5300" in found
        assert nz.amounts_in_text("") == []
        assert nz.amounts_in_text(None) == []


# --------------------------------------------------------------------------------------------------
# Years
# --------------------------------------------------------------------------------------------------


class TestYears:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("FY 2027", 2027),
            ("FY2027", 2027),
            ("FY27", 2027),
            ("FY 27", 2027),
            ("FY'27", 2027),
            ("2026-27", 2027),  # fiscal label resolves to the year the FY ends
            ("FY2026-27", 2027),
            ("FY 2026/27", 2027),
            ("adopted FY 2027-28", 2028),
            ("in 2025", 2025),
        ],
    )
    def test_parse_year(self, text, expected):
        assert nz.parse_year(text) == expected

    @pytest.mark.parametrize("text", [None, "", "no years here", "$2026", "1,2026", "1985", "2099"])
    def test_no_plausible_year(self, text):
        assert nz.parse_year(text) is None
        assert nz.year_tokens(text) == []

    def test_first_vs_last(self):
        assert nz.parse_year("2026-2027") == 2026
        assert nz.parse_year("2026-2027", prefer="last") == 2027
        assert nz.parse_year("built 2024, expanded 2031", prefer="last") == 2031

    def test_year_tokens_are_position_sorted(self):
        tokens = nz.year_tokens("Start 2024; finish FY27")
        assert [y for _, y in tokens] == [2024, 2027]
        assert tokens == sorted(tokens)

    def test_year_followed_by_a_comma_is_still_a_year(self):
        assert [y for _, y in nz.year_tokens("In 2024, the city adopted the plan")] == [2024]

    def test_year_inside_a_thousands_separated_number_is_not_a_year(self):
        assert nz.year_tokens("budget $12,026 and 2,026 units") == []

    def test_fiscal_label_is_not_counted_twice(self):
        # '2026-27' must not additionally yield a bare 2026
        assert [y for _, y in nz.year_tokens("FY 2026-27 budget")] == [2027]


class TestCoverage:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("FY 2027-2031", (2027, 2031)),
            ("Fiscal Years 2027 through 2031", (2027, 2031)),
            ("2026/27 - 2030/31", (2027, 2031)),
            ("2026-27 to 2030-31", (2027, 2031)),
            ("FY2027-FY2031", (2027, 2031)),
            ("FY 2027 - FY 2031", (2027, 2031)),
            ("Fiscal Year 2027 through Fiscal Year 2031", (2027, 2031)),
            ("Capital Improvement Plan 2026-2030", (2026, 2030)),
            ("FY 2027", (2027, 2027)),
            ("Capital Plan 2025", (2025, 2025)),
        ],
    )
    def test_coverage(self, text, expected):
        assert nz.parse_coverage(text) == expected

    @pytest.mark.parametrize("text", [None, "", "Capital Improvement Plan"])
    def test_no_coverage(self, text):
        assert nz.parse_coverage(text) == (None, None)


# --------------------------------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------------------------------


class TestIdTokens:
    def test_keeps_real_project_ids(self):
        found = nz.find_id_tokens("Project H898001C and ST85100123 in FY2026")
        assert found == ["H898001C", "ST85100123"]

    @pytest.mark.parametrize("token", ["W2301A", "ABC123", "CIP-2026-001", "A1234"])
    def test_keeps_other_id_shapes(self, token):
        assert nz.find_id_tokens(f"row {token} row") == [token]

    @pytest.mark.parametrize(
        "text",
        [
            "FY2026",
            "FY27",
            "I-10 widening",
            "US-101",
            "SR-99",
            "2026",
            "2026-27",
            "COVID-19",
            "K-12",
            "7TH 19TH",
            "AB12",  # fewer than three digits
            "250KV",
            "100MGD",
            "",
        ],
    )
    def test_rejects_non_ids(self, text):
        assert nz.find_id_tokens(text) == []

    def test_none_is_empty(self):
        assert nz.find_id_tokens(None) == []  # type: ignore[arg-type]

    def test_mixed_line(self):
        found = nz.find_id_tokens("I-10 and COVID-19 relief FY2026 project ST85100123 (K-12)")
        assert found == ["ST85100123"]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("H-898001-C", "H898001C"),
            ("ST 85100123", "ST85100123"),
            ("st85100123", "ST85100123"),
            ("ab1", "AB1"),
        ],
    )
    def test_normalize_project_id(self, raw, expected):
        assert nz.normalize_project_id(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "abc", "12", "n/a"])
    def test_normalize_project_id_rejects(self, raw):
        assert nz.normalize_project_id(raw) is None

    def test_id_shape(self):
        assert nz.id_shape("ST85100123") == "AADDDDDDDD"
        assert nz.id_shape("H898001C") == "ADDDDDDA"
        assert nz.id_shape("ST85100123") != nz.id_shape("H898001C")


# --------------------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------------------


class TestNames:
    def test_abbreviations_expand_to_the_same_key(self):
        assert nz.name_key("Main St. Rd Impr") == nz.name_key("Main Street Road Improvements")
        assert nz.name_key("Main St. Rd Impr") == "main street road improvements"
        assert nz.name_key("Oak Rd Resurfacing") == nz.name_key("Oak Road Resurfacing")
        assert nz.name_key("Community Ctr Bldg") == nz.name_key("Community Center Building")
        assert nz.name_key("Rehab. of Elm Blvd") == "rehabilitation elm boulevard"
        assert nz.name_key("WWTP Rehab") == "wastewater treatment plant rehabilitation"

    def test_ampersand_stopwords_and_punctuation(self):
        assert nz.name_key("N Main St & Elm Ave") == "north main street elm avenue"
        assert nz.name_key("The Bridge of Hope Project") == "bridge hope"
        assert nz.name_key("ELM-AVE. WIDENING") == "elm avenue widening"

    def test_ordinals_lose_their_suffix(self):
        assert nz.name_key("7th Ave Widening") == "7 avenue widening"
        assert nz.name_key("19th Ave Widening") == "19 avenue widening"
        assert nz.name_key("7th Ave Widening") != nz.name_key("19th Ave Widening")

    def test_empty(self):
        assert nz.name_key(None) == ""
        assert nz.name_key("") == ""

    def test_name_numbers(self):
        assert nz.name_numbers("7th Ave Widening") == {"7"}
        assert nz.name_numbers("19th Ave Widening") == {"19"}
        assert nz.name_numbers("Bridge X Phase 2") == {"2"}
        assert nz.name_numbers("I-5 MP 12.5 Overlay") == {"5", "12.5"}
        assert nz.name_numbers("Elm Street Repaving") == set()
        assert nz.name_numbers(None) == set()

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Bridge X Phase 1", {"phase:1"}),
            ("Bridge X Ph. 2", {"phase:2"}),
            ("Bridge X Phase II", {"phase:2"}),
            ("Bridge X Phase I", {"phase:1"}),
            ("Corridor Segment B", {"segment:b"}),
            ("Corridor Seg 3", {"segment:3"}),
            ("Plant Pkg 3", {"package:3"}),
            ("Admin Building Unit IV", {"unit:4"}),
            ("Phase 1 and Phase 2", {"phase:1", "phase:2"}),
            ("Bridge X", set()),
            (None, set()),
        ],
    )
    def test_phase_markers(self, name, expected):
        assert nz.phase_markers(name) == expected

    def test_roman_and_arabic_phase_markers_are_equal(self):
        assert nz.phase_markers("Bridge X Phase I") == nz.phase_markers("Bridge X Phase 1")


# --------------------------------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------------------------------


class TestUrls:
    def test_canonical_url_strips_tracking_and_fragment(self):
        url = "http://www.Example.com/cip?utm_source=x&a=1&gclid=zz#frag"
        assert nz.canonical_url(url) == "https://example.com/cip?a=1"

    def test_canonical_url_ports_slashes_and_case(self):
        assert nz.canonical_url("https://example.com:443//a//b") == "https://example.com/a/b"
        assert nz.canonical_url("  http://example.com:80/x  ") == "https://example.com/x"
        assert nz.canonical_url("HTTP://EXAMPLE.COM") == "https://example.com/"

    def test_canonical_url_keeps_trailing_slash_and_real_query(self):
        assert nz.canonical_url("https://www.examplecity.gov/cip/") == "https://examplecity.gov/cip/"
        assert nz.canonical_url("https://x.gov/f.pdf?id=7&utm_campaign=a") == "https://x.gov/f.pdf?id=7"

    def test_doc_id_is_stable_across_url_variants(self):
        a = nz.doc_id_for_url("http://www.example.com/a.pdf?utm_source=x")
        b = nz.doc_id_for_url("https://example.com/a.pdf")
        assert a == b
        assert a.startswith("doc_")
        assert nz.doc_id_for_url("https://example.com/b.pdf") != a

    def test_page_link(self):
        assert nz.page_link("https://x.com/a.pdf", 5) == "https://x.com/a.pdf#page=5"
        assert nz.page_link("https://x.com/a.pdf#page=2", 5) == "https://x.com/a.pdf#page=5"
        assert nz.page_link("https://x.com/a.pdf", None) == "https://x.com/a.pdf"
        assert nz.page_link("https://x.com/a.pdf", 0) == "https://x.com/a.pdf"
        assert nz.page_link("https://x.com/a.html", 3, is_pdf=False) == "https://x.com/a.html"


# --------------------------------------------------------------------------------------------------
# Small text helpers
# --------------------------------------------------------------------------------------------------


class TestTextHelpers:
    def test_clean_text(self):
        assert nz.clean_text("  Hello— world  ") == "Hello- world"
        assert nz.clean_text(None) == ""
        assert nz.clean_text("a \n\t b") == "a b"

    def test_norm_for_match(self):
        assert nz.norm_for_match("It's a “Test”  – x") == "its a test - x"
        assert nz.norm_for_match(None) == ""

    def test_slugify(self):
        assert nz.slugify("Los Angeles, CA!") == "los-angeles-ca"
        assert nz.slugify("") == "org"
        assert nz.slugify("City of St. Paul") == "city-of-st-paul"

    def test_hashes_are_deterministic_and_part_sensitive(self):
        assert nz.short_hash("a", "b") == nz.short_hash("a", "b")
        assert len(nz.short_hash("a", "b")) == 12
        assert nz.short_hash("ab", "c") != nz.short_hash("a", "bc")
        assert nz.sha256_text("x") == nz.sha256_text("x")
        assert len(nz.sha256_text("x")) == 64

    def test_digits_only_and_estimate_tokens(self):
        assert nz.digits_only("$1,234.50") == "123450"
        assert nz.digits_only(None) == ""
        assert nz.estimate_tokens("") == 1
        assert nz.estimate_tokens("a" * 36) == 10
