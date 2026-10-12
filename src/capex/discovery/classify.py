"""Keyword classification of candidate documents from title / URL / snippet / cover text.

Used for every discovered document (cheap metadata for the inventory) and as a prior for ranking.
"""

from __future__ import annotations

import re
from datetime import date

from capex.models import DocStatus, DocType
from capex.normalize import clean_text, parse_coverage

_TYPE_RULES: list[tuple[DocType, re.Pattern[str]]] = [
    (DocType.TIP, re.compile(r"\b(s?tip|(statewide )?transportation improvement program)\b", re.I)),
    (
        DocType.MULTI_YEAR_PROGRAM,
        re.compile(
            r"\b((five|six|ten|5|6|10)[- ]year\b.*\b(program|plan)|facilities construction program|"
            r"construction program|capital program)\b",
            re.I,
        ),
    ),
    (DocType.CIP_PROGRAM, re.compile(r"\b(capital improvements? program|cip)\b", re.I)),
    (DocType.CIP_PLAN, re.compile(r"\bcapital improvements? plan\b", re.I)),
    (DocType.CAPITAL_BUDGET, re.compile(r"\b(capital (budget|expenditures?|outlay|projects? budget)|capex)\b", re.I)),
    (DocType.INFRASTRUCTURE_PLAN, re.compile(r"\binfrastructure (investment )?(plan|program|strategy)\b", re.I)),
    (DocType.FACILITY_PROGRAM, re.compile(r"\b(facilit(y|ies) (master |improvement |condition )?(plan|program)|bond program)\b", re.I)),
    (DocType.BOARD_SCHEDULE, re.compile(r"\b(board|council|commission)\b.*\b(approv|adopt|schedule|agenda|resolution)", re.I)),
    (DocType.PROPOSED_BUDGET, re.compile(r"\b(proposed|recommended|tentative|preliminary|requested)\b.*\bbudget\b", re.I)),
    (DocType.ADOPTED_BUDGET, re.compile(r"\b(adopted|approved|final|annual|biennial)\b.*\bbudget\b|\bbudget book\b", re.I)),
    (DocType.CAPITAL_REPORT, re.compile(r"\b(capital (planning|needs|plan)|asset management plan|needs assessment)\b", re.I)),
]

_ADOPTED = re.compile(r"\b(adopted|approved|final|enacted|board[- ]approved)\b", re.I)
_PROPOSED = re.compile(r"\b(proposed|tentative|draft|recommended|preliminary|requested|planning (draft|version))\b", re.I)

STRONG_TERMS = re.compile(
    r"capital improvement|\bcip\b|capital program|capital plan|(five|six|ten|5|6|10)[- ]year|\bs?tip\b|"
    r"transportation improvement|construction program|capital budget|bond program|facilities (plan|program)|"
    r"infrastructure (plan|program)|capital projects?",
    re.I,
)
BUDGET_TERMS = re.compile(r"\bbudget\b|\bappropriation", re.I)
NEGATIVE_TERMS = re.compile(
    r"\b(minutes|press release|news|newsletter|job|career|rfp|bid tab|solicitation|map|brochure|faq|manual|"
    r"policy|polic(y|ies)|acfr|cafr|annual (comprehensive )?financial report|audit|survey|presentation|"
    r"powerpoint|slides|flyer|ordinance|application|form|guidelines?|standards?|specifications?|"
    r"user guide|how to|instructions|public notice|hearing notice)\b",
    re.I,
)


def classify_doc_type(text: str) -> DocType:
    text = clean_text(text)
    for doc_type, pattern in _TYPE_RULES:
        if pattern.search(text):
            return doc_type
    return DocType.OTHER


def classify_status(text: str) -> DocStatus:
    text = clean_text(text)
    proposed = _PROPOSED.search(text)
    adopted = _ADOPTED.search(text)
    if adopted and (not proposed or adopted.start() < proposed.start()):
        return DocStatus.ADOPTED
    if proposed:
        return DocStatus.PROPOSED
    return DocStatus.UNKNOWN


def series_key(title: str) -> str:
    """Title with years/status words removed: versions of the same document share a series key."""
    text = clean_text(title).lower()
    text = re.sub(r"\b(fy|fiscal years?)\b", " ", text)
    text = re.sub(r"\d{2,4}", " ", text)
    text = re.sub(r"\b(adopted|approved|final|proposed|tentative|draft|recommended|preliminary|annual|update[d]?)\b", " ", text)
    text = re.sub(r"[^a-z]+", " ", text)
    return " ".join(text.split())


def current_fiscal_year(today: date | None = None, start_month: int = 7) -> int:
    """US state/local fiscal years mostly start July 1 (FY2027 = Jul 2026 - Jun 2027)."""
    today = today or date.today()
    return today.year + 1 if today.month >= start_month else today.year


def coverage_from_text(*texts: str) -> tuple[int | None, int | None]:
    for text in texts:
        start, end = parse_coverage(text)
        if start:
            return start, end
    return None, None
