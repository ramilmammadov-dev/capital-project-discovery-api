"""Deterministic parsing and normalization (spec §12: use code, not LLMs, for clear-cut work).

The extraction model copies values *verbatim* (e.g. ``"$1,250"`` with the table unit ``thousands``);
this module turns them into numbers/years. Keeping normalization deterministic makes every value
traceable to the printed text and keeps the LLM from inventing arithmetic.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# --------------------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------------------

_DASHES = re.compile(r"[‐-―−]")
_QUOTES = re.compile(r"[‘’‚‛“”„‟\"'`´]")
_WS = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _DASHES.sub("-", text)
    return _WS.sub(" ", text).strip()


def norm_for_match(text: str | None) -> str:
    """Aggressive normalization used for fuzzy grounding checks."""
    text = clean_text(text).lower()
    text = _QUOTES.sub("", text)
    text = re.sub(r"[^\w$%.,/&#\- ]+", " ", text)
    return _WS.sub(" ", text).strip()


def digits_only(text: str | None) -> str:
    return re.sub(r"\D", "", text or "")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(*parts: str, length: int = 12) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]


def slugify(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "org"


def estimate_tokens(text: str) -> int:
    # ~3.6 characters per token for English prose/tables with layout whitespace collapsed
    return max(1, int(len(_WS.sub(" ", text)) / 3.6))


# --------------------------------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------------------------------

_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}


def canonical_url(url: str) -> str:
    """Canonical form used for de-duplicating candidate documents."""
    parts = urlsplit(url.strip())
    scheme = "https" if parts.scheme in ("http", "https") else parts.scheme
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    netloc = netloc.removesuffix(":443").removesuffix(":80")
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in _TRACKING_PARAMS])
    path = re.sub(r"/{2,}", "/", parts.path) or "/"
    return urlunsplit((scheme, netloc, path, query, ""))


def doc_id_for_url(url: str) -> str:
    return "doc_" + short_hash(canonical_url(url), length=10)


def page_link(url: str, page: int | None, is_pdf: bool = True) -> str:
    """Deep link that opens a PDF at the cited page in browsers/Acrobat (#page=N)."""
    if not page or not is_pdf:
        return url
    return f"{url.split('#')[0]}#page={page}"


# --------------------------------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------------------------------

_UNIT_MULTIPLIER = {"dollars": 1.0, "thousands": 1e3, "millions": 1e6, "unknown": 1.0}
_SUFFIX_MULTIPLIER = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mm": 1e6,
    "mil": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
}
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_MONEY_TOKEN = re.compile(
    rf"(?P<neg>\(|-)?\s*\$?\s*(?P<num>{_NUM})\s*(?P<suffix>billion|bn|million|millions|mil|mm|thousand|[kmb])?\b",
    re.IGNORECASE,
)
_RANGE = re.compile(
    rf"\$?\s*(?P<a>{_NUM})\s*(?:-|to|through)\s*\$?\s*(?P<b>{_NUM})\s*(?P<suffix>billion|bn|million|millions|mil|mm|thousand|[kmb])\b",
    re.IGNORECASE,
)
_NULL_MONEY = {"", "-", "--", "n/a", "na", "tbd", "tba", "none", "$-", "$ -", "unknown", "varies"}


def parse_money(text: str | None, unit: str = "dollars") -> float | None:
    """Parse a printed amount into US dollars.

    ``unit`` is the table-level unit reported by the extractor (``dollars``/``thousands``/
    ``millions``). An explicit suffix in the text (``$1.2M``, ``3 million``) wins over the unit.
    Returns ``None`` for blanks, dashes, zero and non-numeric text.
    """
    if text is None:
        return None
    raw = clean_text(text)
    if raw.lower() in _NULL_MONEY:
        return None
    multiplier = _UNIT_MULTIPLIER.get(unit, 1.0)
    range_match = _RANGE.search(raw)
    if range_match:
        suffix_mult = _SUFFIX_MULTIPLIER[range_match.group("suffix").lower()]
        value = max(float(range_match.group("a").replace(",", "")), float(range_match.group("b").replace(",", "")))
        return value * suffix_mult or None
    for match in _MONEY_TOKEN.finditer(raw):
        number = match.group("num")
        # skip things that are clearly years or fiscal-year labels, e.g. "FY2026"
        start = match.start("num")
        if re.fullmatch(r"(19|20)\d{2}", number) and (
            raw[max(0, start - 2) : start].upper().endswith("FY") or "$" not in raw
        ):
            continue
        value = float(number.replace(",", ""))
        suffix = (match.group("suffix") or "").lower()
        value *= _SUFFIX_MULTIPLIER.get(suffix, multiplier) if suffix else multiplier
        if value <= 0:
            return None
        if match.group("neg") == "(" and raw.strip().endswith(")"):
            value = -value
        return value
    return None


def amounts_in_text(text: str) -> list[str]:
    """Digit strings of every number in ``text`` (used to check a budget is grounded)."""
    return [digits_only(m.group(0)) for m in re.finditer(_NUM, text or "")]


# --------------------------------------------------------------------------------------------------
# Years
# --------------------------------------------------------------------------------------------------

_FY_SPLIT = re.compile(r"\b(?:FY\s*)?(?P<y>(?:19|20)\d{2})\s*[/\-]\s*(?P<yy>\d{2})\b", re.IGNORECASE)
_FY_SHORT = re.compile(r"\bFY\s*'?(?P<yy>\d{2})\b", re.IGNORECASE)
_YEAR = re.compile(r"(?<![\d$,.])(?P<y>(?:19|20)\d{2})(?!\d|,\d)")


def _plausible(year: int) -> bool:
    return 1990 <= year <= 2075


def year_tokens(text: str | None) -> list[tuple[int, int]]:
    """All (position, year) mentions; fiscal 'YYYY-YY' labels resolve to the FY end year."""
    if not text:
        return []
    text = clean_text(text)
    found: list[tuple[int, int]] = []
    taken: list[tuple[int, int]] = []
    for m in _FY_SPLIT.finditer(text):
        start, yy = int(m.group("y")), int(m.group("yy"))
        if (start + 1) % 100 == yy:
            found.append((m.start(), start + 1))
            taken.append(m.span())
    for m in _FY_SHORT.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        year = 2000 + int(m.group("yy"))
        if _plausible(year):
            found.append((m.start(), year))
            taken.append(m.span())
    for m in _YEAR.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        year = int(m.group("y"))
        if _plausible(year):
            found.append((m.start(), year))
    return sorted(found)


def parse_year(text: str | None, prefer: str = "first") -> int | None:
    years = [y for _, y in year_tokens(text)]
    if not years:
        return None
    return max(years) if prefer == "last" else years[0]


_RANGE_SEP = re.compile(r"^\s*(?:-|to|through|thru|and)\s*(?:FY\s*)?", re.IGNORECASE)


_COMPACT_RANGE = re.compile(
    r"\b(?:FY\s*'?)(?P<a>\d{2})\s*[-/]\s*(?:FY\s*)?'?(?P<b>\d{2})\b|"
    r"\b(?:FY\s*)?(?P<A>(?:19|20)\d{2})\s*-\s*(?P<B>\d{2})\b(?!\s*[-/]\s*\d)",
    re.IGNORECASE,
)


def parse_coverage(text: str | None) -> tuple[int | None, int | None]:
    """Fiscal-year coverage of a document title/cover, e.g. 'FY 2027-2031' -> (2027, 2031),
    'FY27-31' / 'FY2027-31' -> (2027, 2031); a single fiscal label 'FY 2026-27' -> (2027, 2027)."""
    if not text:
        return None, None
    text = clean_text(text)
    compact = _COMPACT_RANGE.search(text)
    if compact:
        if compact.group("a"):
            a, b = 2000 + int(compact.group("a")), 2000 + int(compact.group("b"))
        else:
            a = int(compact.group("A"))
            b = a - a % 100 + int(compact.group("B"))
        if 2 <= b - a <= 12:
            return a, b
    tokens = year_tokens(text)
    # explicit range "A - B" / "A through B"
    for (pos_a, a), (pos_b, b) in itertools.pairwise(tokens):
        between = text[pos_a:pos_b]
        between = re.sub(r"^(?:FY\s*)?(?:19|20)\d{2}(?:\s*[/\-]\s*\d{2}\b)?", "", between, flags=re.IGNORECASE)
        if _RANGE_SEP.match(between) and 0 < b - a <= 12:
            return a, b
    # "2026-2030" written without spaces is caught above; a lone FY label -> single year
    years = [y for _, y in tokens]
    if not years:
        return None, None
    end = max(years) if len(years) <= 2 and max(years) - min(years) <= 12 else min(years)
    return min(years), end


# --------------------------------------------------------------------------------------------------
# Project identifiers & names
# --------------------------------------------------------------------------------------------------

_ID_TOKEN = re.compile(r"\b(?=[A-Z0-9\-.]{4,20}\b)(?=[A-Z0-9\-.]*\d)(?=[A-Z0-9\-.]*[A-Z])[A-Z0-9][A-Z0-9\-.]*[A-Z0-9]\b")
_ID_EXCLUDE = re.compile(
    r"^(FY\d{2,4}|\d{4}-\d{2,4}|Q[1-4]\d*|[A-Z]?\d+(ST|ND|RD|TH)|COVID-19|I-\d{1,3}|US-?\d{1,3}|SR-?\d{1,3}|\d+(AM|PM)|K-?12|\d+K|\d+M|\d+B|\d+MW|\d+MGD|\d+KV|H2O)$"
)


def find_id_tokens(text: str) -> list[str]:
    """Tokens that look like project identifiers (letters + digits, e.g. 'H898001C', 'ST85100123')."""
    out = []
    for m in _ID_TOKEN.finditer(text or ""):
        token = m.group(0).strip(".-")
        digits = sum(c.isdigit() for c in token)
        if digits < 3 or _ID_EXCLUDE.match(token):
            continue
        out.append(token)
    return out


def normalize_project_id(value: str | None) -> str | None:
    if not value:
        return None
    token = re.sub(r"[^A-Z0-9]", "", clean_text(value).upper())
    if len(token) < 3 or not any(c.isdigit() for c in token):
        return None
    return token


def id_shape(value: str) -> str:
    """Shape signature of an identifier ('ST85100123' -> 'AADDDDDDDD') used to tell ID schemes apart."""
    return "".join("A" if c.isalpha() else "D" for c in value)


_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "pkwy": "parkway",
    "hwy": "highway",
    "fwy": "freeway",
    "ln": "lane",
    "dr": "drive",
    "ctr": "center",
    "cntr": "center",
    "bldg": "building",
    "impr": "improvements",
    "imp": "improvements",
    "imps": "improvements",
    "improvement": "improvements",
    "rehab": "rehabilitation",
    "repl": "replacement",
    "replace": "replacement",
    "reconst": "reconstruction",
    "recon": "reconstruction",
    "constr": "construction",
    "wtp": "water treatment plant",
    "wwtp": "wastewater treatment plant",
    "wrf": "water reclamation facility",
    "wrp": "water reclamation plant",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "mp": "milepost",
    "tl": "traffic interchange",
    "ti": "traffic interchange",
    "and": "and",
}
_NAME_STOPWORDS = {"the", "of", "a", "an", "for", "to", "in", "at", "on", "project", "program", "and"}
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8"}


def name_key(name: str | None) -> str:
    """Normalized project name used for fuzzy comparison."""
    text = clean_text(name).lower().replace("&", " and ")
    text = _ORDINAL.sub(r"\1", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    words = [_ABBREVIATIONS.get(w, w) for w in text.split()]
    words = [w for w in " ".join(words).split() if w not in _NAME_STOPWORDS]
    return " ".join(words)


def name_numbers(name: str | None) -> set[str]:
    """Numbers inside a name (street numbers, mileposts, phases, years) — distinct numbers => distinct projects."""
    text = _ORDINAL.sub(r"\1", clean_text(name).lower())
    return set(re.findall(r"\d+(?:\.\d+)?", text))


_PHASE_MARKER = re.compile(
    r"\b(phase|ph|segment|seg|stage|part|unit|package|pkg|increment|section)\.?\s*([0-9]+|[ivx]{1,4}|[a-h])\b",
    re.IGNORECASE,
)


def phase_markers(name: str | None) -> set[str]:
    """Component markers ('Phase 2', 'Segment B') — differing markers mean different components."""
    markers = set()
    for kind, value in _PHASE_MARKER.findall(clean_text(name)):
        kind = kind.lower().rstrip(".")
        kind = {"ph": "phase", "seg": "segment", "pkg": "package"}.get(kind, kind)
        value = _ROMAN.get(value.lower(), value.lower())
        markers.add(f"{kind}:{value}")
    return markers
