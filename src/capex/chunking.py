"""Layout-aware chunking (spec §5).

The document is turned into a stream of *units* — the smallest pieces that must never be split:
a heading, a paragraph, or a table row together with its wrapped continuation lines (which may run
onto the next page). Units are then packed into chunks that

* break preferably at headings, then at row/paragraph boundaries — never inside a row,
* carry their context: section path, the table's column header when a table continues from an
  earlier chunk, and the last few lines of the previous chunk (marked as context only),
* keep dense tables small (``max_rows_per_chunk``) so the extractor stays exhaustive,
* contain ``[[PDF page N | printed page X]]`` markers so every project can be cited to its page.

Running headers/footers repeated on many pages are removed (they cost tokens and add noise).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from capex.config import ChunkingConfig
from capex.models import Chunk, ParsedDocument, ParsedLine, ParsedPage
from capex.normalize import estimate_tokens, find_id_tokens, sha256_text

CHUNKER_VERSION = "layout-chunker-7"

_AMOUNT = re.compile(r"\$\s?\d[\d,.]*|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_BULLET = re.compile(r"^\s*(?:[-•▪●◦*]|\d{1,3}[.)]|[a-z][.)])\s+")
_DIGITS = re.compile(r"\d")


@dataclass
class _Line:
    page: int
    printed: str | None
    line: ParsedLine
    region: int | None = None  # tabular region id
    rel_top: float = 0.0  # vertical position on the page (0 = top, 1 = bottom); 0 for HTML

    @property
    def text(self) -> str:
        return self.line.text


@dataclass
class _Unit:
    kind: str  # heading | row | para
    lines: list[_Line] = field(default_factory=list)
    level_size: float = 0.0
    region: int | None = None
    is_row: bool = False

    @property
    def tokens(self) -> int:
        return estimate_tokens("\n".join(line.text for line in self.lines))


def _norm_boiler(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))


_BOILER_WORDS = re.compile(
    r"map data|report a map error|\u00a9|copyright|all rights reserved|printed on|generated (on|by)|page # of #",
    re.IGNORECASE,
)


def _is_edge(line: ParsedLine, height: float) -> bool:
    return bool(height) and (line.top <= height * 0.08 or line.top >= height * 0.9)


def _boilerplate(pages: list[ParsedPage]) -> set[str]:
    """Normalized lines repeated on >= 30% of pages (min 3): running headers/footers, plus
    copyright/map captions anywhere on the page."""
    if len(pages) < 4:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        seen = set()
        h = page.height or (max((ln.top for ln in page.lines), default=0) + 20)
        for line in page.lines:
            key = _norm_boiler(line.text)
            if not key or len(key) > 160 or key in seen:
                continue
            if _is_edge(line, h) or _BOILER_WORDS.search(key):
                seen.add(key)
                counts[key] += 1
    threshold = max(3, int(len(pages) * 0.3))
    return {k for k, c in counts.items() if c >= threshold}


def _has_amount(text: str) -> bool:
    return bool(_AMOUNT.search(text))


def _looks_label_only(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not _DIGITS.search(stripped) and len(stripped) <= 90


def _build_units(doc: ParsedDocument) -> list[_Unit]:
    boiler = _boilerplate(doc.pages)
    flat: list[_Line] = []
    for page in doc.pages:
        h = page.height or 0
        for line in page.lines:
            if boiler and _norm_boiler(line.text) in boiler:
                continue
            flat.append(
                _Line(page=page.pdf_page, printed=page.printed_page, line=line, rel_top=(line.top / h) if h else 0.0)
            )

    # tabular regions: consecutive tabular lines (tolerating one non-tabular line in between)
    region = -1
    last_tab_index = -10
    for i, item in enumerate(flat):
        if item.line.tabular and not item.line.heading:
            if i - last_tab_index > 2:
                region += 1
            item.region = region
            last_tab_index = i
    region_left: dict[int, float] = {}
    for item in flat:
        if item.region is not None:
            region_left[item.region] = min(region_left.get(item.region, 1e9), item.line.x0)

    units: list[_Unit] = []
    current: _Unit | None = None
    prev: _Line | None = None
    for item in flat:
        line = item.line
        if line.heading:
            current = _Unit(kind="heading", lines=[item], level_size=line.size)
            units.append(current)
            prev = item
            current = None
            continue

        start_new = True
        if current is not None and prev is not None:
            if item.region is not None and current.region == item.region:
                left = region_left.get(item.region, 0.0)
                is_row_start = line.x0 <= left + 12.0
                prev_label_only = _looks_label_only(prev.text) and not _has_amount(prev.text)
                # continuation: indented (first column empty) or the previous row line was only a label
                start_new = is_row_start and not prev_label_only
                if start_new and is_row_start and not _has_amount(line.text) and not find_id_tokens(line.text):
                    # a left-edge line without numbers right after a row with numbers can still be a
                    # wrapped name; keep it with the previous row when it is short.
                    start_new = len(line.text.strip()) > 60
            elif item.region is None and current.kind == "para":
                gap = line.top - prev.line.top if item.page == prev.page else 99.0
                ends_sentence = prev.text.rstrip().endswith((".", ":", ";", "?", "!"))
                start_new = gap > 18.0 or ends_sentence or bool(_BULLET.match(line.text)) or bool(
                    find_id_tokens(line.text[:24])
                )
        if start_new or current is None:
            kind = "row" if item.region is not None else "para"
            current = _Unit(kind=kind, lines=[item], region=item.region)
            current.is_row = kind == "row" and (_has_amount(line.text) or bool(find_id_tokens(line.text)))
            units.append(current)
        else:
            current.lines.append(item)
            if current.kind == "row" and not current.is_row:
                current.is_row = _has_amount(line.text) or bool(find_id_tokens(line.text))
        prev = item
    return units


_TOTAL = re.compile(r"^(sub)?total|^grand total", re.IGNORECASE)


def _estimate_entities(units: object) -> int:
    """Lower-bound estimate of distinct project rows: rows with amounts/IDs, counted by distinct first token
    (so per-project attribute rows such as 'DESIGN ... / CON ...' funding lines are not counted as projects)."""
    first_tokens: set[str] = set()
    rows = 0
    for unit in units:  # type: ignore[attr-defined]
        if not unit.is_row:
            continue
        text = unit.lines[0].text.strip()
        if not text or _TOTAL.match(text):
            continue
        rows += 1
        first_tokens.add(text.split()[0].lower())
    return min(rows, len(first_tokens))


def _region_headers(units: list[_Unit]) -> tuple[dict[int, int], dict[int, list[str]]]:
    """First unit index of every table region and its column-header lines (leading rows without
    amounts/IDs, max 3 lines)."""
    starts: dict[int, int] = {}
    headers: dict[int, list[str]] = {}
    done: set[int] = set()
    for index, unit in enumerate(units):
        if unit.region is None:
            continue
        starts.setdefault(unit.region, index)
        if unit.region in done:
            continue
        if unit.is_row:
            done.add(unit.region)
            continue
        lines = headers.setdefault(unit.region, [])
        lines.extend(item.text for item in unit.lines)
        if len(lines) >= 3:
            headers[unit.region] = lines[:3]
            done.add(unit.region)
    return starts, headers


def chunk_document(doc: ParsedDocument, config: ChunkingConfig) -> list[Chunk]:
    units = _build_units(doc)
    region_start, region_header = _region_headers(units)
    chunks: list[Chunk] = []
    heading_stack: list[tuple[float, str]] = []

    def section_path() -> str:
        return " > ".join(text for _, text in heading_stack[-3:])

    buf: list[tuple[int, _Unit]] = []
    buf_tokens = 0
    buf_rows = 0
    buf_section = ""
    prev_tail: list[str] = []

    def flush() -> None:
        nonlocal buf, buf_tokens, buf_rows, prev_tail
        content = [(i, u) for i, u in buf if u.lines]
        if not content or all(u.kind == "heading" for _, u in content):
            # headings alone are carried into the next chunk's section path instead
            buf, buf_tokens, buf_rows = [], 0, 0
            return
        text_lines: list[str] = []
        pages: list[int] = []
        printed: dict[int, str] = {}
        current_page = None
        for _, unit in content:
            for item in unit.lines:
                if item.page != current_page:
                    current_page = item.page
                    pages.append(item.page)
                    label = f"[[PDF page {item.page}"
                    if item.printed and item.printed != str(item.page):
                        label += f" | printed page {item.printed}"
                        printed[item.page] = item.printed
                    text_lines.append(label + "]]")
                text_lines.append(item.text)
        context_parts: list[str] = []
        if buf_section:
            context_parts.append(f"Section: {buf_section}")
        first_index, first = next(((i, u) for i, u in content if u.kind != "heading"), content[0])
        if first.region is not None and region_start.get(first.region, first_index) < first_index:
            # the chunk starts inside a table that began earlier: repeat its column header
            header = region_header.get(first.region)
            if header:
                context_parts.append("Table header (repeated):\n" + "\n".join(header))
        if prev_tail:
            context_parts.append("Preceding lines (context only, already covered):\n" + "\n".join(prev_tail))
        text = "\n".join(text_lines)
        kinds = {u.kind for _, u in content if u.kind != "heading"}
        kind = "table" if kinds == {"row"} else "text" if kinds == {"para"} else "mixed"
        seq = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{seq:04d}",
                doc_id=doc.doc_id,
                seq=seq,
                pages=pages,
                printed_pages=printed,
                section=buf_section,
                kind=kind,  # type: ignore[arg-type]
                context="\n".join(context_parts),
                text=text,
                content_hash=sha256_text(CHUNKER_VERSION + "\n" + "\n".join(context_parts) + "\n" + text),
                est_tokens=estimate_tokens(text),
                est_rows=_estimate_entities(u for _, u in content),
            )
        )
        all_lines = [item.text for _, u in content for item in u.lines]
        prev_tail = all_lines[-config.overlap_lines :] if config.overlap_lines else []
        buf, buf_tokens, buf_rows = [], 0, 0

    def add(index: int, unit: _Unit) -> None:
        nonlocal buf_tokens, buf_rows, buf_section
        if not buf:
            buf_section = section_path()
        buf.append((index, unit))
        buf_tokens += unit.tokens
        buf_rows += 1 if unit.is_row else 0

    last_page = None
    for index, unit in enumerate(units):
        unit_page = unit.lines[0].page if unit.lines else last_page
        new_page = last_page is not None and unit_page != last_page
        last_page = unit.lines[-1].page if unit.lines else last_page

        if unit.kind == "heading":
            # a heading starting a page/section closes the current chunk; mid-page labels do not (they are
            # usually field labels of a project sheet whose rows must stay together)
            near_top = unit.lines[0].rel_top <= 0.25 if unit.lines else True
            if buf_tokens >= config.min_tokens and (near_top or buf_tokens >= 0.8 * config.target_tokens):
                flush()
            while heading_stack and heading_stack[-1][0] <= unit.level_size + 0.1:
                heading_stack.pop()
            heading_stack.append((unit.level_size, " ".join(unit.lines[0].text.split())[:120]))
            add(index, unit)
            continue

        tokens = unit.tokens
        too_big = buf_tokens + tokens > config.target_tokens and buf_tokens >= config.min_tokens
        too_many_rows = unit.is_row and buf_rows >= config.max_rows_per_chunk
        # prefer page boundaries once the chunk is reasonably full (keeps one-page project sheets whole)
        page_break = new_page and buf_tokens >= 0.6 * config.target_tokens
        if buf and (too_big or too_many_rows or page_break):
            flush()
        if tokens > config.max_tokens:
            # pathological unit (e.g. a giant paragraph): split by lines, keeping page markers
            for start in range(0, len(unit.lines), 40):
                add(index, _Unit(kind=unit.kind, lines=unit.lines[start : start + 40], region=unit.region))
                if buf_tokens >= config.target_tokens:
                    flush()
            continue
        add(index, unit)
    flush()
    return chunks


def neighbour_context(chunks: list[Chunk], index: int, max_chars: int = 2500) -> str:
    """Adjacent-chunk text for the 'extract with adjacent context' route."""
    parts = []
    if index > 0:
        parts.append("Previous chunk (context only):\n" + chunks[index - 1].text[-max_chars:])
    if index + 1 < len(chunks):
        parts.append("Next chunk (context only):\n" + chunks[index + 1].text[:max_chars])
    return "\n\n".join(parts)
