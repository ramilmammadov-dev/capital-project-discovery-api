"""Layout-aware PDF parsing with local, open-source tools only (pdfplumber / pypdfium2 / tesseract).

For every page we keep:

* the exact 1-based PDF page number and the printed page label (header/footer) when present,
* lines rebuilt from word coordinates so that table columns stay aligned (rows and columns survive
  as text, which is what the extraction model needs to tie a project to its amounts and years),
* font size / weight per line -> heading detection for section tracking,
* ruled tables (cell grid) detected by pdfplumber,
* a local OCR fallback (tesseract) for scanned pages — never a paid OCR service.

pdfplumber/pdfminer is pure Python and MIT-licensed (PyMuPDF was avoided because of AGPL).
Pages are processed in parallel worker processes.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from capex.models import ParsedDocument, ParsedLine, ParsedPage, ParsedTable

log = logging.getLogger(__name__)

PARSER_VERSION = "pdfplumber-layout-4"

# PDFium (pypdfium2) is not thread-safe: every call into it is serialized process-wide.
# (Concurrent probes from the asyncio thread pool otherwise crash the interpreter.)
_PDFIUM_LOCK = threading.Lock()

_BOLD = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)
_PAGE_LABEL = re.compile(
    r"^(?:page\s+)?(?P<label>(?:[A-Z]{1,3}\s?-\s?)?\d{1,4}|[ivxlc]{1,6})(?:\s+of\s+\d+)?$", re.IGNORECASE
)
_TRAILING_LABEL = re.compile(r"(?:^|\s{2,}|\|\s*)(?:page\s+)?(?P<label>(?:[A-Z]{1,3}\s?-\s?)?\d{1,4})\s*$", re.IGNORECASE)


def page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()


def pdf_metadata(path: Path) -> dict[str, Any]:
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            meta = pdf.get_metadata_dict(skip_empty=True)
            return {k.lower(): v for k, v in meta.items()}
        except Exception:  # pragma: no cover - malformed metadata
            return {}
        finally:
            pdf.close()


def fast_page_texts(path: Path, pages: list[int]) -> dict[int, str]:
    """Plain text of selected 1-based pages via pdfium (milliseconds/page) — used for probing."""
    with _PDFIUM_LOCK:
        return _fast_page_texts(path, pages)


def _fast_page_texts(path: Path, pages: list[int]) -> dict[int, str]:
    import pypdfium2 as pdfium

    out: dict[int, str] = {}
    pdf = pdfium.PdfDocument(str(path))
    try:
        total = len(pdf)
        for number in pages:
            if not 1 <= number <= total:
                continue
            page = pdf[number - 1]
            try:
                textpage = page.get_textpage()
                out[number] = textpage.get_text_range() or ""
                textpage.close()
            except Exception:  # pragma: no cover - broken page
                out[number] = ""
            finally:
                page.close()
    finally:
        pdf.close()
    return out


# --------------------------------------------------------------------------------------------------
# Page-level parsing (runs inside worker processes)
# --------------------------------------------------------------------------------------------------


def _group_lines(words: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_top = None
    for word in words:
        mid = (word["top"] + word["bottom"]) / 2
        if current and current_top is not None and abs(mid - current_top) > tolerance:
            lines.append(sorted(current, key=lambda w: w["x0"]))
            current = []
        if not current:
            current_top = mid
        current.append(word)
    if current:
        lines.append(sorted(current, key=lambda w: w["x0"]))
    return lines


def _render_line(words: list[dict[str, Any]], char_w: float) -> tuple[str, bool]:
    """Place words at character columns proportional to x0 so columns line up across rows."""
    buf = ""
    gaps = 0
    prev_x1 = None
    for word in words:
        col = round(word["x0"] / char_w)
        if prev_x1 is not None and word["x0"] - prev_x1 > 2.8 * char_w:
            gaps += 1
        if buf:
            pad = max(1, col - len(buf))
            buf += " " * pad
        else:
            buf = " " * min(col, 60)
        buf += word["text"]
        prev_x1 = word["x1"]
    return buf.rstrip(), gaps >= 2


def _printed_label(lines: list[ParsedLine], height: float) -> str | None:
    if not lines or not height:
        return None
    footer = [ln for ln in lines if ln.top >= height * 0.88]
    header = [ln for ln in lines if ln.top <= height * 0.08]
    for group in (footer[::-1], header):
        for line in group:
            text = line.text.strip()
            m = _PAGE_LABEL.match(text)
            if m:
                return m.group("label").replace(" ", "")
            m = _TRAILING_LABEL.search(text)
            if m and len(text) > len(m.group("label")) + 6:
                label = m.group("label").replace(" ", "")
                if not re.fullmatch(r"(19|20)\d{2}", label):
                    return label
    return None


def _parse_page(page: Any, number: int) -> ParsedPage:
    width, height = float(page.width), float(page.height)
    try:
        words = page.extract_words(
            x_tolerance=1.5,
            y_tolerance=2.5,
            keep_blank_chars=False,
            use_text_flow=False,
            extra_attrs=["size", "fontname"],
        )
    except Exception as exc:  # pragma: no cover - pdfminer edge cases
        log.warning("page %d: word extraction failed: %s", number, exc)
        words = []
    parsed = ParsedPage(pdf_page=number, width=width, height=height)
    if not words:
        return parsed

    tables: list[ParsedTable] = []
    try:
        for index, table in enumerate(page.find_tables()):
            rows = table.extract()
            if len(rows) >= 2 and max((len(r) for r in rows), default=0) >= 2:
                tables.append(ParsedTable(table_id=index, rows=rows, bbox=tuple(table.bbox)))
    except Exception:  # pragma: no cover - table finder edge cases
        tables = []
    parsed.tables = tables

    widths = [(w["x1"] - w["x0"]) / max(1, len(w["text"])) for w in words if w["text"].strip()]
    char_w = max(2.5, min(8.0, statistics.median(widths) if widths else 5.0))
    sizes = [float(w.get("size") or 0) for w in words]
    median_size = statistics.median(sizes) if sizes else 0.0

    for group in _group_lines(words, tolerance=max(2.0, median_size * 0.35)):
        text, tabular = _render_line(group, char_w)
        if not text.strip():
            continue
        size = max(float(w.get("size") or 0) for w in group)
        bold = sum(bool(_BOLD.search(str(w.get("fontname", "")))) for w in group) >= max(1, len(group) // 2)
        top = min(w["top"] for w in group)
        table_id = None
        for table in tables:
            x0, t0, x1, t1 = table.bbox or (0, 0, 0, 0)
            if t0 - 1 <= top <= t1 + 1 and group[0]["x0"] >= x0 - 2 and group[-1]["x1"] <= x1 + 2:
                table_id = table.table_id
                break
        stripped = text.strip()
        letters = sum(c.isalpha() for c in stripped)
        heading = (
            not tabular
            and table_id is None
            and 3 <= len(stripped) <= 120
            and letters >= max(3, int(len(stripped) * 0.5))
            and not stripped.startswith(("(", "-", "•"))
            and (
                (median_size and size >= median_size * 1.18)
                or (
                    bold
                    and len(stripped.split()) <= 10
                    and not stripped.endswith((".", ",", ";", ":"))
                )
            )
        )
        parsed.lines.append(
            ParsedLine(
                text=text,
                top=round(top, 1),
                x0=round(group[0]["x0"], 1),
                size=round(size, 1),
                bold=bold,
                heading=heading,
                table_id=table_id,
                tabular=tabular or table_id is not None,
            )
        )
    parsed.printed_page = _printed_label(parsed.lines, height)
    return parsed


def _parse_range(path_str: str, start: int, end: int) -> list[dict[str, Any]]:
    import pdfplumber

    out: list[dict[str, Any]] = []
    with pdfplumber.open(path_str) as pdf:
        for index in range(start, end):
            page = pdf.pages[index]
            try:
                out.append(_parse_page(page, index + 1).model_dump())
            except Exception as exc:  # pragma: no cover - defensive: never lose the page number
                log.warning("page %d failed: %s", index + 1, exc)
                out.append(ParsedPage(pdf_page=index + 1).model_dump())
            finally:
                flush = getattr(page, "flush_cache", None)
                if callable(flush):
                    flush()
    return out


# --------------------------------------------------------------------------------------------------
# OCR fallback (local tesseract)
# --------------------------------------------------------------------------------------------------


def _ocr_page(path: Path, number: int) -> list[ParsedLine]:
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            page = pdf[number - 1]
            image = page.render(scale=200 / 72).to_pil()
            page.close()
        finally:
            pdf.close()
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "page.png"
        image.save(png)
        proc = subprocess.run(
            ["tesseract", str(png), "stdout", "--psm", "6", "-c", "preserve_interword_spaces=1"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    lines = []
    for i, raw in enumerate(proc.stdout.splitlines()):
        if raw.strip():
            lines.append(ParsedLine(text=raw.rstrip(), top=float(i * 12), tabular=bool(re.search(r"\S\s{3,}\S.*\s{3,}\S", raw))))
    return lines


# --------------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------------


def parse_pdf(
    path: Path,
    doc_id: str,
    content_hash: str,
    *,
    max_pages: int = 1500,
    workers: int = 4,
    ocr_enabled: bool = True,
    ocr_max_pages: int = 60,
    title: str = "",
) -> ParsedDocument:
    total = page_count(path)
    limit = min(total, max_pages)
    warnings: list[str] = []
    if total > max_pages:
        warnings.append(f"document has {total} pages; only the first {max_pages} were parsed")

    workers = max(1, min(workers, os.cpu_count() or 1, (limit + 24) // 25))
    pages: list[dict[str, Any]] = []
    if workers == 1:
        pages = _parse_range(str(path), 0, limit)
    else:
        step = (limit + workers - 1) // workers
        ranges = [(s, min(limit, s + step)) for s in range(0, limit, step)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_parse_range, str(path), s, e) for s, e in ranges]
            for future in futures:
                pages.extend(future.result())
    parsed_pages = [ParsedPage.model_validate(p) for p in sorted(pages, key=lambda p: p["pdf_page"])]

    empty = [p for p in parsed_pages if not p.lines]
    if empty and ocr_enabled:
        if shutil.which("tesseract"):
            for page in empty[:ocr_max_pages]:
                try:
                    page.lines = _ocr_page(path, page.pdf_page)
                    page.ocr = True
                    page.printed_page = _printed_label(page.lines, page.lines[-1].top if page.lines else 0)
                except Exception as exc:  # pragma: no cover - OCR edge cases
                    warnings.append(f"OCR failed on page {page.pdf_page}: {exc}")
            if len(empty) > ocr_max_pages:
                warnings.append(f"{len(empty) - ocr_max_pages} image-only pages were not OCR'd (limit)")
            warnings.append(f"{min(len(empty), ocr_max_pages)} image-only page(s) read with local OCR (tesseract)")
        else:
            warnings.append(f"{len(empty)} image-only page(s) have no text layer and tesseract is not installed")

    return ParsedDocument(
        doc_id=doc_id,
        content_hash=content_hash,
        file_type="pdf",
        title=title,
        page_count=total,
        pages=parsed_pages,
        parser=PARSER_VERSION,
        parser_version=PARSER_VERSION,
        warnings=warnings,
    )
