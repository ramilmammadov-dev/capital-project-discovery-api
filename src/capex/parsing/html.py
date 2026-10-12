"""HTML documents -> the same line/heading/table structure as PDFs (one logical page).

Navigation, scripts and footers are dropped; headings become section boundaries; table rows are
rendered as ``cell | cell | cell`` lines (header row kept with the table) so the chunker can treat
an HTML project table exactly like a PDF one.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, NavigableString, Tag

from capex.models import ParsedDocument, ParsedLine, ParsedPage, ParsedTable
from capex.normalize import clean_text

PARSER_VERSION = "html-2"

_DROP = ["script", "style", "noscript", "nav", "footer", "header", "form", "svg", "iframe", "button", "aside"]
_BLOCKS = {"p", "li", "dd", "dt", "blockquote", "pre", "figcaption", "address"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def html_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return clean_text(soup.title.string)
    h1 = soup.find("h1")
    return clean_text(h1.get_text(" ")) if h1 else ""


def parse_html(html: str, doc_id: str, content_hash: str, title: str = "") -> ParsedDocument:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_DROP):
        tag.decompose()
    root = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup  # type: ignore[call-overload]

    lines: list[ParsedLine] = []
    tables: list[ParsedTable] = []
    y = 0.0

    def emit(text: str, *, heading: bool = False, table_id: int | None = None, tabular: bool = False) -> None:
        nonlocal y
        text = clean_text(text)
        if text:
            lines.append(ParsedLine(text=text, top=y, heading=heading, table_id=table_id, tabular=tabular, bold=heading))
            y += 12.0

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name or ""
            if name in _HEADINGS:
                emit(child.get_text(" "), heading=True)
            elif name == "table":
                table_id = len(tables)
                rows: list[list[str | None]] = []
                for tr in child.find_all("tr"):
                    cells = [clean_text(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
                    if any(cells):
                        rows.append(list(cells))
                        emit(" | ".join(cells), table_id=table_id, tabular=True)
                tables.append(ParsedTable(table_id=table_id, rows=rows))
            elif name in _BLOCKS:
                if child.find(["table", *_HEADINGS]):
                    walk(child)
                else:
                    emit(child.get_text(" "))
            else:
                if child.find(["p", "li", "table", *_HEADINGS]):
                    walk(child)
                else:
                    emit(child.get_text(" "))

    walk(root)
    page = ParsedPage(pdf_page=1, lines=lines, tables=tables)
    return ParsedDocument(
        doc_id=doc_id,
        content_hash=content_hash,
        file_type="html",
        title=title or html_title(html),
        page_count=1,
        pages=[page],
        parser=PARSER_VERSION,
        parser_version=PARSER_VERSION,
    )
