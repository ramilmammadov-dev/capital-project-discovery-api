"""Layout-aware chunking on a real, generated PDF (reportlab) and on an HTML table page."""

from __future__ import annotations

import re

import pytest

from capex.chunking import chunk_document, neighbour_context
from capex.config import ChunkingConfig
from capex.parsing.html import parse_html
from capex.parsing.pdf import parse_pdf

from .helpers import build_cip_pdf

PAGE_MARKER = re.compile(r"\[\[PDF page (\d+)(?: \| printed page ([^\]]+))?\]\]")


@pytest.fixture(scope="module")
def cip(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "cip.pdf"
    meta = build_cip_pdf(path)
    doc = parse_pdf(path, "doc_test", "hash", workers=1)
    return meta, doc, chunk_document(doc, ChunkingConfig())


def squash(text: str) -> str:
    """Collapse the column padding the layout parser adds between words."""
    return " ".join(text.split())


def row_page(chunk_text: str, pid: str) -> int:
    """PDF page the marker preceding ``pid`` in the chunk text announces."""
    page = None
    for line in chunk_text.splitlines():
        marker = PAGE_MARKER.fullmatch(line.strip())
        if marker:
            page = int(marker.group(1))
        elif pid in line:
            assert page is not None, f"{pid} appears before any page marker"
            return page
    raise AssertionError(f"{pid} not found")


class TestParsedPdf:
    def test_pages_and_headings(self, cip):
        meta, doc, _ = cip
        assert doc.page_count == meta["n_pages"] == 4
        assert [p.pdf_page for p in doc.pages] == [1, 2, 3, 4]
        headings = [" ".join(ln.text.split()) for p in doc.pages for ln in p.lines if ln.heading]
        assert any("Capital Improvement Program" in h for h in headings)
        assert any("Funding Policy" in h for h in headings)

    def test_project_rows_are_tabular_lines(self, cip):
        meta, doc, _ = cip
        tabular = [ln.text for p in doc.pages for ln in p.lines if ln.tabular]
        for pid in meta["ids"]:
            assert sum(pid in line for line in tabular) == 1


class TestChunks:
    def test_every_project_id_is_in_exactly_one_chunk(self, cip):
        meta, _, chunks = cip
        for pid in meta["ids"]:
            holders = [c.chunk_id for c in chunks if pid in c.text]
            assert len(holders) == 1, f"{pid} found in {holders}"
        assert len(meta["ids"]) == 70

    def test_ids_never_appear_in_context_only_text(self, cip):
        meta, _, chunks = cip
        # Context (repeated header / preceding lines) is not extractable content, but IDs may be repeated
        # as the 'preceding lines' overlap; the extractable text must still contain each ID exactly once.
        for pid in meta["ids"]:
            assert sum(c.text.count(pid) for c in chunks) == 1

    def test_wrapped_name_stays_with_its_row(self, cip):
        meta, _, chunks = cip
        assert len(meta["continuations"]) >= 5
        for pid, continuation in meta["continuations"].items():
            holder = next(c for c in chunks if pid in c.text)
            assert continuation in squash(holder.text), f"{pid}: wrapped text is not in the chunk of its row"
            lines = holder.text.splitlines()
            row_index = next(i for i, line in enumerate(lines) if pid in line)
            # the continuation follows its row directly (page markers aside)
            following = [ln for ln in lines[row_index + 1 : row_index + 3] if not PAGE_MARKER.fullmatch(ln.strip())]
            assert continuation in squash(following[0])

    def test_rows_keep_their_amounts(self, cip):
        meta, _, chunks = cip
        for pid in meta["ids"]:
            holder = next(c for c in chunks if pid in c.text)
            row = next(line for line in holder.text.splitlines() if pid in line)
            assert meta["amounts"][pid] in row
            assert meta["names"][pid] in row

    def test_chunks_that_start_mid_table_repeat_the_column_header(self, cip):
        meta, _, chunks = cip
        first_table_chunk = next(c for c in chunks if meta["ids"][0] in c.text)
        assert "Table header (repeated)" not in first_table_chunk.context
        mid_table = [c for c in chunks if c.kind != "text" and c is not first_table_chunk and c.est_rows > 0]
        assert mid_table, "expected more than one table chunk"
        for chunk in mid_table:
            assert "Table header (repeated)" in chunk.context, chunk.chunk_id
            assert "Project ID" in chunk.context
            assert "Location" in chunk.context

    def test_page_markers_match_the_pages_the_rows_were_drawn_on(self, cip):
        meta, _, chunks = cip
        for pid in meta["ids"]:
            holder = next(c for c in chunks if pid in c.text)
            assert row_page(holder.text, pid) == meta["page_of"][pid], pid
        # every chunk starts with a page marker and lists the pages it spans
        for chunk in chunks:
            assert PAGE_MARKER.match(chunk.text.splitlines()[0])
            markers = [int(m.group(1)) for m in PAGE_MARKER.finditer(chunk.text)]
            assert chunk.pages == markers
            assert chunk.pages == sorted(chunk.pages)

    def test_table_chunks_estimate_rows_and_respect_the_row_cap(self, cip):
        _, _, chunks = cip
        config = ChunkingConfig()
        table_chunks = [c for c in chunks if c.est_rows > 0]
        assert len(table_chunks) >= 3
        assert sum(c.est_rows for c in table_chunks) >= 70
        for chunk in table_chunks:
            assert chunk.est_rows <= config.max_rows_per_chunk + 2, chunk.chunk_id
            row_lines = sum(1 for line in chunk.text.splitlines() if re.search(r"ST85100\d{3}", line))
            assert chunk.est_rows == row_lines

    def test_narrative_page_is_its_own_text_chunk_with_section(self, cip):
        _, _, chunks = cip
        narrative = next(c for c in chunks if "reviews its capital needs" in c.text)
        assert narrative.kind == "text"
        assert narrative.est_rows == 0
        assert " ".join(narrative.section.split()) == "Funding Policy and Narrative"
        assert narrative.pages == [4]

    def test_table_chunks_carry_the_section_title(self, cip):
        _, _, chunks = cip
        assert "Capital Improvement Program FY 2027-2031" in " ".join(chunks[0].section.split())
        assert chunks[0].context.startswith("Section: ")

    def test_section_titles_are_whitespace_normalized(self, cip):
        _, _, chunks = cip
        narrative = next(c for c in chunks if "reviews its capital needs" in c.text)
        assert narrative.section == "Funding Policy and Narrative"

    def test_running_footers_are_removed(self, cip):
        _, _, chunks = cip
        assert not any(re.search(r"^\s*Page \d+\s*$", c.text, re.MULTILINE) for c in chunks)

    def test_chunk_identity_fields(self, cip):
        _, doc, chunks = cip
        assert [c.seq for c in chunks] == list(range(len(chunks)))
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert all(c.chunk_id == f"{doc.doc_id}:{c.seq:04d}" for c in chunks)
        assert all(c.doc_id == "doc_test" and c.est_tokens > 0 for c in chunks)
        assert len({c.content_hash for c in chunks}) == len(chunks)

    def test_chunking_is_deterministic(self, cip):
        _, doc, chunks = cip
        again = chunk_document(doc, ChunkingConfig())
        assert [c.content_hash for c in again] == [c.content_hash for c in chunks]

    def test_small_row_cap_still_keeps_every_row_whole(self, cip):
        meta, doc, _ = cip
        chunks = chunk_document(doc, ChunkingConfig(max_rows_per_chunk=10))
        assert max(c.est_rows for c in chunks) <= 10
        assert len(chunks) > len([c for c in chunk_document(doc, ChunkingConfig())])
        for pid in meta["ids"]:
            holders = [c for c in chunks if pid in c.text]
            assert len(holders) == 1
            if pid in meta["continuations"]:
                assert meta["continuations"][pid] in squash(holders[0].text)

    def test_overlap_lines_are_context_only(self, cip):
        _, doc, _ = cip
        chunks = chunk_document(doc, ChunkingConfig(overlap_lines=3))
        assert "Preceding lines" not in chunks[0].context
        assert all("Preceding lines (context only" in c.context for c in chunks[1:])
        no_overlap = chunk_document(doc, ChunkingConfig(overlap_lines=0))
        assert all("Preceding lines" not in c.context for c in no_overlap)

    def test_neighbour_context(self, cip):
        _, _, chunks = cip
        middle = neighbour_context(chunks, 1)
        assert "Previous chunk (context only)" in middle and "Next chunk (context only)" in middle
        assert "Previous chunk" not in neighbour_context(chunks, 0)
        assert "Next chunk" not in neighbour_context(chunks, len(chunks) - 1)
        assert len(neighbour_context(chunks, 1, max_chars=100)) < 400


class TestPrintedPages:
    def test_printed_page_labels_are_cited(self, tmp_path):
        path = tmp_path / "offset.pdf"
        meta = build_cip_pdf(path, n_rows=40, printed_offset=100)
        doc = parse_pdf(path, "doc_off", "hash", workers=1)
        chunks = chunk_document(doc, ChunkingConfig())
        assert any("[[PDF page 1 | printed page 101]]" in c.text for c in chunks)
        first = next(c for c in chunks if meta["ids"][0] in c.text)
        assert first.printed_pages[1] == "101"
        assert doc.pages[0].printed_page == "101"


class TestRepeatedHeaders:
    def test_header_repeated_on_each_page_is_not_lost_or_duplicated_as_a_project(self, tmp_path):
        path = tmp_path / "repeat.pdf"
        meta = build_cip_pdf(path, n_rows=60, repeat_header=True)
        doc = parse_pdf(path, "doc_rep", "hash", workers=1)
        chunks = chunk_document(doc, ChunkingConfig())
        for pid in meta["ids"]:
            assert sum(pid in c.text for c in chunks) == 1


class TestHtml:
    HTML = """
    <html><head><title>Capital Plan</title></head><body>
    <nav>Home | About</nav>
    <main>
      <h1>Capital Improvement Plan FY 2027-2031</h1>
      <p>The following projects were adopted by Council.</p>
      <table>
        <tr><th>Project ID</th><th>Project Name</th><th>FY 2027</th><th>Total</th></tr>
        <tr><td>ST85100001</td><td>Harbor Road Resurfacing</td><td>$1,250,000</td><td>$2,500,000</td></tr>
        <tr><td>ST85100002</td><td>Cedar Water Main Upgrade</td><td>$900,000</td><td>$1,800,000</td></tr>
        <tr><td>ST85100003</td><td>Ridge Park Renovation</td><td>$450,000</td><td>$450,000</td></tr>
      </table>
      <script>var x = 1;</script>
    </main>
    <footer>Copyright</footer>
    </body></html>
    """

    def test_parse_html_structure(self):
        doc = parse_html(self.HTML, "doc_html", "h")
        assert doc.file_type == "html"
        assert doc.title == "Capital Plan"
        assert doc.page_count == 1 and len(doc.pages) == 1
        page = doc.pages[0]
        texts = [ln.text for ln in page.lines]
        assert not any("Home" in t or "Copyright" in t or "var x" in t for t in texts)
        assert any(ln.heading and "Capital Improvement Plan" in ln.text for ln in page.lines)
        assert "ST85100001 | Harbor Road Resurfacing | $1,250,000 | $2,500,000" in texts
        assert len(page.tables) == 1 and len(page.tables[0].rows) == 4
        assert all(ln.tabular and ln.table_id == 0 for ln in page.lines if "|" in ln.text)

    def test_html_title_falls_back_to_h1_and_explicit_title_wins(self):
        no_title = "<html><body><h1> My   Plan </h1></body></html>"
        assert parse_html(no_title, "d", "h").title == "My Plan"
        assert parse_html(no_title, "d", "h", title="Given").title == "Given"

    def test_html_table_chunks_into_rows_with_page_marker(self):
        doc = parse_html(self.HTML, "doc_html", "h")
        chunks = chunk_document(doc, ChunkingConfig())
        assert chunks
        joined = "\n".join(c.text for c in chunks)
        for pid in ("ST85100001", "ST85100002", "ST85100003"):
            assert joined.count(pid) == 1
        assert all(c.text.startswith("[[PDF page 1]]") or "[[PDF page 1]]" in c.text for c in chunks)
        assert sum(c.est_rows for c in chunks) == 3
        assert any("Capital Improvement Plan" in c.section for c in chunks)
