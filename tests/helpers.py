"""Shared test helpers: synthetic PDFs, hand-made chunks and small fakes. No network, no real keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

AREAS = ["Harbor", "Cedar", "Mill Creek", "Ridge", "Lakeview", "Summit", "Prairie", "Westgate", "Riverside", "Oakmont"]
KINDS = ["Road Resurfacing", "Water Main Upgrade", "Park Renovation", "Signal Modernization", "Storm Drain Repair",
         "Library Expansion", "Trail Extension"]

# column x positions (points) of the synthetic CIP table
X_ID, X_NAME, X_LOC, X_FY1, X_FY2, X_TOTAL = 36, 100, 268, 380, 450, 520


def project_id(i: int) -> str:
    return f"ST85100{i:03d}"


def project_name(i: int) -> str:
    return f"{AREAS[i % len(AREAS)]} {KINDS[(i // len(AREAS)) % len(KINDS)]}"


def build_cip_pdf(
    path: Path,
    *,
    n_rows: int = 70,
    rows_per_page: int = 26,
    wrap_rows: tuple[int, ...] = (3, 27, 28, 55, 56, 69),
    printed_offset: int = 0,
    title: str = "Capital Improvement Program FY 2027-2031",
    subtitle: str = "City of Exampleville - Adopted Budget",
    narrative: bool = True,
    repeat_header: bool = False,
    filler_words: int = 0,
) -> dict[str, Any]:
    """Write a multi-page itemized CIP table PDF and return what was drawn where.

    Page 1 carries the title and the column header, later pages continue the table without repeating it
    (unless ``repeat_header``). Rows listed in ``wrap_rows`` have a name that wraps onto a second, indented
    line. ``filler_words`` pads every row with description words (makes rows token-heavy).
    """
    _width, height = letter
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setTitle(title)
    meta: dict[str, Any] = {
        "ids": [], "page_of": {}, "names": {}, "continuations": {}, "amounts": {}, "n_table_pages": 0,
        "header": "Project ID Project Name Location FY 2027 FY 2028 Total", "title": title,
    }
    page_no = 0

    def footer() -> None:
        c.setFont("Helvetica", 8)
        c.drawString(36, 28, f"Page {page_no + printed_offset}")

    def table_header(y: float) -> float:
        c.setFont("Helvetica-Bold", 7.5)
        for x, label in ((X_ID, "Project ID"), (X_NAME, "Project Name"), (X_LOC, "Location"),
                         (X_FY1, "FY 2027"), (X_FY2, "FY 2028"), (X_TOTAL, "Total")):
            c.drawString(x, y, label)
        return y - 16

    row = 0
    while row < n_rows:
        page_no += 1
        meta["n_table_pages"] = page_no
        y = height - 60
        if page_no == 1:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(36, y, title)
            y -= 20
            c.setFont("Helvetica", 7.5)
            c.drawString(36, y, subtitle)
            y -= 26
            y = table_header(y)
        elif repeat_header:
            y = table_header(y)
        for _ in range(rows_per_page):
            if row >= n_rows:
                break
            pid, name = project_id(row), project_name(row)
            amount = 1_250_000 + row * 10_000
            c.setFont("Helvetica", 7.5)
            c.drawString(X_ID, y, pid)
            c.drawString(X_NAME, y, name)
            c.drawString(X_LOC, y, f"Zone {chr(65 + row % 6)} district")
            c.drawString(X_FY1, y, f"${amount:,}")
            c.drawString(X_FY2, y, f"${amount // 2:,}")
            c.drawString(X_TOTAL, y, f"${amount + amount // 2:,}")
            meta["ids"].append(pid)
            meta["page_of"][pid] = page_no
            meta["names"][pid] = name
            meta["amounts"][pid] = f"${amount:,}"
            y -= 14
            sub = y + 5  # baseline of the first extra line under this row
            if row in wrap_rows:
                cont = f"Wrapped Name Part {row:03d}"
                c.drawString(X_NAME, sub, cont)  # second line of the name, indented under the name column
                meta["continuations"][pid] = cont
                sub -= 9
                y -= 9
            if filler_words:
                words = " ".join(f"scopeword{row}x{k}" for k in range(filler_words))
                c.drawString(X_NAME, sub, words[:75])
                meta.setdefault("fillers", {})[pid] = f"scopeword{row}x0"
                sub -= 9
                y -= 9
            row += 1
        footer()
        c.showPage()

    if narrative:
        page_no += 1
        y = height - 70
        c.setFont("Helvetica-Bold", 14)
        c.drawString(36, y, "Funding Policy and Narrative")
        y -= 30
        c.setFont("Helvetica", 10)
        for k in range(14):
            c.drawString(
                36, y,
                f"The City reviews its capital needs every year and keeps reserves for emergencies, item {k}.",
            )
            y -= 14
        footer()
        c.showPage()
    meta["n_pages"] = page_no
    c.save()
    return meta


def build_summary_pdf(path: Path, title: str = "Annual Budget Summary FY 2027") -> None:
    """Aggregate-only budget document: department totals, no itemized capital projects."""
    _width, height = letter
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setTitle(title)
    for page in range(1, 4):
        y = height - 70
        c.setFont("Helvetica-Bold", 14)
        c.drawString(36, y, title if page == 1 else f"Fund Overview {page}")
        y -= 30
        c.setFont("Helvetica", 10)
        for k in range(18):
            c.drawString(36, y, f"The General Fund supports public services across all departments, section {k}.")
            y -= 14
        c.drawString(36, y - 10, "Total capital budget for the fiscal year is $48,000,000 across all departments.")
        c.setFont("Helvetica", 8)
        c.drawString(36, 28, f"Page {page}")
        c.showPage()
    c.save()


# --------------------------------------------------------------------------------------------------
# Hand-made models
# --------------------------------------------------------------------------------------------------

from capex.models import Chunk, DocStatus, DocumentCandidate  # noqa: E402
from capex.normalize import sha256_text  # noqa: E402

TWO_PAGE_TEXT = """[[PDF page 4]]
Project ID   Project Name                    Location         FY 2027      Total
ST85100001   Harbor Road Resurfacing         Zone A district  $5,300       $5,300
ST85100002   Cedar Water Main Upgrade        Zone B district  $1,200       $1,200
Total Capital Program                                         $6,500
[[PDF page 5 | printed page 51]]
ST85100003   Ridge Park Renovation           Zone C district  $900         $900
             Design completed in 2025; construction FY 2028
Subtotal Parks                                                $900"""


def make_chunk(text: str = TWO_PAGE_TEXT, pages: tuple[int, ...] = (4, 5), seq: int = 1, doc_id: str = "doc_x",
               printed: dict[int, str] | None = None, section: str = "Capital Projects", est_rows: int = 3) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{seq:04d}", doc_id=doc_id, seq=seq, pages=list(pages),
        printed_pages=printed if printed is not None else {5: "51"}, section=section, kind="table",
        text=text, content_hash=sha256_text(text), est_tokens=max(1, len(text) // 4), est_rows=est_rows,
    )


def make_document(doc_id: str = "doc_x", status: DocStatus = DocStatus.ADOPTED, fiscal_year: int | None = 2027,
                  url: str = "https://www.examplecity.gov/files/cip.pdf", title: str = "FY2027-2031 CIP",
                  **extra: Any) -> DocumentCandidate:
    return DocumentCandidate(
        doc_id=doc_id, url=url, title=title, file_type="pdf", domain="examplecity.gov", is_official_domain=True,
        doc_status=status, document_fiscal_year=fiscal_year, **extra,
    )


# --------------------------------------------------------------------------------------------------
# Fake LLM client (registered with LLMRegistry.register); records every call like the real client
# --------------------------------------------------------------------------------------------------

import re  # noqa: E402
from collections.abc import Callable  # noqa: E402

from capex.llm import LLMClient, LLMResponse  # noqa: E402

ROW_RE = re.compile(
    r"^\s*(?P<id>ST\d{8})\s+(?P<name>.+?)\s+Zone [A-F] district\s+(?P<amount>\$[\d,]+)"
)


def llm_ok(data: Any, model: str = "fake") -> LLMResponse:
    return LLMResponse(data=data, text="", model=model, stop_reason="end_turn", input_tokens=1000, output_tokens=200)


def llm_truncated(model: str = "fake") -> LLMResponse:
    return LLMResponse(data=None, text="{", model=model, stop_reason="max_tokens", input_tokens=1000, output_tokens=8000)


def llm_invalid(model: str = "fake") -> LLMResponse:
    return LLMResponse(data=None, text="not json", model=model, stop_reason="end_turn", input_tokens=1000, output_tokens=5)


def content_of(user: str) -> str:
    return user.split("CONTENT:\n", 1)[1] if "CONTENT:\n" in user else user


def projects_from_user(user: str, *, limit: int | None = None) -> dict[str, Any]:
    """Extraction answer for a prompt: one project per table row found after 'CONTENT:' (grounded by design)."""
    projects: list[dict[str, Any]] = []
    page = 1
    for line in content_of(user).splitlines():
        marker = re.match(r"\[\[PDF page (\d+)", line.strip())
        if marker:
            page = int(marker.group(1))
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        projects.append({
            "project_name": " ".join(m.group("name").split()),
            "project_id": m.group("id"),
            "budget_text": m.group("amount"),
            "source_excerpt": " ".join(line.split()),
            "source_pages": [page],
            "current_phase": "funded",
            "phase_is_explicit": False,
            "funding_status": "funded",
            "extraction_confidence": 0.92,
            "domain": "transportation",
            "budget_type": "adopted",
        })
    return {"amount_unit": "dollars", "projects": projects[:limit] if limit else projects}


class FakeLLM(LLMClient):
    """Scriptable stand-in for the Anthropic client. ``handler(call) -> LLMResponse``."""

    provider = "anthropic"

    def __init__(self, handler: Callable[[dict[str, Any]], LLMResponse]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    async def json_completion(self, *, model, system, user, schema, max_tokens, stage, task, ledger,
                              schema_name="output") -> LLMResponse:
        call = {"model": model, "system": system, "user": user, "schema": schema, "max_tokens": max_tokens,
                "stage": stage, "task": task, "schema_name": schema_name}
        self.calls.append(call)
        response = self.handler(call)
        ledger.record(stage=stage, task=task, provider=self.provider, model=model,
                      input_tokens=response.input_tokens, output_tokens=response.output_tokens)
        return response

    def calls_for(self, schema_name: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["schema_name"] == schema_name]


def table_chunk(n_rows: int = 10, *, two_pages: bool = False, doc_id: str = "doc_x"):
    """A chunk of ``n_rows`` project rows (optionally continuing onto a second page)."""
    lines = ["[[PDF page 4]]"]
    for i in range(n_rows):
        if two_pages and i == n_rows // 2:
            lines.append("[[PDF page 5]]")
        lines.append(
            f"ST851{i:05d}   {AREAS[i % len(AREAS)]} {KINDS[i % len(KINDS)]}   Zone {chr(65 + i % 6)} district   "
            f"${1_000_000 + i * 1000:,}   ${500_000 + i * 500:,}"
        )
    chunk = make_chunk("\n".join(lines), pages=(4, 5) if two_pages else (4,), doc_id=doc_id, printed={})
    return chunk.model_copy(update={"est_rows": n_rows})
