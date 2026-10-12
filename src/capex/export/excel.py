"""Excel deliverable (spec §10): one workbook per organization.

Sheets (in this order): Projects, Sources, Project References, Extraction & Cost Metrics, then
Discovered Documents, Dedup Audit and About. Data sheets are Excel tables (filter + sort on every
column: domain, phase, budget, location, years, funding status...). Source links open the PDF at the
cited page (``#page=N``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xlsxwriter

from capex.models import JobResult, Phase
from capex.normalize import page_link

_MAX_CELL = 32000


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "; ".join(str(v) for v in value if v not in (None, ""))
    if hasattr(value, "value") and not isinstance(value, str | int | float):
        value = value.value  # enums
    if isinstance(value, str) and len(value) > _MAX_CELL:
        return value[:_MAX_CELL] + "…"
    return value


class _Book:
    def __init__(self, path: Path) -> None:
        self.wb = xlsxwriter.Workbook(str(path), {"strings_to_urls": False, "constant_memory": False})
        wb = self.wb
        self.f_wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        self.f_top = wb.add_format({"valign": "top"})
        self.f_money = wb.add_format({"num_format": "$#,##0", "valign": "top"})
        self.f_money4 = wb.add_format({"num_format": "$#,##0.0000", "valign": "top"})
        self.f_int = wb.add_format({"num_format": "0", "valign": "top"})
        self.f_num = wb.add_format({"num_format": "#,##0", "valign": "top"})
        self.f_pct = wb.add_format({"num_format": "0.00", "valign": "top"})
        self.f_link = wb.add_format({"font_color": "#1F4E79", "underline": 1, "valign": "top"})
        self.f_title = wb.add_format({"bold": True, "font_size": 14})
        self.f_h2 = wb.add_format({"bold": True, "font_size": 11, "font_color": "#1F4E79"})
        self.f_bold = wb.add_format({"bold": True})
        self.f_note = wb.add_format({"italic": True, "font_color": "#595959", "text_wrap": True, "valign": "top"})

    def table(
        self,
        name: str,
        columns: list[tuple[str, int, str]],
        rows: list[list[Any]],
        links: dict[int, int] | None = None,
        sheet_name: str | None = None,
        start_row: int = 0,
        ws: Any = None,
    ) -> Any:
        """columns: (header, width, kind) — kind in text|wrap|money|money4|int|num|pct|link."""
        own_sheet = ws is None
        ws = ws or self.wb.add_worksheet(sheet_name or name)
        fmt = {
            "text": self.f_top,
            "wrap": self.f_wrap,
            "money": self.f_money,
            "money4": self.f_money4,
            "int": self.f_int,
            "num": self.f_num,
            "pct": self.f_pct,
            "link": self.f_link,
        }
        for c, (_, width, _kind) in enumerate(columns):
            ws.set_column(c, c, width)
        for r, row in enumerate(rows, start=start_row + 1):
            for c, value in enumerate(row):
                kind = columns[c][2]
                value = _cell(value)
                if kind == "link" and isinstance(value, tuple):
                    url, label = value
                    if url and url.startswith("http") and len(url) < 2000:
                        ws.write_url(r, c, url, self.f_link, string=label or url)
                    else:
                        ws.write(r, c, label or url or "", self.f_top)
                elif value == "":
                    ws.write_blank(r, c, None, fmt[kind])
                else:
                    ws.write(r, c, value, fmt[kind])
        last_row = start_row + max(1, len(rows))
        ws.add_table(
            start_row,
            0,
            last_row,
            len(columns) - 1,
            {
                "name": name.replace(" ", "_").replace("&", "and"),
                "columns": [{"header": h} for h, _, _ in columns],
                "style": "Table Style Medium 2",
                "autofilter": True,
            },
        )
        if own_sheet:
            ws.freeze_panes(start_row + 1, 2)
        return ws

    def close(self) -> None:
        self.wb.close()


def write_workbook(result: JobResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    book = _Book(path)
    docs = {d.doc_id: d for d in result.selected_documents}

    # Sheet 1 — Projects ---------------------------------------------------------------------------
    project_cols = [
        ("project_uid", 16, "text"),
        ("organization_name", 22, "text"),
        ("project_id", 16, "text"),
        ("project_name", 42, "wrap"),
        ("project_description", 60, "wrap"),
        ("domain", 16, "text"),
        ("project_type", 24, "wrap"),
        ("location", 28, "wrap"),
        ("estimated_budget", 16, "money"),
        ("budget_currency", 9, "text"),
        ("budget_type", 13, "text"),
        ("budget_basis", 26, "wrap"),
        ("document_fiscal_year", 11, "int"),
        ("anticipated_start_year", 11, "int"),
        ("anticipated_construction_year", 13, "int"),
        ("anticipated_completion_year", 12, "int"),
        ("current_phase", 22, "text"),
        ("phase_evidence", 40, "wrap"),
        ("phase_basis", 10, "text"),
        ("funding_status", 15, "text"),
        ("funding_sources", 26, "wrap"),
        ("source_document", 40, "wrap"),
        ("source_url", 22, "link"),
        ("source_pages", 10, "text"),
        ("printed_pages", 10, "text"),
        ("source_section", 30, "wrap"),
        ("source_excerpt", 50, "wrap"),
        ("extraction_confidence", 11, "pct"),
        ("source_count", 8, "int"),
        ("other_references", 34, "wrap"),
        ("budget_conflicts", 34, "wrap"),
        ("phase_history", 34, "wrap"),
        ("aliases", 30, "wrap"),
        ("needs_review", 9, "text"),
        ("review_reasons", 30, "wrap"),
        ("possible_duplicates", 20, "wrap"),
    ]
    rows = []
    for p in result.projects:
        primary_page = p.source_pages[0] if p.source_pages else None
        doc = docs.get(p.references[0].doc_id) if p.references else None
        is_pdf = doc.file_type == "pdf" if doc else True
        url = page_link(p.source_url, primary_page, is_pdf)
        printed = p.references[0].printed_pages if p.references else []
        others = [
            f"{r.document_title[:50]} p.{','.join(map(str, r.pdf_pages))}"
            for r in p.references[1:]
        ]
        conflicts = [
            f"{c.get('document', '')[:40]}: ${c.get('estimated_budget', 0):,.0f} ({c.get('budget_basis') or 'n/a'})"
            for c in p.budget_conflicts
        ]
        history = [f"{h.get('document', '')[:40]}: {h.get('phase')}" for h in p.phase_history]
        rows.append(
            [
                p.project_uid,
                p.organization_name,
                p.project_id,
                p.project_name,
                p.project_description,
                p.domain,
                p.project_type,
                p.location,
                p.estimated_budget,
                p.budget_currency,
                p.budget_type,
                p.budget_basis,
                p.document_fiscal_year,
                p.anticipated_start_year,
                p.anticipated_construction_year,
                p.anticipated_completion_year,
                p.current_phase.value,
                p.phase_evidence,
                p.phase_basis,
                p.funding_status.value,
                p.funding_sources,
                p.source_document,
                (url, f"Open p.{primary_page}" if primary_page and is_pdf else "Open source"),
                ", ".join(map(str, p.source_pages)),
                ", ".join(printed),
                p.source_section,
                p.source_excerpt,
                p.extraction_confidence,
                len(p.references),
                others,
                conflicts,
                history,
                p.aliases,
                "yes" if p.needs_review else "no",
                p.review_reasons,
                p.possible_duplicates,
            ]
        )
    book.table("Projects", project_cols, rows)

    # Sheet 2 — Sources ----------------------------------------------------------------------------
    source_cols = [
        ("rank", 6, "int"),
        ("title", 44, "wrap"),
        ("url", 24, "link"),
        ("publisher", 24, "wrap"),
        ("publication_date", 13, "text"),
        ("coverage_years", 16, "text"),
        ("document_type", 26, "wrap"),
        ("status", 16, "text"),
        ("pages", 8, "int"),
        ("ranking_score", 10, "pct"),
        ("score_breakdown", 40, "wrap"),
        ("estimated_projects", 11, "num"),
        ("records_extracted", 11, "int"),
        ("selection_rationale", 70, "wrap"),
    ]
    per_doc_records: dict[str, int] = {}
    for r in result.records:
        per_doc_records[r.doc_id] = per_doc_records.get(r.doc_id, 0) + 1
    source_rows = []
    for d in sorted(result.selected_documents, key=lambda x: x.selection_rank or 99):
        breakdown = ", ".join(f"{k}={v:.2f}" for k, v in d.score_breakdown.items())
        source_rows.append(
            [
                d.selection_rank,
                d.title,
                (d.final_url or d.url, "Open document"),
                d.publisher,
                d.publication_date,
                d.coverage_years,
                d.doc_type.value,
                d.doc_status.value,
                d.probe.page_count if d.probe else None,
                d.score,
                breakdown,
                d.probe.est_projects if d.probe else None,
                per_doc_records.get(d.doc_id, 0),
                d.selection_rationale,
            ]
        )
    book.table("Sources", source_cols, source_rows)

    # Sheet 3 — Project References -----------------------------------------------------------------
    ref_cols = [
        ("project_uid", 16, "text"),
        ("project_name", 40, "wrap"),
        ("original_project_name", 40, "wrap"),
        ("original_project_id", 16, "text"),
        ("document", 40, "wrap"),
        ("pdf_page", 9, "text"),
        ("printed_page", 10, "text"),
        ("open_at_page", 14, "link"),
        ("section", 30, "wrap"),
        ("source_excerpt", 60, "wrap"),
        ("document_fiscal_year", 11, "int"),
        ("budget_in_source", 15, "money"),
        ("phase_in_source", 20, "text"),
        ("reference_verified", 10, "text"),
        ("record_id", 24, "text"),
    ]
    ref_rows = []
    for p in result.projects:
        for ref in p.references:
            ref_rows.append(
                [
                    p.project_uid,
                    p.project_name,
                    ref.original_project_name,
                    ref.original_project_id,
                    ref.document_title,
                    ", ".join(map(str, ref.pdf_pages)),
                    ", ".join(ref.printed_pages),
                    (ref.page_url, f"Open p.{ref.pdf_pages[0]}" if ref.pdf_pages else "Open"),
                    ref.section,
                    ref.excerpt,
                    ref.document_fiscal_year,
                    ref.estimated_budget,
                    ref.current_phase.value,
                    "yes" if ref.verified else "check",
                    ref.record_id,
                ]
            )
    book.table("Project References", ref_cols, ref_rows)

    # Sheet 4 — Extraction & Cost Metrics ------------------------------------------------------------
    m = result.metrics
    ws = book.wb.add_worksheet("Extraction & Cost Metrics")
    ws.set_column(0, 0, 44)
    ws.set_column(1, 1, 22)
    ws.set_column(2, 12, 15)
    ws.write(0, 0, f"Extraction & cost metrics — {result.organization_name}", book.f_title)
    ws.write(1, 0, f"Job {result.job_id} · generated {result.generated_at:%Y-%m-%d %H:%M UTC}", book.f_note)
    counts = [
        ("Documents discovered", m.documents_discovered),
        ("Documents probed (downloaded & profiled)", m.documents_probed),
        ("Documents selected", m.documents_selected),
        ("Pages processed", m.pages_processed),
        ("Chunks produced", m.chunks_produced),
        ("Chunks screened by Jev" if m.screening_provider.startswith("jev") else f"Chunks screened ({m.screening_provider})", m.chunks_screened),
        ("Chunks accepted for extraction", m.chunks_accepted),
        ("Chunks rejected (skipped)", m.chunks_rejected),
        ("  of accepted: kept by deterministic recall guard", m.chunks_recall_guard),
        ("Chunks extracted", m.chunks_extracted),
        ("Chunks split (truncation / dense tables)", m.chunks_split),
        ("Chunks escalated to stronger model", m.chunks_escalated),
        ("Projects extracted before merging (validated records)", m.projects_extracted_raw),
        ("Records rejected by validation", m.records_rejected_validation),
        ("Duplicates merged", m.duplicates_merged),
        ("Unique projects after merging", m.unique_projects),
        ("Uncertain possible duplicates (not merged)", m.possible_duplicates),
        ("Projects flagged for review", m.projects_needing_review),
    ]
    row = 3
    ws.write(row, 0, "Pipeline counts", book.f_h2)
    row += 1
    for label, value in counts:
        ws.write(row, 0, label)
        ws.write(row, 1, value, book.f_num)
        row += 1
    row += 1
    ws.write(row, 0, "Totals", book.f_h2)
    row += 1
    totals = [
        ("Total processing cost (USD)", m.total_cost_usd, book.f_money4),
        ("Cost per unique project (USD)", m.cost_per_unique_project_usd, book.f_money4),
        ("Execution time (seconds)", m.execution_time_s, book.f_num),
        ("Paid model/search requests", m.paid_requests, book.f_num),
        ("Requests served from cache (no cost)", m.cached_requests, book.f_num),
    ]
    for label, total_value, f in totals:
        ws.write(row, 0, label)
        ws.write(row, 1, total_value if total_value is not None else "", f)
        row += 1
    ws.write(row, 0, "Cost basis")
    ws.write(
        row,
        1,
        "includes ESTIMATED usage (provider did not report tokens / no price configured)"
        if m.cost_includes_estimates
        else "actual provider-reported usage x configured prices",
    )
    row += 1
    ws.write(row, 0, "Screening provider")
    ws.write(row, 1, m.screening_provider)
    row += 1
    ws.write(row, 0, "Extraction model")
    ws.write(row, 1, m.extraction_model)
    row += 2

    ws.write(row, 0, "Cost per processing stage", book.f_h2)
    row += 1
    stage_rows = [[stage, cost] for stage, cost in m.cost_by_stage.items()]
    book.table("CostByStage", [("stage", 44, "text"), ("cost_usd", 22, "money4")], stage_rows, start_row=row, ws=ws)
    row += len(stage_rows) + 3

    ws.write(row, 0, "Token usage by model", book.f_h2)
    row += 1
    usage_cols = [
        ("provider / model", 44, "text"),
        ("tasks", 22, "wrap"),
        ("requests", 15, "num"),
        ("failed", 15, "num"),
        ("retries", 15, "num"),
        ("input_tokens", 15, "num"),
        ("output_tokens", 15, "num"),
        ("cache_read_tokens", 15, "num"),
        ("cache_write_tokens", 15, "num"),
        ("searches", 15, "num"),
        ("input $/MTok", 15, "money4"),
        ("output $/MTok", 15, "money4"),
        ("cost_usd", 15, "money4"),
    ]
    usage_rows = [
        [
            f"{u['provider']} / {u['model']}" + (" (estimated)" if u["estimated"] else ""),
            u["tasks"],
            u["requests"],
            u["failed_requests"],
            u["retries"],
            u["input_tokens"],
            u["output_tokens"],
            u["cache_read_tokens"],
            u["cache_write_tokens"],
            u["search_requests"],
            u["input_price_per_mtok"],
            u["output_price_per_mtok"],
            u["cost_usd"],
        ]
        for u in m.token_usage_by_model
    ]
    book.table("TokenUsage", usage_cols, usage_rows, start_row=row, ws=ws)
    row += len(usage_rows) + 3

    ws.write(row, 0, "Stage timings (seconds)", book.f_h2)
    row += 1
    for stage, seconds in m.stage_timings_s.items():
        ws.write(row, 0, stage)
        ws.write(row, 1, seconds, book.f_num)
        row += 1
    row += 1
    if m.rejection_reasons:
        ws.write(row, 0, "Validation rejections by reason", book.f_h2)
        row += 1
        for reason, count in m.rejection_reasons.items():
            ws.write(row, 0, reason)
            ws.write(row, 1, count, book.f_num)
            row += 1
        row += 1
    ws.write(row, 0, "Errors and warnings", book.f_h2)
    row += 1
    issues = [("error", e) for e in m.errors] + [("warning", w) for w in m.warnings]
    if not issues:
        ws.write(row, 0, "none")
    for kind, text in issues:
        ws.write(row, 0, kind, book.f_bold if kind == "error" else None)
        ws.write(row, 1, text)
        row += 1

    # Supporting sheets ----------------------------------------------------------------------------
    disc_cols = [
        ("selected", 8, "text"),
        ("rank", 6, "int"),
        ("title", 44, "wrap"),
        ("url", 24, "link"),
        ("file_type", 8, "text"),
        ("publisher", 24, "wrap"),
        ("publication_date", 13, "text"),
        ("coverage_years", 16, "text"),
        ("document_type", 26, "wrap"),
        ("status", 16, "text"),
        ("official_source", 9, "text"),
        ("itemized_projects", 10, "text"),
        ("estimated_projects", 11, "num"),
        ("ranking_score", 10, "pct"),
        ("preliminary_score", 10, "pct"),
        ("selection_rationale_or_reason", 60, "wrap"),
        ("discovered_via", 40, "wrap"),
    ]
    disc_rows = []
    for d in result.discovered_documents:
        disc_rows.append(
            [
                "yes" if d.selected else "no",
                d.selection_rank,
                d.title,
                (d.final_url or d.url, "Open"),
                d.file_type,
                d.publisher,
                d.publication_date,
                d.coverage_years,
                d.doc_type.value,
                d.doc_status.value,
                "yes" if d.is_official_domain else "no",
                "" if d.contains_itemized_projects is None else ("yes" if d.contains_itemized_projects else "no"),
                d.probe.est_projects if d.probe and d.probe.ok else None,
                d.score,
                d.prelim_score,
                d.selection_rationale or d.rejection_reason,
                d.discovered_via[:3],
            ]
        )
    book.table("Discovered Documents", disc_cols, disc_rows)

    audit_cols = [
        ("decision", 16, "text"),
        ("project_uid", 16, "text"),
        ("other_project_uid", 16, "text"),
        ("rule", 34, "wrap"),
        ("score", 8, "num"),
        ("names", 60, "wrap"),
        ("record_ids", 40, "wrap"),
        ("signals", 50, "wrap"),
    ]
    audit_rows = [
        [
            a.decision,
            a.project_uid,
            a.other_project_uid,
            a.rule,
            a.score,
            a.names,
            a.record_ids,
            ", ".join(f"{k}={v}" for k, v in a.signals.items()),
        ]
        for a in result.merge_audit
    ]
    book.table("Dedup Audit", audit_cols, audit_rows)

    about = book.wb.add_worksheet("About")
    about.set_column(0, 0, 30)
    about.set_column(1, 1, 110)
    about.write(0, 0, "How to read this workbook", book.f_title)
    notes = [
        ("Projects", "One row per unique consolidated capital project. Filter/sort any column (domain, current_phase, "
         "estimated_budget, location, years, funding_status). 'source_url' opens the PDF at the cited page."),
        ("Sources", "The three automatically selected documents with ranking scores and selection rationale."),
        ("Project References", "Every project-to-source relationship (all documents/pages that mention the project)."),
        ("Extraction & Cost Metrics", "Pipeline counts, token usage by model, cost per stage, total cost, timings, errors."),
        ("Discovered Documents", "Inventory of every discovered candidate document with metadata and why it was or was not selected."),
        ("Dedup Audit", "Audit trail of every merge and every uncertain possible duplicate that was NOT merged."),
        ("estimated_budget", "Latest documented budget, normalized to US dollars from the printed figure and the table unit "
         "(e.g. '$000s'). budget_basis says what the figure represents; conflicting values from other documents are kept in budget_conflicts."),
        ("current_phase", "; ".join(p.value for p in Phase if p != Phase.UNKNOWN) + ". Kept separate from funding_status."),
        ("phase_basis", "explicit = stage stated in the source text (see phase_evidence); inferred = derived from document "
         "status/funding by deterministic rules (e.g. adopted program allocation -> Funded)."),
        ("needs_review", "Flags low confidence, partial grounding, budget conflicts or possible duplicates for a human check."),
        ("Values", "Names, budgets, years and identifiers are copied from the source text and verified to occur in it; "
         "missing information is left blank rather than guessed."),
    ]
    for i, (k, v) in enumerate(notes, start=2):
        about.write(i, 0, k, book.f_bold)
        about.write(i, 1, v, book.f_wrap)
    book.close()
    return path
