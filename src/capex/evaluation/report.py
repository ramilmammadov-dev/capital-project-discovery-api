"""Aggregate pilot outputs into the cost report (per organization and per stage)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from capex.evaluation.loader import load_result


def pilot_summary(pilot_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for folder in sorted(p for p in Path(pilot_dir).iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not (folder / "metrics.json").is_file():
            continue
        result = load_result(folder)
        m = result.metrics
        rows.append(
            {
                "organization": result.organization_name,
                "folder": folder.name,
                "selected_documents": [
                    {"rank": d.selection_rank, "title": d.title, "url": d.final_url or d.url, "coverage": d.coverage_years, "score": d.score}
                    for d in sorted(result.selected_documents, key=lambda d: d.selection_rank or 99)
                ],
                "documents_discovered": m.documents_discovered,
                "pages_processed": m.pages_processed,
                "chunks_produced": m.chunks_produced,
                "chunks_screened": m.chunks_screened,
                "chunks_accepted": m.chunks_accepted,
                "chunks_rejected": m.chunks_rejected,
                "projects_extracted_raw": m.projects_extracted_raw,
                "duplicates_merged": m.duplicates_merged,
                "unique_projects": m.unique_projects,
                "possible_duplicates": m.possible_duplicates,
                "cost_by_stage": m.cost_by_stage,
                "total_cost_usd": m.total_cost_usd,
                "cost_per_unique_project_usd": m.cost_per_unique_project_usd,
                "cost_includes_estimates": m.cost_includes_estimates,
                "execution_time_s": m.execution_time_s,
                "token_usage_by_model": m.token_usage_by_model,
                "screening_provider": m.screening_provider,
                "warnings": m.warnings,
                "errors": m.errors,
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    stages = sorted({s for r in rows for s in r["cost_by_stage"]})
    header = ["Organization", "Docs found", "Pages", "Chunks", "Screened", "Extracted chunks", "Raw records", "Merged", "Unique", *[f"$ {s}" for s in stages], "Total $", "$/project", "Time (s)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(x)
                for x in [
                    r["organization"],
                    r["documents_discovered"],
                    r["pages_processed"],
                    r["chunks_produced"],
                    r["chunks_screened"],
                    r["chunks_accepted"],
                    r["projects_extracted_raw"],
                    r["duplicates_merged"],
                    r["unique_projects"],
                    *[f"{r['cost_by_stage'].get(s, 0.0):.4f}" for s in stages],
                    f"{r['total_cost_usd']:.4f}",
                    f"{(r['cost_per_unique_project_usd'] or 0):.5f}",
                    f"{r['execution_time_s']:.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
