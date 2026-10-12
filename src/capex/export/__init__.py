"""Persist job outputs: full JSON result (API source of truth), Excel workbook, and flat JSON views."""

from __future__ import annotations

import json
from pathlib import Path

from capex.export.excel import write_workbook
from capex.models import JobResult


def _dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_outputs(result: JobResult, out_dir: Path, formats: list[str] | None = None) -> dict[str, Path]:
    formats = formats or ["json", "xlsx"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    full = out_dir / "result.json"
    full.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    paths["result"] = full
    if "json" in formats:
        _dump(out_dir / "projects.json", [p.model_dump(mode="json") for p in result.projects])
        _dump(
            out_dir / "sources.json",
            {
                "selected": [d.model_dump(mode="json", exclude={"probe": {"first_page_text", "id_tokens", "name_tokens"}}) for d in result.selected_documents],
                "discovered": [
                    d.model_dump(mode="json", exclude={"probe": {"first_page_text", "id_tokens", "name_tokens"}})
                    for d in result.discovered_documents
                ],
            },
        )
        _dump(out_dir / "metrics.json", result.metrics.model_dump(mode="json"))
        paths["projects"] = out_dir / "projects.json"
        paths["sources"] = out_dir / "sources.json"
        paths["metrics"] = out_dir / "metrics.json"
    if "xlsx" in formats:
        paths["xlsx"] = write_workbook(result, out_dir / "export.xlsx")
    return paths


__all__ = ["save_outputs", "write_workbook"]
