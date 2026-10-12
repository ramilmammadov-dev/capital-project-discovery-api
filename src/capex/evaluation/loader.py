"""Load a job result either from ``result.json`` or from a pilot folder written by scripts/run_pilot.py."""

from __future__ import annotations

import json
from pathlib import Path

from capex.models import JobResult


def load_result(path: Path) -> JobResult:
    path = Path(path)
    if path.is_file():
        return JobResult.model_validate_json(path.read_text("utf-8"))
    if (path / "result.json").is_file():
        return JobResult.model_validate_json((path / "result.json").read_text("utf-8"))
    metrics = json.loads((path / "metrics.json").read_text("utf-8"))
    sources = json.loads((path / "sources.json").read_text("utf-8"))
    return JobResult.model_validate(
        {
            "job_id": metrics["job_id"],
            "organization_name": metrics["organization_name"],
            "selected_documents": sources.get("selected", []),
            "discovered_documents": sources.get("discovered", []),
            "projects": json.loads((path / "projects.json").read_text("utf-8")),
            "records": json.loads((path / "records.json").read_text("utf-8")),
            "usage": metrics.get("usage_records") or [],
            "metrics": metrics["metrics"],
        }
    )
