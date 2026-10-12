"""Jev screening vs. processing all chunks directly (spec §12: measure the accuracy and cost trade-off).

Both runs use the same selected documents (selection is deterministic and cached). Records are
matched by (document, normalized project name); everything the all-chunks run finds that the
screened run does not is a screening miss, reported with the chunk's screening decision so the
cause is visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from capex.evaluation.loader import load_result
from capex.models import JobResult
from capex.normalize import name_key


def _keys(result: JobResult) -> dict[tuple[str, str], Any]:
    return {(r.doc_id, name_key(r.project_name)): r for r in result.records}


def _unique_keys(result: JobResult) -> set[str]:
    return {name_key(p.project_name) for p in result.projects}


def compare(screened_dir: Path, all_chunks_dir: Path) -> dict[str, Any]:
    screened, full = load_result(screened_dir), load_result(all_chunks_dir)
    s_keys, f_keys = _keys(screened), _keys(full)
    missed = [f_keys[k] for k in f_keys.keys() - s_keys.keys()]
    common = len(f_keys.keys() & s_keys.keys())
    sm, fm = screened.metrics, full.metrics
    s_unique, f_unique = _unique_keys(screened), _unique_keys(full)
    return {
        "organization": screened.organization_name,
        "screening_provider": sm.screening_provider,
        "chunks_total": fm.chunks_produced,
        "chunks_extracted_screened": sm.chunks_extracted,
        "chunks_extracted_all": fm.chunks_extracted,
        "chunks_skipped_by_screening": sm.chunks_rejected,
        "records_screened": len(s_keys),
        "records_all_chunks": len(f_keys),
        "record_recall_of_screening": round(common / len(f_keys), 4) if f_keys else None,
        "unique_projects_screened": len(s_unique),
        "unique_projects_all_chunks": len(f_unique),
        "unique_project_recall_of_screening": round(len(s_unique & f_unique) / len(f_unique), 4) if f_unique else None,
        "cost_screened_usd": sm.total_cost_usd,
        "cost_all_chunks_usd": fm.total_cost_usd,
        "screening_cost_usd": sm.cost_by_stage.get("screening", 0.0),
        "extraction_cost_screened_usd": sm.cost_by_stage.get("extraction", 0.0),
        "extraction_cost_all_chunks_usd": fm.cost_by_stage.get("extraction", 0.0),
        "missed_by_screening": [
            {"project_name": r.project_name, "document": r.source_document, "pages": r.source_pages, "chunk_id": r.chunk_id}
            for r in sorted(missed, key=lambda r: r.record_id)[:50]
        ],
    }
