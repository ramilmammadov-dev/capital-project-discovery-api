"""Select representative pages for manual verification (spec §14) and write review sheets.

For every selected document of a pilot result, pages are sampled across content types — dense project
tables, project descriptions/sheets, budget line items — plus a page from a chunk that screening skipped
(to catch screening misses). For each sampled page a review file is written with the page text exactly as
parsed, followed by the records the system extracted for that page. The reviewer builds the gold inventory
from the page text (not from the extractions) in ``evaluation/gold/<org>.yaml``.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from capex.evaluation.loader import load_result
from capex.models import ParsedDocument
from capex.parsing.pdf import PARSER_VERSION
from capex.storage.cache import FileCache


def write_samples(pilot_folder: Path, cache_dir: Path, out_dir: Path, per_doc: int = 4, seed: int = 7) -> Path:
    result = load_result(pilot_folder)
    cache = FileCache(cache_dir)
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_pages: list[dict[str, Any]] = []
    for doc in sorted(result.selected_documents, key=lambda d: d.selection_rank or 99):
        records = [r for r in result.records if r.doc_id == doc.doc_id]
        pages_with_records = sorted({p for r in records for p in r.source_pages})
        parsed_json = cache.get_json("parsed", f"{doc.probe.content_hash}-{PARSER_VERSION}") if doc.probe else None
        if not parsed_json:
            continue
        parsed = ParsedDocument.model_validate(parsed_json)
        page_text = {p.pdf_page: p.text for p in parsed.pages}
        counts: dict[int, int] = defaultdict(int)
        for r in records:
            for p in r.source_pages:
                counts[p] += 1
        dense = [p for p, c in sorted(counts.items(), key=lambda kv: -kv[1])[:10]]
        sparse = [p for p, c in counts.items() if c <= 2]
        chosen: list[int] = []
        for pool in (dense, sparse, pages_with_records):
            candidates = [p for p in pool if p not in chosen]
            if candidates:
                chosen.append(rng.choice(candidates[: max(3, len(candidates) // 2)] if pool is dense else candidates))
            if len(chosen) >= per_doc - 1:
                break
        # one page that produced no records (screened out or genuinely empty) to test for misses
        empty = [p for p in page_text if p not in counts and len(page_text[p]) > 400]
        if empty:
            chosen.append(rng.choice(empty))
        for page in sorted(set(chosen)):
            on_page = [r for r in records if page in r.source_pages]
            lines = [
                f"DOCUMENT: {doc.title}",
                f"URL: {doc.final_url or doc.url}#page={page}",
                f"PDF PAGE: {page}",
                "=" * 100,
                page_text.get(page, "(page text unavailable)"),
                "=" * 100,
                f"SYSTEM RECORDS CITING THIS PAGE ({len(on_page)}):",
            ]
            for r in on_page:
                lines.append(
                    json.dumps(
                        {
                            "name": r.project_name,
                            "id": r.project_id,
                            "budget": r.estimated_budget,
                            "budget_text": r.budget_text,
                            "phase": r.current_phase.value,
                            "phase_evidence": r.phase_evidence,
                            "pages": r.source_pages,
                        },
                        ensure_ascii=False,
                    )
                )
            name = f"{doc.selection_rank}-p{page:04d}.txt"
            (out_dir / name).write_text("\n".join(lines), encoding="utf-8")
            gold_pages.append({"document": doc.title[:80], "pdf_page": page, "complete": True, "projects": []})
    template = {
        "organization": result.organization_name,
        "result": str(pilot_folder),
        "match_threshold": 80,
        "pages": gold_pages,
        "dedup": [],
    }
    path = out_dir / "gold-template.yaml"
    path.write_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
