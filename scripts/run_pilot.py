#!/usr/bin/env python3
"""Five-organization pilot through the REST API (spec §13).

Starts one extraction job per organization via ``POST /api/v1/organizations/extract``, polls
``GET /api/v1/jobs/{id}`` until completion and saves every deliverable under ``pilot/<org-slug>/``:

    job.json        final job status (stage, progress counters, warnings/errors, accumulated cost)
    projects.json   consolidated projects            (GET /jobs/{id}/projects, all pages)
    records.json    raw structured extraction output (GET /jobs/{id}/records, all pages)
    sources.json    selected documents + discovered inventory (GET /jobs/{id}/sources)
    metrics.json    extraction/dedup/cost metrics incl. every measured request (GET /jobs/{id}/metrics)
    <slug>.xlsx     consolidated Excel workbook      (GET /jobs/{id}/export.xlsx)

Usage (server running: ``capex serve``):
    python scripts/run_pilot.py --base-url http://127.0.0.1:8000 --api-key dev-local-key
    python scripts/run_pilot.py --screening none --out pilot/_all_chunks   # screening trade-off run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

PILOT_ORGANIZATIONS = [
    "Arizona Department of Transportation",
    "City of Phoenix, Arizona",
    "City of Tucson, Arizona",
    "Maricopa County, Arizona",
    "City of Mesa, Arizona",
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def fetch_all(client: httpx.Client, url: str, page_size: int) -> list[dict]:
    items: list[dict] = []
    page = 1
    while True:
        resp = client.get(url, params={"page": page, "page_size": page_size})
        resp.raise_for_status()
        body = resp.json()
        items.extend(body["items"])
        if not body.get("next_page"):
            return items
        page = body["next_page"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="dev-local-key")
    parser.add_argument("--out", default="pilot")
    parser.add_argument("--screening", choices=["jev", "heuristic", "none"], default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--timeout-min", type=float, default=90)
    parser.add_argument("organizations", nargs="*", default=PILOT_ORGANIZATIONS)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=args.base_url, headers={"X-API-Key": args.api_key}, timeout=120)

    jobs: dict[str, str] = {}
    for org in args.organizations:
        body = {"organization_name": org, "max_documents": 3, "output_format": ["json", "xlsx"]}
        if args.screening:
            body["screening"] = args.screening
        if args.force_refresh:
            body["force_refresh"] = True
        resp = client.post("/api/v1/organizations/extract", json=body)
        resp.raise_for_status()
        payload = resp.json()
        jobs[org] = payload["job_id"]
        print(f"submitted {org!r}: {payload['job_id']} (created={payload['created']})", flush=True)

    deadline = time.time() + args.timeout_min * 60
    pending = dict(jobs)
    statuses: dict[str, dict] = {}
    while pending and time.time() < deadline:
        time.sleep(10)
        for org, job_id in list(pending.items()):
            status = client.get(f"/api/v1/jobs/{job_id}").json()
            statuses[org] = status
            if status["status"] in ("completed", "failed"):
                print(f"{status['status']:>9}: {org} (${status['accumulated_cost_usd']:.4f})", flush=True)
                pending.pop(org)
        if pending:
            summary = ", ".join(
                f"{org.split(',')[0]}={statuses.get(org, {}).get('stage', '?')}" for org in pending
            )
            print(f"  ... {summary}", flush=True)

    failed = 0
    for org, job_id in jobs.items():
        status = statuses.get(org) or client.get(f"/api/v1/jobs/{job_id}").json()
        folder = out_root / slugify(org)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "job.json").write_text(json.dumps(status, indent=2))
        if status["status"] != "completed":
            failed += 1
            print(f"!! {org}: {status['status']} {status.get('errors')}", file=sys.stderr)
            continue
        projects = fetch_all(client, f"/api/v1/jobs/{job_id}/projects", 500)
        records = fetch_all(client, f"/api/v1/jobs/{job_id}/records", 1000)
        (folder / "projects.json").write_text(json.dumps(projects, indent=2, ensure_ascii=False))
        (folder / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
        sources = client.get(f"/api/v1/jobs/{job_id}/sources").json()
        (folder / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False))
        metrics = client.get(f"/api/v1/jobs/{job_id}/metrics", params={"include_requests": True}).json()
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        xlsx = client.get(f"/api/v1/jobs/{job_id}/export.xlsx")
        xlsx.raise_for_status()
        (folder / f"{slugify(org)}.xlsx").write_bytes(xlsx.content)
        m = metrics["metrics"]
        print(
            f"saved {folder}: {m['unique_projects']} unique projects from {m['projects_extracted_raw']} records, "
            f"${m['total_cost_usd']:.4f}, {m['execution_time_s']:.0f}s",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
