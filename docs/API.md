# REST API

Interactive OpenAPI/Swagger documentation is served at **`/docs`** (ReDoc at `/redoc`, schema at `/openapi.json`).

* Base path: `/api/v1`
* Authentication: every endpoint except `/health` requires an API key in the `X-API-Key` header.
  Accepted keys are configured with `CAPEX_API_KEYS` (comma-separated).
* Errors are always structured:

```json
{ "error": { "code": "job_not_completed", "message": "job is running (stage: extracting)", "details": {"status": "running", "stage": "extracting"} } }
```

| Code | HTTP | Meaning |
|---|---|---|
| `unauthorized` | 401 | missing/invalid `X-API-Key` |
| `invalid_request` | 422 | body/query validation failed (`details` lists the fields) |
| `job_not_found` | 404 | unknown job id |
| `job_not_completed` | 409 | results requested before the job completed |
| `idempotency_conflict` | 409 | `Idempotency-Key` reused for a different request |
| `result_unavailable` | 410 | job output no longer on disk |

Job-level error codes (in `GET /jobs/{id}` → `errors`): `timeout`, `cost_cap_exceeded`, `provider_not_configured`,
`upstream_unavailable` (after retries), `internal_error`, and `stage_error` for non-fatal per-document problems.

## Start extraction

`POST /api/v1/organizations/extract`

```bash
curl -s -X POST http://localhost:8000/api/v1/organizations/extract \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -H "Idempotency-Key: adot-2026-10-12" \
  -d '{"organization_name": "Arizona Department of Transportation", "max_documents": 3, "output_format": ["json", "xlsx"]}'
```

```json
{
  "job_id": "job_4f0c2a9b1e7d43c6a2b1",
  "status": "queued",
  "created": true,
  "links": {
    "self": "/api/v1/jobs/job_4f0c2a9b1e7d43c6a2b1",
    "projects": "/api/v1/jobs/job_4f0c2a9b1e7d43c6a2b1/projects",
    "sources": "/api/v1/jobs/job_4f0c2a9b1e7d43c6a2b1/sources",
    "metrics": "/api/v1/jobs/job_4f0c2a9b1e7d43c6a2b1/metrics",
    "export_xlsx": "/api/v1/jobs/job_4f0c2a9b1e7d43c6a2b1/export.xlsx"
  }
}
```

| Field | Default | Notes |
|---|---|---|
| `organization_name` | — | any U.S. public organization |
| `max_documents` | 3 | documents selected for detailed extraction (1–5) |
| `output_format` | `["json","xlsx"]` | |
| `screening` | config (`jev`) | `jev`, `heuristic` or `none` (process all chunks) |
| `force_refresh` | false | start a new job even if an identical recent job exists |

**Idempotency.** `202` = new job. `200` with `"created": false` = an existing job was returned:
* same `Idempotency-Key` + same request → that job (different request → `409 idempotency_conflict`);
* without a key, an identical request (organization + parameters) returns the queued/running job or one completed
  in the last 72 h (`limits.job_reuse_hours`).
Even with `force_refresh`, unchanged documents are not re-paid: downloads, parsed pages, chunks, screening and
extraction results are cached by content hash.

## Job status

`GET /api/v1/jobs/{job_id}`

```json
{
  "job_id": "job_4f0c…",
  "organization_name": "Arizona Department of Transportation",
  "status": "running",
  "stage": "extracting",
  "progress": {"documents_discovered": 41, "documents_probed": 14, "documents_selected": 3, "pages_processed": 389,
               "chunks_produced": 151, "chunks_screened": 151, "chunks_accepted": 118, "chunks_extracted": 60,
               "projects_extracted_raw": 402},
  "warnings": [],
  "errors": [],
  "accumulated_cost_usd": 0.1432,
  "attempts": 1,
  "created_at": "2026-10-12T19:02:11Z",
  "started_at": "2026-10-12T19:02:11Z",
  "finished_at": null,
  "links": { "...": "..." }
}
```

`status`: `queued` → `running` → `completed` | `failed`. `stage`: `discovering`, `ranking`, `downloading`,
`parsing`, `chunking`, `screening`, `extracting`, `validating`, `deduplicating`, `exporting`, `completed`.
`accumulated_cost_usd` is updated live from the per-request usage ledger.

`GET /api/v1/jobs?limit=20` lists recent jobs.

## Extracted projects

`GET /api/v1/jobs/{job_id}/projects?page=1&page_size=50`

Filters: `domain`, `phase` (e.g. `Funded`, `Design`, `Construction / In Progress`), `funding_status`,
`min_budget`, `q` (text search in name/description/location).

```json
{
  "job_id": "job_4f0c…", "organization_name": "…", "total": 512, "page": 1, "page_size": 50, "next_page": 2,
  "items": [
    {
      "project_uid": "PRJ-3F9A1C07B2",
      "organization_name": "Arizona Department of Transportation",
      "project_id": "104397",
      "project_name": "Sanders Port of Entry Scale Replacement",
      "project_description": "Replace scale and construct inspection pit",
      "domain": "transportation",
      "project_type": "port of entry scale replacement",
      "location": "Point location on I 40 at Sanders Port of Entry, Apache County",
      "estimated_budget": 10000000.0,
      "budget_currency": "USD",
      "budget_type": "programmed",
      "budget_basis": "Total Programmed",
      "document_fiscal_year": 2027,
      "anticipated_start_year": null,
      "anticipated_construction_year": 2029,
      "anticipated_completion_year": null,
      "current_phase": "Proposed",
      "phase_evidence": "CON … FY2029 $9,000,000",
      "phase_basis": "inferred",
      "funding_status": "Requested",
      "source_document": "2027-2031 Tentative Five-Year Transportation Facilities Construction Program",
      "source_url": "https://…/Tentative-Program.pdf",
      "source_pages": [60],
      "source_section": "Highways and Regional Programs",
      "source_excerpt": "Project Name  Sanders Port of Entry Scale Replacement",
      "references": [ { "document_title": "…", "page_url": "https://…pdf#page=60", "pdf_pages": [60], "printed_pages": ["60"], "excerpt": "…", "verified": true } ],
      "extraction_confidence": 0.92,
      "needs_review": false,
      "review_reasons": [],
      "merged_record_ids": ["doc_52da10a25a:0024#001"],
      "possible_duplicates": []
    }
  ]
}
```

(Values above are illustrative of the shape.)

`GET /api/v1/jobs/{job_id}/records?page=1&page_size=200` returns the raw structured extraction output — one
record per project per chunk, before deduplication, with validation flags (`excerpt_verified`, `name_verified`,
`flags`) and the chunk it came from.

## Excel

`GET /api/v1/jobs/{job_id}/export.xlsx` → `capital-projects-<organization>.xlsx`

## Sources

`GET /api/v1/jobs/{job_id}/sources` → `selected` (the three documents with `selection_rank`, `score`,
`score_breakdown`, `selection_rationale`, publication date, coverage years, type, status) and `discovered`
(the full inventory with `rejection_reason` for every document that was not selected).

## Metrics and costs

`GET /api/v1/jobs/{job_id}/metrics?include_requests=true`

`metrics` contains documents discovered/probed/selected, pages processed, chunks produced/screened/accepted/
rejected (and kept by the recall guard), chunks extracted/split/escalated, raw records, validation rejections by
reason, duplicates merged, unique projects, possible duplicates, token usage by model (requests, failures, retries,
input/output/cache tokens, searches, price per MTok, cost), cost by stage, total cost, cost per unique project,
whether any cost is estimated, execution time, stage timings, errors and warnings.
With `include_requests=true`, `usage_records` lists every measured request (provider, model, task, stage, tokens,
prices, cost, duration, retries, success/error, `estimated`).
