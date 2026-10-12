# Public-Sector Capital Project Discovery, Extraction & Consolidation API

Give it the name of a U.S. public organization — *"Arizona Department of Transportation"* — and it finds the
organization's official capital-planning documents on the web, picks the three that together cover the most
projects, extracts every capital project with its description, budget, timing, lifecycle stage and funding status,
deduplicates projects across documents and years, and returns JSON plus an Excel workbook in which every project
links to the exact PDF page it came from — with a measured cost for every model and search request.

```
organization name → discovery → ranking → top-3 selection → PDF/HTML parsing → layout-aware chunking
                  → Jev screening → low-cost LLM extraction → validation → deduplication → Excel + API
```

<!-- PILOT-SUMMARY:START -->
> **Status:** implementation complete — API, pipeline, Excel/JSON outputs, cost ledger and evaluation tooling,
> covered by 366 offline tests (CI green). The five-organization pilot (ADOT, Phoenix, Tucson, Maricopa County,
> Mesa) is ready to run with `make pilot` and is pending provider API credentials (Anthropic for search/extraction,
> TypeSafe for Jev screening). Method and cost model: [docs/EVALUATION.md](docs/EVALUATION.md),
> [docs/COST_REPORT.md](docs/COST_REPORT.md).
<!-- PILOT-SUMMARY:END -->

## Highlights

* **Recall-first, cheap by construction.** Local PDF parsing, Jev (non-generative, $0.042/MTok, no output cost)
  screens every chunk with calibrated probabilities; only plausible chunks reach Claude Haiku 5.5
  ($0.10/$0.50 per MTok) — with deterministic recall guards so dense CIP tables are never skipped.
* **Verifiable.** Values are copied verbatim by the model and normalized by code; every name, ID, amount and year
  is checked against the source text, every cited page is verified (and corrected when wrong), and every
  project carries clickable `#page=N` links to all its sources.
* **No silent merges.** Conservative, auditable deduplication (identifiers, names, street numbers, phase/segment
  markers, location, budget, description) with constrained clustering; ambiguous pairs are flagged, not merged.
* **Production plumbing.** Async jobs with idempotency, retries, timeouts, cost caps and crash recovery; API-key
  auth; OpenAPI docs; content-hash caches so repeat runs make zero paid calls; per-request cost ledger.
* **Provider-agnostic & configurable.** Claude by default, any OpenAI-compatible endpoint (OpenAI, GLM, …),
  Jev/heuristic/no screening, Claude web search / Brave / Serper — models, prices and thresholds in
  [`config/capex.yaml`](config/capex.yaml). Nothing is hard-coded for the pilot organizations.

## Quickstart

```bash
make install                      # python3 -m venv .venv && pip install -e ".[dev]"
cp .env.example .env              # add ANTHROPIC_API_KEY (and TYPESAFE_API_KEY for Jev)
make serve                        # API on http://localhost:8000 — Swagger UI at /docs
```

```bash
KEY=dev-local-key   # from CAPEX_API_KEYS in .env
JOB=$(curl -s -X POST localhost:8000/api/v1/organizations/extract -H "X-API-Key: $KEY" \
        -H 'Content-Type: application/json' \
        -d '{"organization_name": "City of Mesa, Arizona", "max_documents": 3, "output_format": ["json","xlsx"]}' \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')
curl -s localhost:8000/api/v1/jobs/$JOB -H "X-API-Key: $KEY"                 # status, stage, progress, cost
curl -s "localhost:8000/api/v1/jobs/$JOB/projects?page_size=100" -H "X-API-Key: $KEY"
curl -s localhost:8000/api/v1/jobs/$JOB/export.xlsx -H "X-API-Key: $KEY" -o mesa.xlsx
```

Docker: `docker compose up --build` (same `.env`; data persisted in `./data`).

CLI (same pipeline, no server): `capex run "City of Tucson, Arizona" --out out/tucson`,
`capex discover "<org>"` (discovery + selection only), `capex parse <pdf-url>` (inspect parsing/chunking).

### Keys

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | LLM-assisted search (Claude `web_search`), query planning, document profiling, extraction |
| `TYPESAFE_API_KEY` | Jev screening (without it the zero-cost heuristic screener is used and reported) |
| `OPENAI_API_KEY` (+ `openai_base_url`) | optional alternative extraction provider (OpenAI, GLM, …) |
| `BRAVE_API_KEY` / `SERPER_API_KEY` | optional additional search providers |
| `CAPEX_API_KEYS` | API keys accepted by this service |

## Outputs

**Excel workbook** (one per organization): `Projects` (one row per unique project: all required fields, current
phase + evidence, funding status, anticipated years, latest budget, confidence, review flags, clickable source
link) · `Sources` (the three selected documents, URLs, publication dates, coverage years, ranking scores and
rationale) · `Project References` (one row per project–source relationship with page, printed page, excerpt and
original name) · `Extraction & Cost Metrics` (counts for every stage, token usage by model, cost per stage, total
cost, cost per project, execution time, errors/warnings) · `Discovered Documents` · `Dedup Audit` · `About`.

**API**: `POST /api/v1/organizations/extract`, `GET /api/v1/jobs/{id}`, `/projects` (paginated, filterable),
`/records` (raw extraction output), `/export.xlsx`, `/sources`, `/metrics` — see [docs/API.md](docs/API.md).

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design, stage by stage, with the reasoning behind each choice
* [docs/API.md](docs/API.md) — endpoints, idempotency, errors, examples
* [docs/COST_REPORT.md](docs/COST_REPORT.md) — measured cost per organization and per stage
* [docs/EVALUATION.md](docs/EVALUATION.md) — accuracy evaluation method and results, including deduplication and the
  Jev-vs-all-chunks trade-off
* [config/capex.yaml](config/capex.yaml) — provider keys (by env-var name), pricing, models, thresholds, limits

## Development

```bash
make check        # ruff + mypy + pytest (offline: HTTP and model calls are mocked)
make pilot        # five-organization pilot through the API (server running)
make report       # cost report + screening trade-off
make evaluate     # accuracy against the manually verified samples in evaluation/gold/
```

## Repository layout

```
src/capex/
  api/             FastAPI app, request/response schemas, API-key auth, structured errors
  discovery/       query planner, search providers, crawler, candidate inventory, probing, ranking/selection
  parsing/         layout-aware PDF parsing (pdfplumber/pdfium/tesseract) and HTML parsing
  chunking.py      layout-aware chunker (rows never split, table headers repeated, page markers)
  screening/       Jev screener, heuristic fallback, pass-through, routing policy
  extraction/      strict JSON schema, prompt, extractor (split/escalate/cache), grounding validation
  lifecycle.py     deterministic lifecycle & funding-status rules
  consolidation/   deduplication, constrained clustering, survivorship, audit trail
  export/          Excel workbook + JSON outputs
  costs.py         per-request usage & cost ledger
  jobs.py          async job manager (idempotency, retries, timeouts, crash recovery)
  pipeline.py      orchestration of all stages
  evaluation/      gold-sample scoring, screening trade-off, cost report
config/capex.yaml  models, prices, thresholds, limits
scripts/run_pilot.py  runs the five-organization pilot through the API
pilot/             pilot deliverables per organization (JSON + Excel)
evaluation/gold/   manually verified samples used for the accuracy evaluation
tests/             offline test suite
```
