# Architecture

```
POST /api/v1/organizations/extract  ──►  Job queue (DB-backed, idempotent)  ──►  Pipeline
                                                                                   │
 ┌─────────────────────────────────────────────────────────────────────────────────┘
 │
 │  1. Plan          org name ─► Claude Haiku (structured) ─► official domains, org-specific document titles, queries
 │  2. Discover      web search (Claude web_search / Brave / Serper) + depth-2 crawl of official capital/budget pages
 │  3. Probe         download (cached) ─► pdfium fast text on ~14 sampled pages ─► itemization, volume, years, status
 │  4. Rank/select   explainable score ─► greedy top-3 maximizing combined UNIQUE coverage (overlap-penalized)
 │  5. Parse         pdfplumber layout lines + headings + ruled tables + exact/printed page numbers; local OCR fallback
 │  6. Chunk         layout-aware: never split a table row; repeat table header; page markers; ≤28 rows per chunk
 │  7. Screen (Jev)  3 Choice questions per chunk ─► calibrated probabilities ─► extract / extract+context / skip
 │  8. Extract       Claude Haiku 5.5, strict JSON schema, verbatim values ─► split on truncation ─► escalate (capped)
 │  9. Validate      grounding (name/ID/amount/year must occur in source text), page verification/correction,
 │                   aggregate-row rejection, deterministic money/year normalization, lifecycle & funding rules
 │ 10. Deduplicate   ID + name + numbers/phase markers + location/budget/description; constrained clustering;
 │                   stable project UIDs from a registry; audit trail; possible duplicates flagged, never merged
 │ 11. Export        JSON (API) + Excel workbook (Projects / Sources / Project References / Metrics / …)
 └─► every model/search request ─► CostLedger ─► usage_records table ─► per-stage / per-model cost, live in job status
```

Code map: `src/capex/{discovery,parsing,chunking.py,screening,extraction,lifecycle.py,consolidation,export,api,jobs.py,pipeline.py}`.

## Design principles

1. **Recall first, then cost.** Every cost-saving step has a recall safeguard: screening skips a chunk only when
   Jev is confident *and* a deterministic recall guard (table rows / IDs / amounts) does not fire; dense tables are
   capped at 28 rows per chunk and re-split when the model returns far fewer records than the row estimate;
   uncertain-but-plausible chunks are always extracted.
2. **Documented facts only.** The extraction model copies values verbatim (`"$5,300"`, unit `thousands`,
   `"FY 2029"`); deterministic code normalizes them. Validation proves every name, identifier, amount and year
   occurs in the source text — otherwise the value is dropped (or the record rejected) and flagged. Lifecycle
   stages carry their evidence text and an `explicit` / `inferred` basis.
3. **Verifiable references.** Exact PDF page numbers are preserved through parsing → chunk page markers →
   extraction → validation (the cited page is checked against the excerpt and corrected when the quote sits on
   another page). Printed page labels are kept when they differ. Links open the PDF at the page (`#page=N`).
4. **No silent merges.** Deduplication is deterministic and conservative; anything ambiguous is reported as a
   possible duplicate and left for review. Every merge is in the audit trail.
5. **Deterministic, cached, idempotent.** Content-hash caches for downloads, parsed pages, chunks, screening and
   extraction; identical requests return the existing job; a repeat run of an unchanged organization makes zero
   paid model calls.
6. **Provider-agnostic, configuration over code.** Models, prices, thresholds and providers live in
   `config/capex.yaml`. Nothing is specific to the five pilot organizations.

## Stage details

### 1–2. Discovery (`discovery/planner.py`, `search.py`, `candidates.py`)
* **Planner.** One Haiku call (strict JSON schema, cached per organization and fiscal year) returns canonical
  name, organization type, state, official domains, aliases, the organization's *own* document titles
  (e.g. ADOT's "Five-Year Transportation Facilities Construction Program") and 8–10 queries including
  `site:` queries. Template queries are always appended as a safety net (and used alone without an LLM).
* **Search.** LLM-assisted search through Claude's server-side `web_search` tool (`$10 / 1,000 searches`):
  queries are executed in groups of four, and result URLs/titles are harvested from the `web_search_tool_result`
  blocks — the model's prose is ignored so ranking stays deterministic. Brave and Serper are drop-in providers;
  DuckDuckGo is a keyless fallback (it often refuses automated queries, which is reported).
* **Crawl.** Breadth-first, depth ≤ 2, ≤ 40 pages, only official domains, only links whose URL/anchor mentions
  capital/CIP/budget/program/bond/facilities…; document links (PDF, CivicPlus `DocumentCenter`, etc.) are collected
  with their anchor text. This finds every version of a CIP (adopted, proposed, prior years) even when search
  surfaces only one. robots.txt is honoured and requests are paced per host.
* **Inventory.** Every candidate gets metadata from title/URL/snippet/cover: document type (the 10 categories of
  spec §3), status (adopted vs proposed/tentative), coverage fiscal years, publisher, publication date.

### 3–4. Probing, ranking and selection (`discovery/probe.py`, `ranking.py`)
The top 14 candidates (max three versions per document series) are downloaded and profiled *without* any LLM:
~14 sampled pages via pdfium's text layer (milliseconds per page) measure the share of pages with itemized project
rows, the rows per page (→ estimated project count), budgets/years/descriptions, readable text share, and
identifier/name tokens for overlap estimation. Aggregate-only reports (department totals, fund summaries) score
0 on itemization — spec §3's "distinguish itemized projects from high-level reports". One **batched** Haiku call
then reads the covers of all probed documents for publisher, publication date, type, status and coverage years;
model-reported years are accepted only if printed on the cover.

Score = weighted features (itemization 0.30, volume 0.15, authority 0.12, recency 0.15, detail 0.12, doc type
0.08, relevance 0.05, readability 0.03 — all configurable). **Selection maximizes combined unique coverage**:
greedy on `score × (1 − 0.9 × overlap)`, where overlap is estimated from document series (same title without
years/status words), shared project identifiers and shared row keys. Each selected document gets a written
rationale; each rejected one a reason (e.g. "overlaps ~95% with selected #1").

*Source trust.* Official domain = 1.0 authority; a verified copy of the organization's own document on another
host = 0.8 (used when the official host blocks automated downloads); other government domains 0.6; others 0.2.
When an official host answers with a bot-protection challenge the system does **not** try to bypass it: it uses
the latest Internet Archive capture of the same URL if one exists, and records that provenance.

### 5–6. Parsing and layout-aware chunking (`parsing/pdf.py`, `parsing/html.py`, `chunking.py`)
* pdfplumber words → lines rebuilt at character columns proportional to x-position, so table columns stay
  aligned across rows (the extractor needs this to tie amounts to years); font size/weight → headings; ruled
  tables → cell grids; header/footer → printed page labels. Pages are parsed in parallel processes
  (202 pages ≈ 12 s). pdfplumber/pdfminer are MIT-licensed — PyMuPDF was rejected because of AGPL.
* Image-only pages are OCR'd locally with tesseract (bounded per document); no paid OCR.
* The chunker builds **units** that may never be split — heading, paragraph, or a table row with its wrapped
  continuation lines (even across a page break) — then packs units into chunks of ~1,800 tokens, breaking at
  page-top headings or page boundaries (keeps one-page project sheets whole), never inside a row, at most 28 project
  rows per chunk. Each chunk carries its section path, the table's column header when the table started earlier,
  and the last lines of the previous chunk as *context only*. Running headers/footers and repeated captions are
  removed. Parsed pages and chunks are cached by content hash for reuse and debugging (`capex parse` dumps them).

### 7. Jev screening (`screening/jev.py`, `screening/base.py`)
Jev (TypeSafe) is a non-generative decision model: for each chunk one request asks three `Choice` questions —
**A. presence** (explicit / possible / context only / none), **B. content type** (the eight types of spec §6),
**C. need** (extract / extract with adjacent context / skip) — and returns a calibrated probability for every label
at $0.042 per million input tokens with no output cost. Routing uses probabilities, not labels:

```
p_project = P(explicit) + P(possible)
extract                if p_project ≥ 0.50  or P(skip) < 0.5
extract (uncertain)    if p_project ≥ 0.15          ← plausible projects are never discarded
extract (recall guard) if table rows / IDs / amounts are present
skip                   otherwise
with adjacent context  if P(extract_with_context) ≥ 0.35
```

Requests run concurrently (semaphore + paced starts, well under Jev's 80 req/s), are retried with backoff
(counted in the ledger), and cached; routing is recomputed from cached probabilities when thresholds change.
If Jev is not configured or fails, a zero-cost heuristic screener with the same output shape takes over per chunk
— and every report says so. `screening: none` processes all chunks (used to measure the trade-off).

### 8–9. Extraction and validation (`extraction/`, `lifecycle.py`)
* **Model.** Claude Haiku 5.5 by default ($0.10 / $0.50 per MTok), strict structured output
  (`output_config.format` JSON schema; every field required, nothing nullable — empty strings mean "not stated",
  which keeps the schema inside strict-mode limits). The static instructions are a cached system prompt; the
  per-chunk message carries only document metadata, context and the chunk.
* **Exhaustiveness.** "Return every project row" is in the prompt; dense tables are kept small; a chunk whose
  output is truncated is split at a row boundary and retried; a table chunk that yields < 60% of its estimated
  rows is split once and the better result kept.
* **Escalation.** To Claude Sonnet 5.5 only when justified — still-invalid output after splitting, or ≥ 3 records
  with names that do not occur in the source (hallucination signal) — capped at 8 chunks per job.
* **Validation.** Name grounding (fuzzy, whitespace-insensitive) or rejection; identifier/amount/year grounding or
  drop + flag; page verification/correction; totals/subtotals rejected; money normalized with the table unit
  (`($000)` → ×1,000); fiscal labels (`2026-27` → FY2027); confidence penalties; `needs_review` flags.
* **Lifecycle** (spec §8). The model proposes one of the eight stages with evidence; deterministic rules then
  require explicit evidence words for Design / RFP-Procurement / Construction / Completed / Deferred, downgrade
  stage claims that only cite a *future* year ("Construction FY2029" = programmed, not underway), and default
  from document status (adopted program allocation → Funded; tentative/proposed document → Proposed).
  `funding_status` is classified separately (Unfunded / Requested / Partially funded / Funded / Unknown).

### 10. Deduplication and consolidation (`consolidation/dedup.py`)
* Candidate pairs: same normalized identifier, or name similarity ≥ 72 (token-set) found with rapidfuzz.
* Hard **cannot-link** rules: different identifiers of the same ID scheme; different numbers inside the name
  (7th Ave ≠ 19th Ave, MP 120–135 ≠ MP 135–150); different phase/segment markers; two rows of the same table with
  different budgets. "Bridge X" vs "Bridge X Phase 2" (component vs whole) is never merged automatically.
* **Merge** when identifiers match with compatible names, names are near-identical (≥ 93 after abbreviation
  normalization), or similar (≥ 85) and corroborated by location, budget (±25%) or description.
  Everything else ≥ 72 is a **possible duplicate** — flagged, audited, not merged.
* **Constrained union-find**: clusters join only if no cross pair is a cannot-link, so A~B~C chains cannot merge
  distinct A and C.
* **Survivorship**: newest document first for name, phase, schedule and latest budget; the longest description;
  conflicting budgets and earlier stages are preserved (`budget_conflicts`, `phase_history`); all references kept.
* **Stable IDs**: `project_uid` comes from a registry keyed by organization + alias keys (identifier, or name +
  location), so re-runs and later document versions map to the same project instead of creating duplicates.

### 11. Outputs (`export/`, `api/`)
Excel: **Projects**, **Sources**, **Project References**, **Extraction & Cost Metrics** (spec §10), plus
Discovered Documents, Dedup Audit and About. All data sheets are Excel tables (filter/sort on domain, phase,
budget, location, years, funding status). JSON: the same data via the API (paginated).

## Cost model

| Stage | What runs | Typical cost / organization |
|---|---|---|
| Planning | 1 Haiku call (~1k tokens) | < $0.001 |
| Search | ≈10 Claude web searches + result tokens | ≈ $0.10–0.12 |
| Probing | local pdfium on ~14 pages × 14 docs | $0 |
| Profiling | 1 batched Haiku call | < $0.002 |
| Parsing/chunking | local | $0 |
| Screening | Jev, all chunks (~1–1.5k tokens each) | ≈ $0.005–0.02 |
| Extraction | Haiku on accepted chunks; cached system prompt | ≈ $0.05–0.40 |
| Escalation | Sonnet, ≤ 8 chunks, only when justified | $0–0.30 |

Search dominates the fixed cost, extraction the variable cost — and both are cached, so a repeat run is free.
Actual pilot numbers are in [COST_REPORT.md](COST_REPORT.md).

## Failure handling
* HTTP: per-host pacing, retries with backoff, size limits, robots.txt, conditional GET; bot-challenge detection
  with an archive fallback (never bypass).
* Models: retries with exponential backoff and `retry-after`; every attempt and failure recorded; unparseable or
  truncated output → split → escalate; provider missing → explicit `provider_not_configured` job error.
* Jev: per-chunk heuristic fallback on error, with the error recorded in the screening result.
* Jobs: per-job timeout, cost cap (remaining chunks are skipped with a warning rather than failing the job),
  job-level retry for transient upstream errors, crash recovery (running jobs are re-queued on start-up).

## Extending
* **New LLM provider**: implement `LLMClient.json_completion` (or point `openai_compatible` at any OpenAI-style
  endpoint such as GLM), add a price entry.
* **New search provider**: subclass `SearchProvider`.
* **New document format** (e.g. XLSX budgets): add a parser that emits `ParsedDocument`; chunking, screening,
  extraction and validation are format-agnostic.
* **Scenario-specific logic** (e.g. a customer interested in HVAC/boiler/central-plant work) belongs in
  configuration and a post-extraction scoring layer over `ConsolidatedProject`, not in the core pipeline.
* **Scale-out**: `CAPEX_DATABASE_URL` → Postgres; run API and workers as separate processes on the same DB;
  move `data/cache` to object storage; the Anthropic Batch API (−50%) fits the screening/extraction fan-out for
  non-interactive monitoring runs.
