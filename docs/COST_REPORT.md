# Cost report

> **Status:** every model and search request is measured by the per-job cost ledger (provider, model, task, stage,
> input/output/cache tokens, price per token, request fee, cost, duration, retries, failures, estimated-flag).
> The measured per-organization numbers are produced by the five-organization pilot, which is pending provider API
> credentials. Until then this page documents the cost model and **pre-pilot estimates** derived from documents
> that were actually downloaded, parsed and chunked by the system. They are estimates, not measurements.

## Prices used (configurable in `config/capex.yaml`)

| Item | Price |
|---|---|
| Claude Haiku 5.5 (extraction, planning, profiling, LLM-assisted search) | $0.10 / MTok input · $0.50 / MTok output · $0.01 / MTok cache read |
| Claude Sonnet 5.5 (escalation only) | $2.00 / MTok input · $10.00 / MTok output |
| Jev (TypeSafe) screening | $0.042 / MTok input · output free |
| Claude web search | $10 / 1,000 searches |
| PDF parsing, chunking, OCR (pdfplumber / pdfium / tesseract) | local, $0 |

## Where the money goes

| Stage | Driver | Cost control |
|---|---|---|
| Discovery | ~10 searches + result tokens | planner writes targeted queries; results cached 7 days |
| Probing & selection | local sampling of ~14 candidates + 1 batched Haiku call | no LLM per page; one call for all covers |
| Screening | Jev on every chunk | non-generative, no output tokens; cached by chunk hash |
| Extraction | Haiku output tokens (one record ≈ 250–300 tokens) | only accepted chunks; verbatim fields, short evidence; cached system prompt |
| Escalation | Sonnet, only when justified | capped at 8 chunks per job |

## Pre-pilot estimates from measured documents

| Document (measured by `capex parse`) | Pages | Chunks | Content tokens | Est. Jev screening | Est. extraction (Haiku) |
|---|---|---|---|---|---|
| ADOT 2027–2031 Tentative Five-Year Program | 202 | 74 | ~91k | ≈ $0.005 | ≈ $0.04–0.06 |
| City of Phoenix 2026–31 Capital Improvement Program | 493 | 439 | ~214k | ≈ $0.03 | ≈ $0.15–0.25 |

Assumptions: Jev input ≈ chunk + context + questions (~1.5k tokens per chunk); ~85% of chunks accepted;
~280 output tokens per extracted record; the static extraction instructions are read from the prompt cache.

Expected total per organization (three documents, search included): **≈ $0.25–0.75**, below the $1 target for
ordinary digitally readable CIPs. The largest variable is the number of project rows (output tokens), not pages.
Repeat runs of unchanged documents cost $0 (all stages are cached by content hash).

## Measured results

_Pending the pilot run._ `make report` writes `pilot/summary.json` and prints the per-organization table:
documents discovered, pages, chunks, screened/accepted chunks, raw records, merges, unique projects, cost per stage,
total cost, cost per unique project and execution time; `make report` with the all-chunks run also reports the
Jev-vs-all-chunks cost and recall trade-off.
