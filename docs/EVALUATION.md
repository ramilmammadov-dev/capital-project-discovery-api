# Accuracy evaluation

> **Status:** the evaluation tooling is implemented and tested; the five-organization pilot run that produces
> the numbers is pending provider API credentials (Anthropic for search/extraction, TypeSafe for Jev).
> This document describes exactly how the results will be measured; the results table is filled from
> `capex evaluate` once the pilot has run.

## Targets (spec §14)

| Metric | Target |
|---|---|
| Recall against a manually verified project inventory (representative sample) | ≥ 95% |
| Precision of records accepted as genuine projects (same sample) | ≥ 95% |
| Accuracy of document and PDF page references | ≥ 98% |
| Accuracy of normalized lifecycle stages where the source clearly identifies a stage | ≥ 95% |
| Invented project names, budgets or dates | none |
| Silently merged distinct projects | none |
| Deduplication precision and recall (false merges weighted as most serious) | reported separately |
| Repeat runs | no duplicate records, no unnecessary paid processing |

## Method

1. **Sampling** — `capex sample pilot/<org>` selects pages from each of the three selected documents across
   content types: dense project tables, sparse pages / project sheets, budget line items, and at least one page
   that produced **no** records (to detect screening or extraction misses). For every sampled page it writes a
   review sheet with the page text exactly as parsed, followed by the system's records for that page.
2. **Gold inventory** — the reviewer lists every capital project printed on each sampled page from the page text
   (and the PDF itself), independently of the system output: name, identifier, budget as printed × table unit,
   and the lifecycle stage *only* where the page states it clearly. Pages are marked `complete` when every project
   on them is listed. Files: `evaluation/gold/<org>.yaml`.
3. **Scoring** — `capex evaluate`:
   * **recall**: gold projects matched (identifier, or fuzzy name ≥ 80) by any record of that document;
   * **precision**: records citing a complete sample page that match a gold project on it;
   * **page-reference accuracy**: matched records whose cited PDF page(s) include the gold page;
   * **lifecycle accuracy**: matched records whose normalized stage equals the gold stage (where given);
   * **budget accuracy**: matched records whose normalized budget equals the gold value (±0.5%);
   * **deduplication**: reviewed pairs (merged clusters sampled from the audit trail + high-similarity pairs that
     were *not* merged) → precision and recall of "same project" decisions; every false merge is listed.
4. **Screening trade-off** (spec §12) — the same organizations are re-run with `screening: none` (every chunk
   extracted, same selected documents thanks to deterministic, cached selection). `capex report` compares record
   and unique-project recall of the Jev-screened run against the all-chunks run, lists every project only found
   without screening together with the chunk's screening decision, and compares cost per stage.

## Safeguards that support the targets (measured, not assumed)

* **No invented values** — validation rejects records whose name does not occur in the source text and drops
  identifiers, budgets, yearly amounts and years that do not occur in it; counts by reason are in the metrics
  (`rejection_reasons`) of every job.
* **Page references** — every record's cited page is checked against the excerpt/name and corrected when the quote
  sits on another page of the chunk; unverifiable references are flagged (`reference_verified = check`).
* **Lifecycle** — advanced stages require explicit evidence words; stage claims citing only a future year are
  downgraded; the evidence text and `explicit`/`inferred` basis are exported for every project.
* **Deduplication** — hard cannot-link rules and constrained clustering; ambiguous pairs are reported as
  possible duplicates instead of being merged.

## Results

_Pending the pilot run._ Per-organization results, misses, false positives, page errors, stage errors and
deduplication errors will be listed here with their causes and proposed corrections, as the specification
requires when a target is missed.
