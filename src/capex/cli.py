"""Command-line interface.

    capex serve                       # REST API on :8000 (Swagger UI at /docs)
    capex run "City of Mesa, Arizona" --out out/mesa
    capex discover "Maricopa County, Arizona"
    capex parse https://example.gov/cip.pdf --out out/debug
    capex config
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import typer

from capex.config import get_app_config, get_settings

app = typer.Typer(add_completion=False, help="Public-sector capital project discovery & extraction")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "pdfminer", "anthropic", "typesafe_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the REST API."""
    import uvicorn

    uvicorn.run("capex.api.main:app", host=host, port=port, reload=reload)


@app.command()
def run(
    organization: str = typer.Argument(..., help="Public organization name"),
    out: Path = typer.Option(None, help="Output directory (default: data/runs/<slug>)"),
    max_documents: int = typer.Option(3, min=1, max=5),
    screening: str = typer.Option(None, help="jev | heuristic | none (default from config)"),
    force_refresh: bool = typer.Option(False, help="Re-download documents even if cached"),
    verbose: bool = False,
) -> None:
    """Run the full pipeline in-process (no API) and write JSON + Excel."""
    from capex.costs import CostLedger
    from capex.export import save_outputs
    from capex.normalize import slugify
    from capex.pipeline import Pipeline, PipelineOptions
    from capex.storage import Database, FileCache

    _setup_logging(verbose)
    settings, config = get_settings(), get_app_config()
    db = Database(settings.resolved_database_url)
    cache = FileCache(settings.data_dir / "cache")
    job_id = f"cli_{time.strftime('%Y%m%d_%H%M%S')}"
    ledger = CostLedger(config, max_cost_usd=config.limits.max_job_cost_usd, on_record=lambda r: db.add_usage(job_id, r))
    out = out or settings.data_dir / "runs" / slugify(organization)

    def progress(stage: str, counters: dict) -> None:
        typer.echo(f"  [{stage}] {json.dumps(counters)}")

    async def main() -> None:
        pipeline = Pipeline(config, cache, db)
        try:
            result = await pipeline.run(
                organization,
                PipelineOptions(max_documents=max_documents, screening=screening, force_refresh=force_refresh),
                job_id,
                ledger,
                progress,
            )
        finally:
            await pipeline.registry.aclose()
        paths = save_outputs(result, out)
        m = result.metrics
        typer.echo("")
        typer.echo(f"Organization:      {result.organization_name}")
        for doc in result.selected_documents:
            typer.echo(f"  #{doc.selection_rank} {doc.title[:90]}  ({doc.coverage_years or 'n/a'}, score {doc.score})")
        typer.echo(f"Raw records:       {m.projects_extracted_raw}")
        typer.echo(f"Unique projects:   {m.unique_projects}  (merged {m.duplicates_merged}, possible dups {m.possible_duplicates})")
        typer.echo(f"Total cost:        ${m.total_cost_usd:.4f}  (${(m.cost_per_unique_project_usd or 0):.5f}/project)")
        typer.echo(f"Execution time:    {m.execution_time_s:.1f}s")
        typer.echo(f"Workbook:          {paths.get('xlsx')}")

    asyncio.run(main())


@app.command()
def discover(organization: str, max_documents: int = 3, verbose: bool = False) -> None:
    """Discovery + ranking only: list candidates and the selected documents."""
    from capex.costs import CostLedger
    from capex.discovery import discover as run_discovery
    from capex.fetch import Fetcher
    from capex.llm import LLMRegistry
    from capex.storage import Database, FileCache

    _setup_logging(verbose)
    settings, config = get_settings(), get_app_config()
    db = Database(settings.resolved_database_url)
    cache = FileCache(settings.data_dir / "cache")
    ledger = CostLedger(config)
    registry = LLMRegistry(config)

    async def main() -> None:
        fetcher = Fetcher(config, cache, db)
        try:
            result = await run_discovery(organization, config, registry, ledger, cache, fetcher, max_documents)
        finally:
            await fetcher.aclose()
            await registry.aclose()
        p = result.profile
        typer.echo(f"{p.canonical_name} [{p.org_type}, {p.state}] domains={p.official_domains} search={result.search_providers}")
        typer.echo(f"queries: {p.queries}")
        for c in result.candidates[:30]:
            probe = c.probe
            flag = f"#{c.selection_rank}" if c.selected else "  "
            est = probe.est_projects if probe and probe.ok else "-"
            typer.echo(
                f"{flag} score={c.score if c.score is not None else '-':<6} prelim={c.prelim_score:<5} est={est:<5} "
                f"{c.file_type:<4} {c.doc_type.value[:28]:<28} {c.coverage_years or '':<14} {c.title[:70]}"
            )
            typer.echo(f"      {c.final_url or c.url}")
            if c.selected:
                typer.echo(f"      -> {c.selection_rationale}")
        for w in result.warnings:
            typer.echo(f"warning: {w}")
        typer.echo(f"discovery cost: ${ledger.total_cost:.4f}")

    asyncio.run(main())


@app.command()
def parse(source: str, out: Path = Path("data/debug"), verbose: bool = False) -> None:
    """Parse + chunk one document (URL or local PDF) and write pages/chunks JSON for inspection."""
    from capex.chunking import chunk_document
    from capex.fetch import Fetcher, FetchResult
    from capex.normalize import doc_id_for_url, sha256_text
    from capex.parsing import parse_document
    from capex.storage import Database, FileCache

    _setup_logging(verbose)
    settings, config = get_settings(), get_app_config()
    cache = FileCache(settings.data_dir / "cache")

    async def fetch() -> FetchResult:
        if Path(source).is_file():
            data = Path(source).read_bytes()
            digest = sha256_text(data.decode("latin-1"))
            path = cache.put_bytes("downloads", digest, data)
            return FetchResult(url=source, final_url=source, status=200, content=data, content_hash=digest, path=path)
        fetcher = Fetcher(config, cache, Database(settings.resolved_database_url))
        try:
            return await fetcher.fetch_document(source)
        finally:
            await fetcher.aclose()

    fetched = asyncio.run(fetch())
    started = time.perf_counter()
    doc, cached = parse_document(fetched, doc_id_for_url(source), cache, config.parsing)
    parse_s = time.perf_counter() - started
    chunks = chunk_document(doc, config.chunking)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pages.txt").write_text(
        "\n\n".join(f"=== PDF page {p.pdf_page} (printed {p.printed_page}) ===\n{p.text}" for p in doc.pages)
    )
    (out / "chunks.txt").write_text(
        "\n\n".join(
            f"##### {c.chunk_id} kind={c.kind} pages={c.pages} est_tokens={c.est_tokens} est_rows={c.est_rows}\n"
            f"--- context ---\n{c.context}\n--- text ---\n{c.text}"
            for c in chunks
        )
    )
    tokens = sum(c.est_tokens for c in chunks)
    typer.echo(
        f"pages={len(doc.pages)} parse={parse_s:.1f}s cached={cached} chunks={len(chunks)} est_tokens={tokens} "
        f"est_rows={sum(c.est_rows for c in chunks)} warnings={doc.warnings} -> {out}"
    )


@app.command()
def evaluate(
    gold_dir: Path = typer.Option(Path("evaluation/gold"), help="Folder with <org>.yaml gold samples"),
    out: Path = typer.Option(Path("evaluation/results.json")),
) -> None:
    """Score pilot outputs against manually verified samples (recall, precision, page refs, stages, dedup)."""
    from capex.evaluation.quality import evaluate_all

    results = evaluate_all(gold_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    overall = results["overall"]
    typer.echo(json.dumps({k: v for k, v in overall.items() if k != "counts"}, indent=2))
    typer.echo(f"details -> {out}")


@app.command()
def report(
    pilot_dir: Path = typer.Option(Path("pilot")),
    all_chunks_dir: Path = typer.Option(None, help="Pilot run with screening=none for the trade-off comparison"),
    out: Path = typer.Option(Path("pilot/summary.json")),
) -> None:
    """Aggregate pilot metrics (cost per organization/stage) and the screening trade-off."""
    from capex.evaluation.report import markdown_table, pilot_summary
    from capex.evaluation.tradeoff import compare

    rows = pilot_summary(pilot_dir)
    summary: dict = {"organizations": rows}
    if all_chunks_dir and all_chunks_dir.is_dir():
        summary["screening_tradeoff"] = [
            compare(pilot_dir / r["folder"], all_chunks_dir / r["folder"])
            for r in rows
            if (all_chunks_dir / r["folder"] / "metrics.json").is_file()
        ]
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    typer.echo(markdown_table(rows))
    typer.echo(f"\nsummary -> {out}")


@app.command()
def sample(
    pilot_folder: Path = typer.Argument(..., help="pilot/<org-slug> folder written by scripts/run_pilot.py"),
    out: Path = typer.Option(None, help="Review folder (default: evaluation/samples/<org-slug>)"),
    per_doc: int = typer.Option(4, help="Pages sampled per selected document"),
) -> None:
    """Write manual-review sheets (page text + system records) and a gold-file template."""
    from capex.evaluation.sampling import write_samples

    settings = get_settings()
    out = out or Path("evaluation/samples") / pilot_folder.name
    template = write_samples(pilot_folder, settings.data_dir / "cache", out, per_doc)
    typer.echo(f"review sheets -> {out}\ngold template -> {template}")


@app.command("config")
def show_config() -> None:
    """Print the effective configuration (secrets are referenced by env-var name only)."""
    typer.echo(json.dumps(get_app_config().model_dump(mode="json"), indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
