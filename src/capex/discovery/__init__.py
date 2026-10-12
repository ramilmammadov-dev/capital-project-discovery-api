"""Document discovery & selection (spec §3-4):

organization name -> query plan -> web search -> official-site crawl -> candidate inventory ->
probe the most promising candidates -> batched metadata profiling -> score -> select top N.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from capex.config import AppConfig
from capex.costs import CostLedger
from capex.discovery.candidates import (
    FOLLOW,
    build_candidates,
    crawl,
    file_type_for,
    official_domains,
    registered_domain,
)
from capex.discovery.classify import series_key
from capex.discovery.planner import plan_organization
from capex.discovery.probe import llm_profile, mark_mirrors, probe_candidate
from capex.discovery.ranking import select_documents
from capex.discovery.search import run_searches
from capex.fetch import Fetcher
from capex.llm import LLMRegistry
from capex.models import DocumentCandidate, OrgProfile
from capex.normalize import canonical_url
from capex.storage.cache import FileCache

log = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    profile: OrgProfile
    candidates: list[DocumentCandidate]
    selected: list[DocumentCandidate]
    search_providers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _probe_order(pool: dict[str, DocumentCandidate], limit: int, already: set[str]) -> list[DocumentCandidate]:
    ordered = sorted(pool.values(), key=lambda c: -c.prelim_score)
    chosen: list[DocumentCandidate] = []
    per_series: Counter[str] = Counter()
    for c in ordered:
        if c.doc_id in already or c.file_type not in ("pdf", "html") or c.prelim_score < 0.5:
            continue
        key = series_key(c.title) or c.doc_id
        if per_series[key] >= 3:  # newest few versions of a series are enough
            continue
        per_series[key] += 1
        chosen.append(c)
        if len(chosen) >= limit:
            break
    return chosen


async def discover(
    organization: str,
    config: AppConfig,
    registry: LLMRegistry,
    ledger: CostLedger,
    cache: FileCache,
    fetcher: Fetcher,
    max_documents: int = 3,
    progress: Callable[[str, dict[str, int]], None] | None = None,
) -> DiscoveryResult:
    warnings: list[str] = []
    profile = await plan_organization(organization, config, registry, ledger, cache)
    hits, providers = await run_searches(profile.queries, config, ledger, registry, cache, region=profile.state)
    if not providers:
        warnings.append("no web search provider configured; discovery relies on crawling known domains only")
    if not hits:
        failed = [r for r in ledger.records if r.task == "web_search" and not r.success]
        reason = f" ({failed[0].error})" if failed else ""
        warnings.append(f"web search returned no results{reason}")

    official = official_domains(profile, hits)
    profile.official_domains = sorted(official)

    seeds: list[str] = []
    for hit in sorted(hits, key=lambda h: h.rank):
        if registered_domain(hit.url) in official and file_type_for(hit.url) == "html":
            if (FOLLOW.search(hit.url) or FOLLOW.search(hit.title)) and canonical_url(hit.url) not in {canonical_url(s) for s in seeds}:
                seeds.append(hit.url)
    seeds = seeds[:12]
    if not seeds:
        seeds = [f"https://www.{d}/" for d in sorted(official)]
    links = await crawl(seeds, official, fetcher, config.discovery.crawl_max_pages, config.discovery.crawl_max_depth)
    pool = build_candidates(hits, links, profile, official)
    if progress:
        progress("discovering", {"documents_discovered": len(pool)})

    sem = asyncio.Semaphore(4)
    probed: set[str] = set()

    async def probe(c: DocumentCandidate) -> list[tuple[str, str, str]]:
        async with sem:
            probed.add(c.doc_id)
            try:
                return await probe_candidate(c, fetcher, profile, config, cache)
            except Exception as exc:  # a broken candidate must not break discovery
                log.warning("probe failed for %s: %s", c.url, exc)
                from capex.models import ProbeResult

                c.probe = ProbeResult(ok=False, error=f"{type(exc).__name__}: {exc}")
                return []

    first_wave = _probe_order(pool, config.discovery.max_probe, probed)
    hub_links = [link for links_ in await asyncio.gather(*(probe(c) for c in first_wave)) for link in links_]
    if hub_links:
        before = set(pool)
        extra_pool = build_candidates([], hub_links, profile, official)
        for key, candidate in extra_pool.items():
            if key not in pool:
                candidate.discovered_via = [v.replace("crawl:", "hub:") for v in candidate.discovered_via]
                pool[key] = candidate
        new = {k: v for k, v in pool.items() if k not in before}
        second_wave = _probe_order(new, max(4, config.discovery.max_probe // 3), probed)
        await asyncio.gather(*(probe(c) for c in second_wave))

    await llm_profile([c for c in pool.values() if c.probe], profile.canonical_name, config, registry, ledger, cache)
    mark_mirrors(list(pool.values()), profile.canonical_name, profile.aliases)
    selected = select_documents(list(pool.values()), config.discovery, top_n=max_documents)
    if len(selected) < max_documents:
        warnings.append(f"only {len(selected)} usable document(s) found (requested {max_documents})")

    candidates = sorted(
        pool.values(),
        key=lambda c: (not c.selected, c.selection_rank or 99, -(c.score or 0.0), -c.prelim_score),
    )[: max(config.discovery.max_candidates, len(selected))]
    if progress:
        progress(
            "ranking",
            {
                "documents_discovered": len(pool),
                "documents_probed": sum(1 for c in pool.values() if c.probe),
                "documents_selected": len(selected),
            },
        )
    return DiscoveryResult(profile=profile, candidates=candidates, selected=selected, search_providers=providers, warnings=warnings)
