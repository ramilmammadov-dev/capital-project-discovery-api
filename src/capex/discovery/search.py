"""Web search providers (spec §3: web search APIs, search engines and/or LLM-assisted search).

* ``anthropic``  — LLM-assisted: Claude runs the planned queries with the server-side web_search tool
* ``brave``      — Brave Search API
* ``serper``     — Google results via Serper
* ``duckduckgo`` — keyless HTML endpoint (fallback when no paid provider is configured)

All providers return ``SearchHit`` lists, are cached per (provider, query) and every request is
recorded in the cost ledger (searches carry a per-request fee).
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
from bs4 import BeautifulSoup

from capex.config import AppConfig
from capex.costs import CostLedger
from capex.llm import AnthropicClient, LLMRegistry
from capex.models import SearchHit
from capex.normalize import sha256_text
from capex.storage.cache import FileCache

log = logging.getLogger(__name__)


class SearchProvider(ABC):
    name = "abstract"

    def __init__(self, config: AppConfig, ledger: CostLedger) -> None:
        self.config = config
        self.ledger = ledger
        self.fee = config.search.price_per_search_usd.get(self.name, 0.0)

    @abstractmethod
    async def search_many(self, queries: list[str], per_query: int, region: str | None) -> list[SearchHit]: ...


class AnthropicSearch(SearchProvider):
    name = "anthropic"

    def __init__(self, config: AppConfig, ledger: CostLedger, registry: LLMRegistry) -> None:
        super().__init__(config, ledger)
        self.registry = registry

    async def search_many(self, queries: list[str], per_query: int, region: str | None) -> list[SearchHit]:
        client = self.registry.get("anthropic")
        assert isinstance(client, AnthropicClient)
        groups = [queries[i : i + 4] for i in range(0, len(queries), 4)]  # small groups keep context short
        results = await asyncio.gather(
            *(
                client.web_search(
                    model=self.config.search.anthropic_model,
                    queries=group,
                    tool_version=self.config.search.anthropic_tool_version,
                    price_per_search=self.fee,
                    stage="discovery",
                    ledger=self.ledger,
                    region=region,
                )
                for group in groups
            ),
            return_exceptions=True,
        )
        hits: list[SearchHit] = []
        for result in results:
            if isinstance(result, BaseException):
                log.warning("anthropic web search failed: %s", result)
                continue
            hits.extend(result)
        return hits


class _HttpSearch(SearchProvider):
    model_name = "web-search"

    def __init__(self, config: AppConfig, ledger: CostLedger) -> None:
        super().__init__(config, ledger)
        self.http = httpx.AsyncClient(timeout=20.0, headers={"User-Agent": config.user_agent})

    async def _one(self, query: str, per_query: int) -> list[SearchHit]:
        raise NotImplementedError

    async def search_many(self, queries: list[str], per_query: int, region: str | None) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for query in queries:  # sequential: these APIs rate-limit aggressively
            started = time.perf_counter()
            try:
                found = await self._one(query, per_query)
                self.ledger.record(
                    stage="discovery",
                    task="web_search",
                    provider=self.name,
                    model=self.model_name,
                    search_requests=1,
                    request_fee_usd=self.fee,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
                hits.extend(found)
            except Exception as exc:
                self.ledger.record(
                    stage="discovery",
                    task="web_search",
                    provider=self.name,
                    model=self.model_name,
                    success=False,
                    error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
                log.warning("%s search failed for %r: %s", self.name, query, exc)
            await asyncio.sleep(0.4)
        await self.http.aclose()
        return hits


class BraveSearch(_HttpSearch):
    name = "brave"
    model_name = "brave-web-search"

    def __init__(self, config: AppConfig, ledger: CostLedger, api_key: str) -> None:
        super().__init__(config, ledger)
        self.api_key = api_key

    async def _one(self, query: str, per_query: int) -> list[SearchHit]:
        resp = await self.http.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(per_query, 20), "country": "us"},
            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
        )
        resp.raise_for_status()
        results = (resp.json().get("web") or {}).get("results") or []
        return [
            SearchHit(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=BeautifulSoup(r.get("description", ""), "lxml").get_text(" "),
                query=query,
                provider=self.name,
                rank=i + 1,
                page_age=r.get("page_age"),
            )
            for i, r in enumerate(results)
            if r.get("url")
        ]


class SerperSearch(_HttpSearch):
    name = "serper"
    model_name = "serper-google-search"

    def __init__(self, config: AppConfig, ledger: CostLedger, api_key: str) -> None:
        super().__init__(config, ledger)
        self.api_key = api_key

    async def _one(self, query: str, per_query: int) -> list[SearchHit]:
        resp = await self.http.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": min(per_query, 20), "gl": "us"},
            headers={"X-API-KEY": self.api_key},
        )
        resp.raise_for_status()
        organic = resp.json().get("organic") or []
        return [
            SearchHit(
                url=r.get("link", ""),
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                query=query,
                provider=self.name,
                rank=i + 1,
                page_age=r.get("date"),
            )
            for i, r in enumerate(organic)
            if r.get("link")
        ]


class DuckDuckGoSearch(_HttpSearch):
    name = "duckduckgo"
    model_name = "duckduckgo-html"

    async def _one(self, query: str, per_query: int) -> list[SearchHit]:
        resp = await self.http.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "us-en"})
        resp.raise_for_status()
        if resp.status_code == 202 or "anomaly" in resp.text[:20000].lower():
            raise RuntimeError("DuckDuckGo refused automated queries (bot check); configure a search API key")
        soup = BeautifulSoup(resp.text, "lxml")
        hits = []
        for i, result in enumerate(soup.select(".result")):
            link = result.select_one("a.result__a")
            if not link or not link.get("href"):
                continue
            href = str(link["href"])
            if "uddg=" in href:
                href = unquote(parse_qs(urlsplit(href).query).get("uddg", [href])[0])
            if not href.startswith("http"):
                continue
            snippet = result.select_one(".result__snippet")
            hits.append(
                SearchHit(
                    url=href,
                    title=link.get_text(" ", strip=True),
                    snippet=snippet.get_text(" ", strip=True) if snippet else "",
                    query=query,
                    provider=self.name,
                    rank=i + 1,
                )
            )
            if len(hits) >= per_query:
                break
        return hits


def available_providers(config: AppConfig, ledger: CostLedger, registry: LLMRegistry) -> list[SearchProvider]:
    p = config.providers
    providers: list[SearchProvider] = []
    for name in config.search.providers:
        if name == "anthropic" and registry.available("anthropic"):
            providers.append(AnthropicSearch(config, ledger, registry))
        elif name == "brave" and config.secret(p.brave_api_key_env):
            providers.append(BraveSearch(config, ledger, config.secret(p.brave_api_key_env) or ""))
        elif name == "serper" and config.secret(p.serper_api_key_env):
            providers.append(SerperSearch(config, ledger, config.secret(p.serper_api_key_env) or ""))
    if "duckduckgo" in config.search.providers and (not providers or not config.search.duckduckgo_only_as_fallback):
        providers.append(DuckDuckGoSearch(config, ledger))
    return providers


async def run_searches(
    queries: list[str],
    config: AppConfig,
    ledger: CostLedger,
    registry: LLMRegistry,
    cache: FileCache,
    region: str | None = None,
) -> tuple[list[SearchHit], list[str]]:
    """Run all queries on every available provider (cached). Returns (hits, provider names)."""
    providers = available_providers(config, ledger, registry)
    ttl = config.search.cache_ttl_hours * 3600
    per_query = config.search.results_per_query
    all_hits: list[SearchHit] = []

    async def run(provider: SearchProvider) -> None:
        # cached per provider for the whole (deterministic, planner-cached) query set; LLM-assisted search
        # may rephrase individual queries, so per-query keys would not be stable
        key = sha256_text(f"{provider.name}|{per_query}|" + "\n".join(queries))
        cached = cache.get_json("search", key, ttl_seconds=ttl)
        if cached is not None:
            ledger.cache_hit("web_search")
            all_hits.extend(SearchHit.model_validate(h) for h in cached)
            return
        fresh = await provider.search_many(queries, per_query, region)
        if fresh:
            cache.put_json("search", key, [h.model_dump() for h in fresh])
        all_hits.extend(fresh)

    await asyncio.gather(*(run(p) for p in providers))
    return all_hits, [p.name for p in providers]
