"""HTTP fetching with caching, conditional re-download, size limits, robots.txt and per-host politeness.

Documents are stored content-addressed (SHA-256). The ``documents`` table remembers the ETag /
Last-Modified per URL, so a repeat run issues a conditional GET and re-uses the cached bytes — and
therefore every downstream cache (parsed pages, chunks, screening, extraction) — when the document
has not changed (spec §12: avoid reprocessing unchanged documents).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from capex.config import AppConfig
from capex.normalize import canonical_url, sha256_text
from capex.storage.cache import FileCache
from capex.storage.db import Database

log = logging.getLogger(__name__)


_CHALLENGE = re.compile(
    rb"just a moment|cf-chl|challenge-platform|attention required|access denied|captcha|perimeterx|incapsula|"
    rb"request unsuccessful|bot protection",
    re.IGNORECASE,
)


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str = ""
    content: bytes | None = None
    content_hash: str | None = None
    from_cache: bool = False
    error: str | None = None
    path: Path | None = None
    blocked: bool = False  # host answered with a bot-protection challenge
    archive_url: str | None = None  # content retrieved from an Internet Archive capture of ``url``

    @property
    def ok(self) -> bool:
        return self.content is not None and self.error is None

    @property
    def is_pdf(self) -> bool:
        return bool(self.content and self.content[:1024].lstrip().startswith(b"%PDF")) or (
            "pdf" in self.content_type.lower()
        )

    @property
    def is_html(self) -> bool:
        if self.is_pdf:
            return False
        if "html" in self.content_type.lower():
            return True
        head = (self.content or b"")[:512].lower()
        return b"<html" in head or b"<!doctype html" in head


class Fetcher:
    def __init__(self, config: AppConfig, cache: FileCache, db: Database | None = None) -> None:
        self.config = config
        self.cache = cache
        self.db = db
        self.max_bytes = config.discovery.max_document_mb * 1024 * 1024
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(config.limits.http_timeout_s, connect=20.0),
            follow_redirects=True,
            headers={
                "User-Agent": config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
            },
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        self._robots: dict[str, RobotFileParser | None] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._host_last: dict[str, float] = {}
        self.min_host_interval = 0.25

    async def aclose(self) -> None:
        await self.http.aclose()

    @staticmethod
    def _is_challenge(resp: httpx.Response) -> bool:
        if resp.status_code not in (403, 429, 503):
            return False
        server = resp.headers.get("server", "").lower()
        return "cloudflare" in server or bool(_CHALLENGE.search(resp.content[:20000] if resp.content else b""))

    async def _archive_capture(self, url: str) -> str | None:
        """Latest successful Internet Archive capture of ``url`` (raw ``id_`` form), if any."""
        try:
            resp = await self.http.get(
                "https://web.archive.org/cdx/search/cdx",
                params={"url": url, "output": "json", "limit": "-1", "filter": "statuscode:200"},
                timeout=45.0,
            )
            rows = resp.json() if resp.status_code == 200 and resp.content.strip() else []
        except (httpx.HTTPError, ValueError):
            return None
        if len(rows) < 2:
            return None
        header, last = rows[0], rows[-1]
        record = dict(zip(header, last, strict=False))
        if not record.get("timestamp"):
            return None
        return f"https://web.archive.org/web/{record['timestamp']}id_/{record.get('original') or url}"

    # -- politeness ---------------------------------------------------------------------------

    async def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            parser: RobotFileParser | None = RobotFileParser()
            try:
                resp = await self.http.get(f"{host}/robots.txt", timeout=10.0)
                if resp.status_code == 200 and "text" in resp.headers.get("content-type", "text"):
                    parser.parse(resp.text.splitlines())  # type: ignore[union-attr]
                else:
                    parser = None
            except httpx.HTTPError:
                parser = None
            self._robots[host] = parser
        robots = self._robots[host]
        return True if robots is None else robots.can_fetch(self.config.user_agent, url)

    async def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self.min_host_interval - (time.monotonic() - self._host_last.get(host, 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
            self._host_last[host] = time.monotonic()

    # -- fetching -----------------------------------------------------------------------------

    async def _get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(3):
            await self._throttle(url)
            try:
                async with self.http.stream("GET", url, headers=headers) as resp:
                    declared = int(resp.headers.get("content-length") or 0)
                    if declared and declared > self.max_bytes:
                        raise ValueError(f"document too large ({declared / 1e6:.0f} MB)")
                    body = bytearray()
                    async for part in resp.aiter_bytes():
                        body.extend(part)
                        if len(body) > self.max_bytes:
                            raise ValueError(f"document larger than {self.max_bytes / 1e6:.0f} MB")
                    resp._content = bytes(body)  # materialise for callers
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    return resp
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                await asyncio.sleep(1.0 * (attempt + 1))
        raise last_exc or RuntimeError(f"failed to fetch {url}")

    async def fetch_page(self, url: str, ttl_hours: float = 24) -> FetchResult:
        """HTML/landing pages for crawling (short TTL cache keyed by URL)."""
        key = sha256_text(canonical_url(url))
        cached = self.cache.get_json("pages", key, ttl_seconds=ttl_hours * 3600)
        if cached:
            return FetchResult(
                url=url,
                final_url=cached["final_url"],
                status=cached["status"],
                content_type=cached["content_type"],
                content=cached["text"].encode("utf-8"),
                from_cache=True,
            )
        if not await self._allowed(url):
            return FetchResult(url=url, final_url=url, status=0, error="disallowed by robots.txt")
        try:
            resp = await self._get(url)
        except Exception as exc:
            return FetchResult(url=url, final_url=url, status=0, error=f"{type(exc).__name__}: {exc}")
        content_type = resp.headers.get("content-type", "")
        result = FetchResult(
            url=url, final_url=str(resp.url), status=resp.status_code, content_type=content_type, content=resp.content
        )
        if resp.status_code >= 400:
            result.blocked = self._is_challenge(resp)
            result.error = "blocked by host bot protection (challenge page)" if result.blocked else f"HTTP {resp.status_code}"
            result.content = None
            return result
        if result.is_html:
            text = resp.content.decode(resp.encoding or "utf-8", errors="replace")
            self.cache.put_json(
                "pages",
                key,
                {"final_url": str(resp.url), "status": resp.status_code, "content_type": content_type, "text": text},
            )
        return result

    async def fetch_document(self, url: str, force: bool = False) -> FetchResult:
        """Download a document; conditional GET + content-hash cache make repeat runs free."""
        canon = canonical_url(url)
        known = self.db.get_document(canon) if self.db else None
        headers: dict[str, str] = {}
        cached_bytes: bytes | None = None
        if known and known.content_hash and not force:
            cached_bytes = self.cache.get_bytes("downloads", known.content_hash)
            if cached_bytes is not None:
                if known.etag:
                    headers["If-None-Match"] = known.etag
                if known.last_modified:
                    headers["If-Modified-Since"] = known.last_modified
        if not await self._allowed(url):
            if cached_bytes is not None and known:
                return self._from_cache(url, known.final_url or url, known.content_type or "", cached_bytes)
            return FetchResult(url=url, final_url=url, status=0, error="disallowed by robots.txt")
        try:
            resp = await self._get(url, headers=headers or None)
        except Exception as exc:
            if cached_bytes is not None and known:
                log.warning("fetch failed, using cached copy of %s: %s", url, exc)
                return self._from_cache(url, known.final_url or url, known.content_type or "", cached_bytes)
            return FetchResult(url=url, final_url=url, status=0, error=f"{type(exc).__name__}: {exc}")
        if resp.status_code == 304 and cached_bytes is not None and known:
            return self._from_cache(url, str(resp.url), known.content_type or "", cached_bytes)
        archive_url = None
        if resp.status_code >= 400:
            blocked = self._is_challenge(resp)
            error = "blocked by host bot protection (challenge page)" if blocked else f"HTTP {resp.status_code}"
            if cached_bytes is not None and known:
                return self._from_cache(url, known.final_url or url, known.content_type or "", cached_bytes)
            archived = await self._archive_capture(url) if self.config.discovery.archive_fallback else None
            if not archived:
                return FetchResult(url=url, final_url=str(resp.url), status=resp.status_code, error=error, blocked=blocked)
            try:
                resp = await self._get(archived)
            except Exception as exc:
                return FetchResult(url=url, final_url=url, status=0, error=f"{error}; archive copy failed: {exc}", blocked=blocked)
            if resp.status_code >= 400:
                return FetchResult(url=url, final_url=url, status=resp.status_code, error=f"{error}; archive HTTP {resp.status_code}", blocked=blocked)
            archive_url = archived
            log.info("using Internet Archive capture for %s (%s)", url, error)
        content = resp.content
        digest = hashlib.sha256(content).hexdigest()
        path = self.cache.put_bytes("downloads", digest, content)
        content_type = resp.headers.get("content-type", "")
        final_url = url if archive_url else str(resp.url)
        if self.db:
            self.db.upsert_document(
                canon,
                final_url=final_url,
                content_hash=digest,
                content_type=content_type,
                size_bytes=len(content),
                etag=resp.headers.get("etag"),
                last_modified=resp.headers.get("last-modified"),
            )
        return FetchResult(
            url=url,
            final_url=final_url,
            status=resp.status_code,
            content_type=content_type,
            content=content,
            content_hash=digest,
            from_cache=bool(cached_bytes is not None and known and known.content_hash == digest),
            path=path,
            archive_url=archive_url,
        )

    def _from_cache(self, url: str, final_url: str, content_type: str, content: bytes) -> FetchResult:
        digest = hashlib.sha256(content).hexdigest()
        return FetchResult(
            url=url,
            final_url=final_url,
            status=200,
            content_type=content_type,
            content=content,
            content_hash=digest,
            from_cache=True,
            path=self.cache.path("downloads", digest, ""),
        )
