"""Candidate documents: official-site crawling, merging of search hits and links, cheap metadata.

Search finds landing pages and documents; the crawler follows capital/budget/program links on the
organization's official domains (depth-limited) and collects document links with their anchor text —
which is how all versions of a CIP (adopted, proposed, prior years) are found even when search only
surfaces one of them.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from urllib.parse import urljoin, urlsplit

import tldextract
from bs4 import BeautifulSoup

from capex.discovery.classify import (
    BUDGET_TERMS,
    NEGATIVE_TERMS,
    STRONG_TERMS,
    classify_doc_type,
    classify_status,
    coverage_from_text,
    current_fiscal_year,
)
from capex.fetch import Fetcher
from capex.models import DocType, DocumentCandidate, OrgProfile, SearchHit
from capex.normalize import canonical_url, clean_text, doc_id_for_url

log = logging.getLogger(__name__)

_extract = tldextract.TLDExtract(suffix_list_urls=())  # bundled public-suffix list, no network

DOC_LINK = re.compile(
    r"\.pdf($|[?#])|documentcenter/view|showpublisheddocument|/download(s)?/|/files?/|\.ashx|"
    r"/document(s)?/|/home/showdocument|/archive(center)?/viewfile|/media/[^/]+/.*(cip|capital|budget|program)",
    re.IGNORECASE,
)
FOLLOW = re.compile(
    r"capital|\bcip\b|improvement|budget|program|construction|five[- ]?year|5[- ]?year|\btip\b|stip|bond|"
    r"facilit|infrastructure|projects?|finance|planning|transportation-?program",
    re.IGNORECASE,
)
SKIP_LINK = re.compile(
    r"facebook|twitter|x\.com|linkedin|youtube|instagram|mailto:|tel:|javascript:|#$|/login|/signin|"
    r"/calendar|/news|/events?/|/jobs?|/careers?|/employment|\.(jpg|jpeg|png|gif|zip|mp4|mp3|docx?|pptx?)($|\?)",
    re.IGNORECASE,
)
EXCLUDED_DOMAINS = {"wikipedia.org", "facebook.com", "linkedin.com", "youtube.com", "twitter.com", "x.com", "reddit.com"}


def registered_domain(url_or_host: str) -> str:
    host = urlsplit(url_or_host).netloc or url_or_host
    parts = _extract(host)
    return f"{parts.domain}.{parts.suffix}".lower() if parts.suffix else parts.domain.lower()


def is_gov(domain: str) -> bool:
    return domain.endswith((".gov", ".us", ".mil")) or ".gov." in domain or ".k12." in domain


def file_type_for(url: str) -> str:
    path = urlsplit(url).path.lower()
    if path.endswith(".pdf") or "documentcenter/view" in path or "showpublisheddocument" in path:
        return "pdf"
    if path.endswith((".xlsx", ".xls", ".csv")):
        return "xlsx"
    if path.endswith((".doc", ".docx", ".ppt", ".pptx", ".zip")):
        return "other"
    return "html"


def official_domains(profile: OrgProfile, hits: list[SearchHit]) -> set[str]:
    domains = {registered_domain(d) for d in profile.official_domains if d}
    if domains:
        return domains
    # fall back to the most frequent government domain among hits whose title names the organization
    tokens = [t for t in re.findall(r"[a-z]+", profile.canonical_name.lower()) if len(t) > 3 and t not in {"city", "county", "department", "state"}]
    votes: Counter[str] = Counter()
    for hit in hits:
        domain = registered_domain(hit.url)
        title = hit.title.lower()
        if is_gov(domain) and (not tokens or sum(t in title or t in domain for t in tokens) >= max(1, len(tokens) // 2)):
            votes[domain] += 1
    return {d for d, _ in votes.most_common(1)}


def _anchor_text(link: object) -> str:
    text = clean_text(link.get_text(" "))  # type: ignore[attr-defined]
    for attr in ("title", "aria-label"):
        if len(text) < 6 and link.get(attr):  # type: ignore[attr-defined]
            text = clean_text(str(link.get(attr)))  # type: ignore[attr-defined]
    return text[:240]


async def crawl(
    seeds: list[str],
    official: set[str],
    fetcher: Fetcher,
    max_pages: int,
    max_depth: int,
) -> list[tuple[str, str, str]]:
    """Breadth-first crawl of official capital/budget pages; returns (doc_url, anchor_text, parent_url)."""
    queue: list[tuple[str, int]] = [(s, 0) for s in seeds]
    seen: set[str] = set()
    found: dict[str, tuple[str, str, str]] = {}
    fetched = 0
    while queue and fetched < max_pages:
        url, depth = queue.pop(0)
        canon = canonical_url(url)
        if canon in seen:
            continue
        seen.add(canon)
        result = await fetcher.fetch_page(url)
        fetched += 1
        if not result.ok or not result.is_html:
            continue
        html = (result.content or b"").decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        for link in soup.find_all("a", href=True):
            href = str(link["href"]).strip()
            if not href or SKIP_LINK.search(href):
                continue
            absolute = urljoin(result.final_url, href)
            if not absolute.startswith("http"):
                continue
            anchor = _anchor_text(link)
            domain = registered_domain(absolute)
            if DOC_LINK.search(absolute):
                key = canonical_url(absolute)
                if key not in found or len(anchor) > len(found[key][1]):
                    found[key] = (absolute, anchor, result.final_url)
            elif domain in official and depth + 1 <= max_depth and (FOLLOW.search(urlsplit(absolute).path) or FOLLOW.search(anchor)):
                if canonical_url(absolute) not in seen:
                    queue.append((absolute, depth + 1))
    return list(found.values())


def prelim_score(candidate: DocumentCandidate, fiscal_year: int) -> float:
    text = f"{candidate.title} {urlsplit(candidate.url).path} {candidate.snippet}"
    score = 0.0
    score += 3.0 if STRONG_TERMS.search(text) else 0.0
    score += 1.0 if BUDGET_TERMS.search(text) else 0.0
    score -= 2.5 if NEGATIVE_TERMS.search(f"{candidate.title} {urlsplit(candidate.url).path}") else 0.0
    score += 2.0 if candidate.is_official_domain else (0.8 if is_gov(candidate.domain) else -1.0)
    score += {"pdf": 1.5, "html": 0.3, "xlsx": 0.2}.get(candidate.file_type, -1.0)
    year = candidate.coverage_end_year or candidate.coverage_start_year
    if year:
        if year >= fiscal_year - 1:
            score += 1.5
        else:
            score -= min(3.0, 0.6 * (fiscal_year - 1 - year))
    score += 0.5 if candidate.doc_type != DocType.OTHER else 0.0
    score += 0.3 * min(3, len(candidate.discovered_via))
    return round(score, 3)


def build_candidates(
    hits: list[SearchHit],
    links: list[tuple[str, str, str]],
    profile: OrgProfile,
    official: set[str],
) -> dict[str, DocumentCandidate]:
    fiscal_year = current_fiscal_year()
    pool: dict[str, DocumentCandidate] = {}

    def add(url: str, title: str, via: str, snippet: str = "", page_age: str | None = None) -> None:
        if not url.startswith("http"):
            return
        domain = registered_domain(url)
        if domain in EXCLUDED_DOMAINS:
            return
        key = canonical_url(url)
        candidate = pool.get(key)
        if candidate is None:
            candidate = DocumentCandidate(
                doc_id=doc_id_for_url(url),
                url=url,
                title=clean_text(title) or urlsplit(url).path.rsplit("/", 1)[-1],
                file_type=file_type_for(url),  # type: ignore[arg-type]
                domain=domain,
                is_official_domain=domain in official,
                snippet=clean_text(snippet)[:400],
                publication_date=page_age,
            )
            pool[key] = candidate
        elif title and (len(clean_text(title)) > len(candidate.title) or candidate.title.lower().endswith(".pdf")):
            candidate.title = clean_text(title)
        if via not in candidate.discovered_via:
            candidate.discovered_via.append(via)
        if snippet and not candidate.snippet:
            candidate.snippet = clean_text(snippet)[:400]

    for hit in hits:
        add(hit.url, hit.title, f"search:{hit.provider}:{hit.query}", hit.snippet, hit.page_age)
    for url, anchor, parent in links:
        add(url, anchor, f"crawl:{parent}")

    for candidate in pool.values():
        text = f"{candidate.title} {candidate.url} {candidate.snippet}"
        candidate.doc_type = classify_doc_type(text)
        candidate.doc_status = classify_status(f"{candidate.title} {candidate.url}")
        start, end = coverage_from_text(candidate.title, urlsplit(candidate.url).path.replace("-", " ").replace("_", " "), candidate.snippet)
        candidate.coverage_start_year, candidate.coverage_end_year = start, end
        candidate.document_fiscal_year = start
        candidate.publisher = profile.canonical_name if candidate.is_official_domain else candidate.domain
        candidate.prelim_score = prelim_score(candidate, fiscal_year)
    return pool
