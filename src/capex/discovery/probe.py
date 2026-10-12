"""Probing candidate documents before selection (spec §4).

A probe downloads the document (cached) and samples ~14 pages with pdfium's fast text layer to
measure — deterministically — whether the document contains *itemized* projects (vs. aggregate
department budgets), roughly how many, whether it has budgets / years / descriptions, its fiscal-year
coverage and status, and identifier/name tokens for overlap estimation. A single batched call to the
cheap model then reads the covers of all probed documents to fill publisher, publication date,
document type, status and coverage years.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from capex.config import AppConfig
from capex.costs import CostLedger
from capex.discovery.candidates import DOC_LINK, _anchor_text
from capex.discovery.classify import classify_doc_type, classify_status, coverage_from_text, series_key
from capex.fetch import Fetcher
from capex.llm import LLMError, LLMRegistry
from capex.models import DocStatus, DocType, DocumentCandidate, OrgProfile, ProbeResult
from capex.normalize import clean_text, find_id_tokens, name_key, sha256_text, year_tokens
from capex.parsing.html import parse_html
from capex.parsing.pdf import fast_page_texts, page_count, pdf_metadata
from capex.storage.cache import FileCache

log = logging.getLogger(__name__)

PROBE_VERSION = "probe-v2"

_AMOUNT = re.compile(r"\$\s?\d[\d,.]*|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_CAPITAL = re.compile(
    r"\b(improvements?|rehabilitat\w*|replace\w*|expan\w*|construct\w*|upgrade\w*|design|renovat\w*|"
    r"widen\w*|reconstruct\w*|install\w*|pavement|bridge|interchange|pipeline|main|reservoir|station|"
    r"plant|park|building|roof|hvac|facility|road|street|lane|well|sewer|drainage|signal|terminal|runway)\b",
    re.IGNORECASE,
)
_LABELS = re.compile(
    r"\b(project (name|number|no\.?|id|description|title|location|scope|manager)|description|scope of work|"
    r"location|estimated (total )?cost|total project cost|funding source)\b",
    re.IGNORECASE,
)
_SUMMARY = re.compile(
    r"\b(revenues?|fund balance|personnel|full[- ]time|fte|operating|debt service|transfers?|property tax|"
    r"department total|expenditures by)\b",
    re.IGNORECASE,
)


def _page_signals(text: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = [
        ln
        for ln in lines
        if _AMOUNT.search(ln) and (find_id_tokens(ln) or _CAPITAL.search(ln) or len(ln.split()) >= 5)
    ]
    ids = set(find_id_tokens(text))
    amounts = len(_AMOUNT.findall(text))
    labels = len(_LABELS.findall(text))
    descriptive = sum(1 for ln in lines if len(ln.split()) >= 12 and _CAPITAL.search(ln))
    itemized = len(rows) >= 3 or (len(ids) >= 3 and amounts >= 3) or (labels >= 2 and amounts >= 1 and _CAPITAL.search(text) is not None)
    return {
        "itemized": bool(itemized),
        "rows": len(rows),
        "ids": ids,
        "row_keys": {name_key(" ".join(r.split()[:5])) for r in rows[:60]},
        "amounts": amounts,
        "descriptive": descriptive,
        "summary": len(_SUMMARY.findall(text)),
        "has_text": len(text) > 200,
    }


def _sample_pages(total: int, k: int) -> list[int]:
    if total <= k:
        return list(range(1, total + 1))
    head = [1, 2, 3]
    step = (total - 3) / (k - 3)
    rest = [int(4 + i * step) for i in range(k - 3)]
    return sorted({p for p in head + rest if 1 <= p <= total})


def _profile_texts(texts: dict[int, str], total_pages: int, org_terms: list[str]) -> ProbeResult:
    signals = {p: _page_signals(t) for p, t in texts.items()}
    sampled = sorted(texts)
    body = [p for p in sampled if p > 3] or sampled
    itemized_pages = [p for p in body if signals[p]["itemized"]]
    ratio = len(itemized_pages) / max(1, len(body))
    per_page = (
        sum(max(len(signals[p]["ids"]), signals[p]["rows"] // 2, 1) for p in itemized_pages) / len(itemized_pages)
        if itemized_pages
        else 0.0
    )
    first = "\n".join(texts.get(p, "") for p in sampled[:3])
    lowered = first.lower()
    mentions = sum(lowered.count(t.lower()) for t in org_terms if t)
    years = sorted({y for p in sampled[:3] for _, y in year_tokens(texts.get(p, ""))})
    all_ids: set[str] = set()
    keys: set[str] = set()
    for p in itemized_pages:
        all_ids |= signals[p]["ids"]
        keys |= signals[p]["row_keys"]
    summary = sum(signals[p]["summary"] for p in sampled)
    return ProbeResult(
        ok=True,
        page_count=total_pages,
        text_page_ratio=round(sum(signals[p]["has_text"] for p in sampled) / max(1, len(sampled)), 3),
        sampled_pages=sampled,
        itemized_page_ratio=round(ratio, 3),
        project_rows_per_page=round(per_page, 2),
        est_projects=round(ratio * total_pages * per_page * 0.8),
        has_budgets=any(signals[p]["amounts"] >= 2 for p in itemized_pages),
        has_years=bool(years) or any(year_tokens(texts.get(p, "")) for p in itemized_pages),
        has_descriptions=sum(signals[p]["descriptive"] for p in itemized_pages) >= 2,
        aggregate_only=ratio < 0.05 and summary >= 3,
        org_mentions=mentions,
        fiscal_years=years[:12],
        status_hint=classify_status(first).value,
        id_tokens=sorted(all_ids)[:400],
        name_tokens=sorted(k for k in keys if k)[:400],
        first_page_text=clean_text(first)[:2500],
    )


async def probe_candidate(
    candidate: DocumentCandidate,
    fetcher: Fetcher,
    profile: OrgProfile,
    config: AppConfig,
    cache: FileCache,
) -> list[tuple[str, str, str]]:
    """Probe in place. Returns document links found when the candidate is an HTML hub page."""
    result = await fetcher.fetch_document(candidate.url)
    if not result.ok:
        candidate.probe = ProbeResult(ok=False, error=result.error)
        if result.blocked:
            candidate.notes.append("host blocks automated downloads (bot-protection challenge); no archive copy available")
        return []
    candidate.final_url = result.final_url
    if result.archive_url:
        candidate.archive_url = result.archive_url
        candidate.notes.append(f"content read from Internet Archive capture (host refused automated download): {result.archive_url}")
    org_terms = [profile.canonical_name, *profile.aliases]
    key = sha256_text(f"{PROBE_VERSION}|{result.content_hash}|{'|'.join(org_terms)}")
    cached = cache.get_json("probe", key)
    hub_links: list[tuple[str, str, str]] = []
    if result.is_pdf:
        candidate.file_type = "pdf"
        if cached:
            probe = ProbeResult.model_validate(cached)
        else:
            path = result.path or cache.path("downloads", result.content_hash or "", "")
            loop = asyncio.get_running_loop()
            total = await loop.run_in_executor(None, page_count, path)
            sample = _sample_pages(total, config.discovery.probe_sample_pages)
            texts = await loop.run_in_executor(None, fast_page_texts, path, sample)
            meta = await loop.run_in_executor(None, pdf_metadata, path)
            probe = _profile_texts(texts, total, org_terms)
            probe.pdf_title = clean_text(meta.get("title")) or None
            raw_date = meta.get("creationdate") or meta.get("moddate") or ""
            m = re.search(r"(19|20)\d{2}(\d{2})?(\d{2})?", raw_date)
            if m:
                d = m.group(0)
                probe.pdf_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d[:4]
    elif result.is_html:
        candidate.file_type = "html"
        html = (result.content or b"").decode("utf-8", errors="replace")
        parsed = parse_html(html, candidate.doc_id, result.content_hash or "")
        text = "\n".join(line.text for line in parsed.pages[0].lines)
        probe = ProbeResult.model_validate(cached) if cached else _profile_texts({1: text}, 1, org_terms)
        if parsed.title and (not candidate.title or candidate.title.startswith("http")):
            candidate.title = parsed.title
        # hub page: many document links -> hand them back as new candidates
        soup = BeautifulSoup(html, "lxml")
        for link in soup.find_all("a", href=True):
            absolute = urljoin(result.final_url, str(link["href"]))
            if absolute.startswith("http") and DOC_LINK.search(absolute):
                hub_links.append((absolute, _anchor_text(link), result.final_url))
    else:
        candidate.probe = ProbeResult(ok=False, error=f"unsupported content type {result.content_type!r}")
        return []
    probe.content_hash = result.content_hash
    probe.size_bytes = len(result.content or b"")
    if not cached:
        cache.put_json("probe", key, probe.model_dump(mode="json"))
    candidate.probe = probe
    # deterministic metadata refinement from the cover pages
    cover = f"{candidate.title}\n{probe.pdf_title or ''}\n{probe.first_page_text[:1200]}"
    if candidate.doc_type == DocType.OTHER:
        candidate.doc_type = classify_doc_type(cover)
    if candidate.doc_status == DocStatus.UNKNOWN:
        candidate.doc_status = classify_status(cover)
    if not candidate.coverage_start_year:
        start, end = coverage_from_text(candidate.title, probe.pdf_title or "", probe.first_page_text[:1500])
        candidate.coverage_start_year, candidate.coverage_end_year = start, end
        candidate.document_fiscal_year = start
    if not candidate.publication_date and probe.pdf_date:
        candidate.publication_date = probe.pdf_date
    candidate.contains_itemized_projects = probe.itemized_page_ratio >= 0.05 and not probe.aggregate_only
    return hub_links


# --------------------------------------------------------------------------------------------------
# Batched LLM document profiling
# --------------------------------------------------------------------------------------------------

_TYPE_CODES = {
    "capital_improvement_plan": DocType.CIP_PLAN,
    "capital_improvement_program": DocType.CIP_PROGRAM,
    "multi_year_capital_program": DocType.MULTI_YEAR_PROGRAM,
    "transportation_improvement_program": DocType.TIP,
    "proposed_budget": DocType.PROPOSED_BUDGET,
    "adopted_budget": DocType.ADOPTED_BUDGET,
    "capital_expenditure_budget": DocType.CAPITAL_BUDGET,
    "infrastructure_investment_plan": DocType.INFRASTRUCTURE_PLAN,
    "capital_planning_report": DocType.CAPITAL_REPORT,
    "board_approved_schedule": DocType.BOARD_SCHEDULE,
    "facility_construction_program": DocType.FACILITY_PROGRAM,
    "other": DocType.OTHER,
}
_STATUS_CODES = {"adopted_approved": DocStatus.ADOPTED, "proposed_tentative": DocStatus.PROPOSED, "unknown": DocStatus.UNKNOWN}

PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["documents"],
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "ref",
                    "document_type",
                    "status",
                    "publisher",
                    "publication_date",
                    "coverage_start_fy",
                    "coverage_end_fy",
                    "itemized_projects",
                    "about_requested_organization",
                ],
                "properties": {
                    "ref": {"type": "integer"},
                    "document_type": {"type": "string", "enum": list(_TYPE_CODES)},
                    "status": {"type": "string", "enum": list(_STATUS_CODES)},
                    "publisher": {"type": "string", "description": "Publishing organization as printed; empty if unclear."},
                    "publication_date": {
                        "type": "string",
                        "description": "Publication/adoption date as printed (YYYY-MM-DD, YYYY-MM or YYYY); empty if not stated.",
                    },
                    "coverage_start_fy": {"type": "integer", "description": "First fiscal year covered; 0 if unknown."},
                    "coverage_end_fy": {"type": "integer", "description": "Last fiscal year covered; 0 if unknown."},
                    "itemized_projects": {
                        "type": "boolean",
                        "description": "true if the excerpt shows individual projects (not only aggregate totals).",
                    },
                    "about_requested_organization": {"type": "boolean"},
                },
            },
        }
    },
}

PROFILE_SYSTEM = (
    "You classify public-sector documents from their cover/first pages. Use only the provided text; when a "
    "value is not stated use empty strings or 0. Return JSON only."
)


async def llm_profile(
    candidates: list[DocumentCandidate],
    organization: str,
    config: AppConfig,
    registry: LLMRegistry,
    ledger: CostLedger,
    cache: FileCache,
) -> None:
    """One batched cheap-model call that profiles every probed document (in place)."""
    ref = config.models.profiler
    probed = [c for c in candidates if c.probe and c.probe.ok]
    if not probed or not registry.available(ref.provider):
        return
    blocks = []
    for i, c in enumerate(probed):
        excerpt = (c.probe.first_page_text if c.probe else "")[:1400]
        blocks.append(f"<doc ref={i}>\nTITLE: {c.title}\nURL: {c.final_url or c.url}\nFIRST PAGES:\n{excerpt}\n</doc>")
    user = f"Requested organization: {organization}\n\n" + "\n\n".join(blocks)
    key = sha256_text(f"{ref.model}|profile-v1|{user}")
    data = cache.get_json("profile", key)
    if data is None:
        try:
            response = await registry.get(ref.provider).json_completion(
                model=ref.model,
                system=PROFILE_SYSTEM,
                user=user,
                schema=PROFILE_SCHEMA,
                max_tokens=max(ref.max_tokens, 120 * len(probed) + 400),
                stage="discovery",
                task="document_profiling",
                ledger=ledger,
                schema_name="document_profiles",
            )
            data = response.data if isinstance(response.data, dict) else None
            if data:
                cache.put_json("profile", key, data)
        except LLMError as exc:
            log.warning("document profiling failed: %s", exc)
            return
    else:
        ledger.cache_hit("document_profiling")
    for item in (data or {}).get("documents", []):
        index = item.get("ref")
        if not isinstance(index, int) or not 0 <= index < len(probed):
            continue
        c = probed[index]
        assert c.probe is not None
        c.probe.llm_profile = item
        cover = f"{c.title} {c.probe.pdf_title or ''} {c.probe.first_page_text}"
        doc_type = _TYPE_CODES.get(item.get("document_type", "other"), DocType.OTHER)
        if doc_type != DocType.OTHER:
            c.doc_type = doc_type
        status = _STATUS_CODES.get(item.get("status", "unknown"), DocStatus.UNKNOWN)
        if status != DocStatus.UNKNOWN:
            c.doc_status = status
        if item.get("publisher"):
            c.publisher = clean_text(item["publisher"])
        if item.get("publication_date"):
            c.publication_date = clean_text(item["publication_date"])
        start, end = item.get("coverage_start_fy") or 0, item.get("coverage_end_fy") or 0
        # accept model-reported years only when they are printed in the title/cover (no invented metadata)
        printed = {y for _, y in year_tokens(cover)}
        if start and start in printed:
            c.coverage_start_year = start
            c.document_fiscal_year = start
            if end and end >= start and (end in printed or end - start <= 10):
                c.coverage_end_year = end
        if item.get("about_requested_organization") is False:
            c.notes.append("cover suggests a different organization")
        if item.get("itemized_projects") is True and not c.contains_itemized_projects:
            c.notes.append("model saw itemized projects on the cover pages")


def overlap_estimate(a: DocumentCandidate, b: DocumentCandidate) -> float:
    """Estimated share of projects two documents have in common (0-1)."""
    pa, pb = a.probe, b.probe
    estimates = []
    sa, sb = series_key(a.title), series_key(b.title)
    if sa and sa == sb and a.domain == b.domain:
        same_period = (a.coverage_start_year or 0) == (b.coverage_start_year or 0)
        estimates.append(0.95 if same_period else 0.8)
    if pa and pb:
        if len(pa.id_tokens) >= 5 and len(pb.id_tokens) >= 5:
            ia, ib = set(pa.id_tokens), set(pb.id_tokens)
            estimates.append(len(ia & ib) / max(1, min(len(ia), len(ib))))
        if len(pa.name_tokens) >= 5 and len(pb.name_tokens) >= 5:
            na, nb = set(pa.name_tokens), set(pb.name_tokens)
            estimates.append(0.8 * len(na & nb) / max(1, min(len(na), len(nb))))
        if pa.content_hash and pa.content_hash == pb.content_hash:
            estimates.append(1.0)
    return round(max(estimates, default=0.0), 3)


def mark_mirrors(candidates: list[DocumentCandidate], organization: str, aliases: list[str]) -> None:
    """Flag copies of the organization's own documents hosted on other domains (spec §3: prefer official
    sources — a verified copy of an official document ranks below the official host, above third parties)."""
    from rapidfuzz import fuzz

    names = [n for n in [organization, *aliases] if n and len(n) >= 3]
    for c in candidates:
        if c.is_official_domain or not c.probe or not c.probe.ok:
            continue
        publisher = (c.probe.llm_profile or {}).get("publisher") or ""
        publisher_match = bool(publisher) and any(fuzz.token_set_ratio(publisher.lower(), n.lower()) >= 90 for n in names)
        about_org = (c.probe.llm_profile or {}).get("about_requested_organization")
        if publisher_match or (about_org is not False and c.probe.org_mentions >= 2):
            c.is_mirror_of_official = True
            c.notes.append(f"copy of an official {organization} document hosted on {c.domain}")
