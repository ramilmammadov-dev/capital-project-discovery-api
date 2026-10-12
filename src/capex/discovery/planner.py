"""Organization profiling & search planning.

One small structured LLM call turns the organization name into: canonical name, organization type,
state, likely official domains, aliases, the organization's own names for its capital-planning
documents ("Five-Year Transportation Facilities Construction Program", "Capital Improvement Program",
"Transportation Improvement Program", ...) and targeted search queries — spec §3 "search intelligently
using organization-specific terminology and document titles". Template queries are always added as a
safety net, and are used alone when no LLM is configured.
"""

from __future__ import annotations

import logging
from datetime import date

from capex.config import AppConfig
from capex.costs import CostLedger
from capex.discovery.classify import current_fiscal_year
from capex.llm import LLMError, LLMRegistry
from capex.models import OrgProfile
from capex.normalize import clean_text, sha256_text
from capex.storage.cache import FileCache

log = logging.getLogger(__name__)

PLANNER_VERSION = "planner-v2"

ORG_TYPES = [
    "state_agency",
    "state_dot",
    "city",
    "town",
    "county",
    "school_district",
    "college_university",
    "hospital_district",
    "airport_authority",
    "transit_agency",
    "utility_district",
    "port_authority",
    "special_district",
    "regional_planning_org",
    "other",
]

PLANNER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["canonical_name", "org_type", "state", "official_domains", "aliases", "document_terms", "queries"],
    "properties": {
        "canonical_name": {"type": "string"},
        "org_type": {"type": "string", "enum": ORG_TYPES},
        "state": {"type": "string", "description": "Two-letter US state code, or empty."},
        "official_domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Registered domains of the organization's official websites, e.g. 'azdot.gov'.",
        },
        "aliases": {"type": "array", "items": {"type": "string"}, "description": "Abbreviations / other names."},
        "document_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "This organization's own titles for documents that list individual capital projects.",
        },
        "queries": {"type": "array", "items": {"type": "string"}, "description": "8-10 web search queries."},
    },
}

PLANNER_SYSTEM = """\
You plan web searches that find official public documents listing an organization's individual capital \
projects: capital improvement plans/programs (CIP), five/six/ten-year capital or construction programs, \
transportation improvement programs (TIP/STIP), capital budgets, adopted/proposed budgets with capital \
project sections, bond/facility programs and board-approved project schedules.
Return JSON only. Use the organization's own document titles and terminology. Prefer official sources: \
include several queries restricted with site:<official domain>. Include current and next fiscal years. \
Do not invent domains you are unsure of; list only domains you are confident are official."""


def template_queries(name: str, fiscal_year: int, domains: list[str] | None = None) -> list[str]:
    fy, nxt = fiscal_year, fiscal_year + 1
    queries = [
        f'"{name}" capital improvement program {fy}',
        f'"{name}" capital improvement plan pdf',
        f'"{name}" five-year capital program {fy}-{fy + 4}',
        f'"{name}" adopted budget {fy} capital projects',
        f'"{name}" proposed budget {nxt} capital improvement',
    ]
    lowered = name.lower()
    if any(t in lowered for t in ("transportation", "dot", "county", "city", "metro", "regional")):
        queries.append(f'"{name}" transportation improvement program')
    for domain in (domains or [])[:2]:
        queries.append(f"site:{domain} capital improvement program pdf")
    return queries


async def plan_organization(
    name: str,
    config: AppConfig,
    registry: LLMRegistry,
    ledger: CostLedger,
    cache: FileCache,
    today: date | None = None,
) -> OrgProfile:
    name = clean_text(name)
    fy = current_fiscal_year(today)
    profile = OrgProfile(input_name=name, canonical_name=name)
    ref = config.models.planner
    if registry.available(ref.provider):
        key = sha256_text(f"{PLANNER_VERSION}|{ref.model}|{name.lower()}|{fy}")
        data = cache.get_json("planner", key)
        if data is None:
            try:
                response = await registry.get(ref.provider).json_completion(
                    model=ref.model,
                    system=PLANNER_SYSTEM,
                    user=(
                        f"Organization: {name}\nToday: {(today or date.today()).isoformat()}. "
                        f"Current fiscal year for most US agencies (July-June): FY{fy}.\n"
                        "Return the profile and 8-10 queries."
                    ),
                    schema=PLANNER_SCHEMA,
                    max_tokens=ref.max_tokens,
                    stage="discovery",
                    task="query_planning",
                    ledger=ledger,
                    schema_name="org_search_plan",
                )
                data = response.data if isinstance(response.data, dict) else None
                if data:
                    cache.put_json("planner", key, data)
            except LLMError as exc:
                log.warning("planner failed, using templates: %s", exc)
                data = None
        else:
            ledger.cache_hit("query_planning")
        if data:
            domains = [d.lower().removeprefix("www.").strip("/") for d in data.get("official_domains", []) if d]
            profile = OrgProfile(
                input_name=name,
                canonical_name=clean_text(data.get("canonical_name")) or name,
                org_type=data.get("org_type") or "other",
                state=(data.get("state") or "").upper()[:2] or None,
                official_domains=domains[:4],
                aliases=[clean_text(a) for a in data.get("aliases", []) if a][:6],
                document_terms=[clean_text(t) for t in data.get("document_terms", []) if t][:8],
                queries=[clean_text(q) for q in data.get("queries", []) if q],
                planner=ref.model,
            )

    queries: list[str] = []
    planned = profile.queries[: max(1, config.search.max_queries - 3)]  # keep room for template safety net
    for query in [*planned, *template_queries(profile.canonical_name, fy, profile.official_domains)]:
        if query and query.lower() not in {q.lower() for q in queries}:
            queries.append(query)
    profile.queries = queries[: config.search.max_queries]
    return profile
