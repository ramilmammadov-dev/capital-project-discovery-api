"""Runtime configuration.

Two layers:

* ``Settings`` — process/environment level (data directory, database URL, service API keys).
* ``AppConfig`` — pipeline behaviour (providers, model selection, pricing, thresholds, limits),
  loaded from ``config/capex.yaml`` (or ``$CAPEX_CONFIG``). Every field has a safe default, so the
  YAML file only needs the values you want to change.

Provider secrets are never stored in YAML: the config names the environment variable to read.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["anthropic", "openai_compatible"]


class ModelRef(BaseModel):
    provider: ProviderName = "anthropic"
    model: str
    max_tokens: int = 8000


class Price(BaseModel):
    """USD per one million tokens (plus an optional flat fee per request)."""

    input: float
    output: float = 0.0
    cache_write: float | None = None
    cache_read: float | None = None
    # Tiered pricing (e.g. Claude Haiku 5.5 is cheaper for prompts up to 100k tokens).
    long_context_threshold: int | None = None
    long_input: float | None = None
    long_output: float | None = None
    per_request: float = 0.0


class ProvidersConfig(BaseModel):
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    openai_api_key_env: str = "OPENAI_API_KEY"
    # Any OpenAI-compatible endpoint: OpenAI, GLM (https://open.bigmodel.cn/api/paas/v4), DeepSeek, vLLM...
    openai_base_url: str = "https://api.openai.com/v1"
    openai_response_format: Literal["json_schema", "json_object"] = "json_schema"
    typesafe_api_key_env: str = "TYPESAFE_API_KEY"
    typesafe_base_url: str | None = None
    brave_api_key_env: str = "BRAVE_API_KEY"
    serper_api_key_env: str = "SERPER_API_KEY"


class ModelsConfig(BaseModel):
    # Query planning + document metadata profiling (small structured outputs)
    planner: ModelRef = ModelRef(model="claude-haiku-5-5", max_tokens=2500)
    profiler: ModelRef = ModelRef(model="claude-haiku-5-5", max_tokens=1500)
    # Low-cost project extraction (default) and escalation for difficult chunks
    extraction: ModelRef = ModelRef(model="claude-haiku-5-5", max_tokens=16000)
    escalation: ModelRef = ModelRef(model="claude-sonnet-5-5", max_tokens=16000)


class SearchConfig(BaseModel):
    # Providers are used in this order; every configured one contributes results.
    # "anthropic" = LLM-assisted search (Claude + server-side web_search tool).
    providers: list[Literal["anthropic", "brave", "serper", "duckduckgo"]] = [
        "anthropic",
        "brave",
        "serper",
        "duckduckgo",
    ]
    # DuckDuckGo (keyless) is only used when no paid provider is configured.
    duckduckgo_only_as_fallback: bool = True
    max_queries: int = 10
    results_per_query: int = 10
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_tool_version: str = "web_search_20250305"
    price_per_search_usd: dict[str, float] = {
        "anthropic": 0.01,
        "brave": 0.005,
        "serper": 0.001,
        "duckduckgo": 0.0,
    }
    cache_ttl_hours: int = 168


class DiscoveryConfig(BaseModel):
    max_candidates: int = 80
    max_probe: int = 14
    crawl_max_pages: int = 40
    crawl_max_depth: int = 2
    max_document_mb: int = 200
    probe_sample_pages: int = 14
    select_top: int = 3
    # When an official host refuses automated downloads (bot challenge / 403 / 404), read the latest
    # Internet Archive capture of the same URL instead (provenance is recorded on the document).
    archive_fallback: bool = True
    # Document scoring weights (see discovery/ranking.py for the meaning of each feature)
    weights: dict[str, float] = {
        "itemization": 0.30,
        "volume": 0.15,
        "authority": 0.12,
        "recency": 0.15,
        "detail": 0.12,
        "doc_type": 0.08,
        "relevance": 0.05,
        "readability": 0.03,
    }
    # Penalty applied to a candidate's score per unit of estimated overlap with already-selected docs
    overlap_penalty: float = 0.9


class ParsingConfig(BaseModel):
    max_pages: int = 1500
    workers: int = 6
    ocr_enabled: bool = True  # local tesseract only (no paid OCR)
    ocr_max_pages: int = 60


class ChunkingConfig(BaseModel):
    target_tokens: int = 1800
    max_tokens: int = 3200
    min_tokens: int = 200
    max_rows_per_chunk: int = 28
    overlap_lines: int = 3


class ScreeningConfig(BaseModel):
    provider: Literal["jev", "heuristic", "none"] = "jev"
    # Used automatically when the primary provider is not configured or fails.
    fallback: Literal["heuristic", "none"] = "heuristic"
    jev_model: str = "jev-1.13.0"
    concurrency: int = 8
    max_state_chars: int = 24000
    # Routing thresholds on Jev probabilities (high-recall by design):
    #   p_project = P(explicit) + P(possible)
    extract_threshold: float = 0.50
    plausible_threshold: float = 0.15
    context_threshold: float = 0.35
    # Deterministic recall guard: never skip a chunk with strong project-table signals.
    recall_guard: bool = True


class ExtractionConfig(BaseModel):
    concurrency: int = 8
    max_split_depth: int = 2
    # Escalate a chunk to the stronger model when, after splitting, the cheap model still
    # (a) fails schema validation, (b) has more than this share of records not grounded in the text,
    # or (c) returns fewer than this share of the deterministically estimated table rows.
    escalate_unverified_ratio: float = 0.34
    escalate_row_recall_ratio: float = 0.6
    max_escalations_per_job: int = 8
    review_confidence_below: float = 0.6
    excerpt_match_threshold: int = 85
    name_match_threshold: int = 80


class DedupConfig(BaseModel):
    id_name_min: int = 55
    name_auto_merge: int = 93
    name_corroborated_merge: int = 85
    name_possible: int = 72
    location_min: int = 70
    budget_tolerance: float = 0.25


class LimitsConfig(BaseModel):
    job_timeout_s: int = 5400
    http_timeout_s: float = 60.0
    llm_timeout_s: float = 240.0
    max_job_cost_usd: float = 5.0
    max_concurrent_jobs: int = 2
    job_max_attempts: int = 2
    job_reuse_hours: int = 72


DEFAULT_PRICING: dict[str, Price] = {
    # Claude list prices (USD / MTok), October 2026
    "claude-haiku-5-5": Price(
        input=0.10,
        output=0.50,
        cache_write=0.125,
        cache_read=0.01,
        long_context_threshold=100_000,
        long_input=0.50,
        long_output=2.50,
    ),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_write=2.50, cache_read=0.10),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_write=5.00, cache_read=0.20),
    # Jev (TypeSafe): input tokens only, output is free
    "jev-1.13.0": Price(input=0.042, output=0.0),
    "jev-latest": Price(input=0.042, output=0.0),
}


class AppConfig(BaseModel):
    user_agent: str = (
        "Mozilla/5.0 (compatible; CapexDiscovery/1.0; public capital-plan research; +https://github.com/)"
    )
    providers: ProvidersConfig = ProvidersConfig()
    models: ModelsConfig = ModelsConfig()
    pricing: dict[str, Price] = Field(default_factory=lambda: dict(DEFAULT_PRICING))
    search: SearchConfig = SearchConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    parsing: ParsingConfig = ParsingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    screening: ScreeningConfig = ScreeningConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    dedup: DedupConfig = DedupConfig()
    limits: LimitsConfig = LimitsConfig()

    def price_for(self, model: str) -> Price | None:
        return self.pricing.get(model)

    def secret(self, env_name: str) -> str | None:
        value = os.environ.get(env_name, "").strip()
        return value or None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAPEX_", extra="ignore")

    data_dir: Path = Path("data")
    database_url: str | None = None
    config: Path | None = None
    api_keys: str = ""  # comma-separated
    log_level: str = "INFO"

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'capex.db').as_posix()}"

    @property
    def accepted_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load AppConfig from YAML, layered over the built-in defaults."""
    candidates = [path, Path(os.environ["CAPEX_CONFIG"]) if os.environ.get("CAPEX_CONFIG") else None]
    candidates.append(Path("config/capex.yaml"))
    for candidate in candidates:
        if candidate and candidate.is_file():
            raw = yaml.safe_load(candidate.read_text()) or {}
            base = AppConfig().model_dump()
            if "pricing" in raw:
                # pricing entries replace defaults per model, they are not deep-merged field by field
                base["pricing"].update(raw.pop("pricing") or {})
            return AppConfig.model_validate(_deep_merge(base, raw))
    return AppConfig()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(override=False)
    return Settings()


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    load_dotenv(override=False)
    return load_app_config(get_settings().config)
