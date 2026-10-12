"""Shared fixtures. Every test is hermetic: no network, no real API keys, data under tmp_path."""

from __future__ import annotations

import pytest

from capex.config import AppConfig

_SECRET_ENV = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "TYPESAFE_API_KEY",
    "BRAVE_API_KEY",
    "SERPER_API_KEY",
    "CAPEX_API_KEYS",
    "CAPEX_DATA_DIR",
    "CAPEX_DATABASE_URL",
    "CAPEX_CONFIG",
    "CAPEX_LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch):
    for name in _SECRET_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def app_config() -> AppConfig:
    """Default pipeline configuration (screening heuristic so nothing needs a key)."""
    config = AppConfig()
    config.screening.provider = "heuristic"
    return config
