"""Chunk screening: Jev (primary), heuristic (free fallback) and pass-through (evaluation)."""

from __future__ import annotations

from capex.config import AppConfig
from capex.costs import CostLedger
from capex.screening.base import Screener
from capex.screening.heuristic import HeuristicScreener, PassThroughScreener
from capex.storage.cache import FileCache


def build_screener(
    config: AppConfig, ledger: CostLedger, cache: FileCache, mode: str | None = None
) -> tuple[Screener, list[str]]:
    """Return the screener for ``mode`` (default: config) plus any warnings about fallbacks."""
    mode = mode or config.screening.provider
    warnings: list[str] = []
    if mode == "jev":
        key = config.secret(config.providers.typesafe_api_key_env)
        if key:
            from capex.screening.jev import JevScreener

            return JevScreener(config, key, ledger, cache), warnings
        warnings.append(
            f"Jev screening unavailable ({config.providers.typesafe_api_key_env} not set); "
            f"using the '{config.screening.fallback}' fallback screener"
        )
        mode = config.screening.fallback
    if mode == "heuristic":
        return HeuristicScreener(config.screening), warnings
    return PassThroughScreener(), warnings


__all__ = ["HeuristicScreener", "PassThroughScreener", "Screener", "build_screener"]
