"""Usage & cost accounting (spec §12 — every model request is measured).

Each provider adapter reports raw usage; the ledger prices it from the configured price table and
keeps per-request records, which roll up into per-stage / per-model totals for the job metrics,
the API and the Excel "Extraction & Cost Metrics" sheet.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from capex.config import AppConfig, Price
from capex.models import UsageRecord


class BudgetExceededError(RuntimeError):
    """Raised when a job's accumulated cost passes ``limits.max_job_cost_usd``."""


class CostLedger:
    def __init__(
        self,
        config: AppConfig,
        max_cost_usd: float | None = None,
        on_record: Callable[[UsageRecord], None] | None = None,
    ) -> None:
        self.config = config
        self.max_cost_usd = max_cost_usd
        self.on_record = on_record
        self.records: list[UsageRecord] = []
        self.cache_hits: dict[str, int] = defaultdict(int)

    # -- pricing ------------------------------------------------------------------------------

    def _price(self, model: str) -> Price | None:
        return self.config.price_for(model)

    def price_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> tuple[float, float | None, float | None, bool]:
        """Return (cost, input_price, output_price, estimated)."""
        price = self._price(model)
        if price is None:
            return 0.0, None, None, True
        prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
        in_price, out_price = price.input, price.output
        cache_read = price.cache_read if price.cache_read is not None else price.input
        cache_write = price.cache_write if price.cache_write is not None else price.input
        if price.long_context_threshold and prompt_tokens > price.long_context_threshold:
            scale = (price.long_input or in_price) / in_price if in_price else 1.0
            in_price = price.long_input or in_price
            out_price = price.long_output or out_price
            cache_read *= scale
            cache_write *= scale
        cost = (
            input_tokens * in_price
            + output_tokens * out_price
            + cache_read_tokens * cache_read
            + cache_write_tokens * cache_write
        ) / 1_000_000 + price.per_request
        return cost, in_price, out_price, False

    # -- recording ----------------------------------------------------------------------------

    def record(
        self,
        *,
        stage: str,
        task: str,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        search_requests: int = 0,
        request_fee_usd: float = 0.0,
        duration_ms: int = 0,
        retries: int = 0,
        success: bool = True,
        error: str | None = None,
        usage_reported: bool = True,
    ) -> UsageRecord:
        cost, in_price, out_price, estimated = self.price_request(
            model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
        )
        record = UsageRecord(
            stage=stage,
            task=task,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            search_requests=search_requests,
            input_price_per_mtok=in_price,
            output_price_per_mtok=out_price,
            request_fee_usd=request_fee_usd,
            cost_usd=round(cost + request_fee_usd, 8),
            duration_ms=duration_ms,
            retries=retries,
            success=success,
            error=error,
            # spec: flag estimated (not provider-reported) usage/cost explicitly
            estimated=estimated or not usage_reported,
        )
        self.records.append(record)
        if self.on_record:
            self.on_record(record)
        return record

    def cache_hit(self, task: str) -> None:
        self.cache_hits[task] += 1

    def check_budget(self) -> None:
        if self.max_cost_usd is not None and self.total_cost > self.max_cost_usd:
            raise BudgetExceededError(
                f"job cost ${self.total_cost:.4f} exceeded the configured cap ${self.max_cost_usd:.2f}"
            )

    # -- aggregation --------------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 6)

    @property
    def has_estimates(self) -> bool:
        return any(r.estimated for r in self.records)

    def cost_by_stage(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for r in self.records:
            totals[r.stage] += r.cost_usd
        return {k: round(v, 6) for k, v in sorted(totals.items())}

    def usage_by_model(self) -> list[dict[str, Any]]:
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for r in self.records:
            key = (r.provider, r.model)
            row = rows.setdefault(
                key,
                {
                    "provider": r.provider,
                    "model": r.model,
                    "requests": 0,
                    "failed_requests": 0,
                    "retries": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "search_requests": 0,
                    "input_price_per_mtok": r.input_price_per_mtok,
                    "output_price_per_mtok": r.output_price_per_mtok,
                    "cost_usd": 0.0,
                    "estimated": False,
                    "tasks": set(),
                },
            )
            row["requests"] += 1
            row["failed_requests"] += 0 if r.success else 1
            row["retries"] += r.retries
            row["input_tokens"] += r.input_tokens
            row["output_tokens"] += r.output_tokens
            row["cache_read_tokens"] += r.cache_read_tokens
            row["cache_write_tokens"] += r.cache_write_tokens
            row["search_requests"] += r.search_requests
            row["cost_usd"] += r.cost_usd
            row["estimated"] = row["estimated"] or r.estimated
            row["tasks"].add(r.task)
        out = []
        for row in rows.values():
            row["cost_usd"] = round(row["cost_usd"], 6)
            row["tasks"] = ", ".join(sorted(row["tasks"]))
            out.append(row)
        return sorted(out, key=lambda r: -r["cost_usd"])
