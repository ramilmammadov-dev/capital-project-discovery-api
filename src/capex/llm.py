"""Generative-model providers behind one small interface.

* ``AnthropicClient`` — Claude via the Messages API with strict JSON-schema output
  (``output_config.format``) and the server-side ``web_search`` tool for LLM-assisted discovery.
* ``OpenAICompatibleClient`` — any ``/chat/completions`` endpoint with JSON-schema / JSON mode:
  OpenAI, GLM (Zhipu), DeepSeek, Together, vLLM, Ollama…

Every request — success or failure — is written to the job's ``CostLedger`` with tokens, price,
duration and retry count.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from capex.config import AppConfig
from capex.costs import CostLedger
from capex.models import SearchHit

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMUnavailableError(LLMError):
    """Provider is not configured (missing API key)."""


@dataclass
class LLMResponse:
    data: Any | None
    text: str
    model: str
    stop_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    duration_ms: int = 0
    retries: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.stop_reason in ("max_tokens", "length")


def _backoff(attempt: int, retry_after: float | None = None) -> float:
    if retry_after:
        return min(retry_after, 60.0)
    return min(30.0, (2**attempt) * 0.75) * (0.75 + random.random() / 2)


def _parse_json(text: str) -> Any | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1 :] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


class LLMClient(ABC):
    provider: str = "abstract"

    @abstractmethod
    async def json_completion(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
        stage: str,
        task: str,
        ledger: CostLedger,
        schema_name: str = "output",
    ) -> LLMResponse: ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


# --------------------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str, timeout: float = 240.0, max_retries: int = 4) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self.max_retries = max_retries

    def _retryable(self, exc: Exception) -> tuple[bool, float | None]:
        a = self._anthropic
        if isinstance(exc, a.RateLimitError | a.APIConnectionError | a.APITimeoutError | a.InternalServerError):
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    retry_after = float(response.headers.get("retry-after", "") or 0) or None
                except ValueError:
                    retry_after = None
            return True, retry_after
        if isinstance(exc, a.APIStatusError) and exc.status_code in (408, 409, 429, 500, 502, 503, 504, 529):
            return True, None
        return False, None

    async def _create(self, stage: str, task: str, model: str, ledger: CostLedger, **kwargs: Any) -> tuple[Any, int, int]:
        """messages.create with retry; returns (message, retries, duration_ms). Failures are recorded."""
        attempt = 0
        started = time.perf_counter()
        while True:
            try:
                message = await self.client.messages.create(model=model, **kwargs)
                return message, attempt, int((time.perf_counter() - started) * 1000)
            except Exception as exc:
                retryable, retry_after = self._retryable(exc)
                if retryable and attempt < self.max_retries:
                    attempt += 1
                    delay = _backoff(attempt, retry_after)
                    log.warning("anthropic %s retry %d in %.1fs: %s", task, attempt, delay, exc)
                    await asyncio.sleep(delay)
                    continue
                ledger.record(
                    stage=stage,
                    task=task,
                    provider=self.provider,
                    model=model,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    retries=attempt,
                    success=False,
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
                raise LLMError(f"{task} failed: {exc}") from exc

    async def json_completion(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
        stage: str,
        task: str,
        ledger: CostLedger,
        schema_name: str = "output",
    ) -> LLMResponse:
        message, retries, duration_ms = await self._create(
            stage,
            task,
            model,
            ledger,
            max_tokens=max_tokens,
            # The (long, static) instructions are cached across chunks of the same job.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        usage = message.usage
        text = "".join(getattr(block, "text", "") for block in message.content if block.type == "text")
        response = LLMResponse(
            data=_parse_json(text) if message.stop_reason != "max_tokens" else None,
            text=text,
            model=message.model or model,
            stop_reason=message.stop_reason,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            duration_ms=duration_ms,
            retries=retries,
        )
        ledger.record(
            stage=stage,
            task=task,
            provider=self.provider,
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
            duration_ms=duration_ms,
            retries=retries,
            success=response.data is not None,
            error=None if response.data is not None else f"unparseable output (stop_reason={message.stop_reason})",
        )
        return response

    async def web_search(
        self,
        *,
        model: str,
        queries: list[str],
        tool_version: str,
        price_per_search: float,
        stage: str,
        ledger: CostLedger,
        region: str | None = None,
    ) -> list[SearchHit]:
        """LLM-assisted search: Claude runs the planned queries with the server-side web_search tool.

        Result URLs/titles are harvested from the ``web_search_tool_result`` blocks; the model's
        prose is ignored, so ranking stays deterministic and auditable.
        """
        if not queries:
            return []
        tool: dict[str, Any] = {"type": tool_version, "name": "web_search", "max_uses": len(queries)}
        if region:
            tool["user_location"] = {"type": "approximate", "country": "US", "region": region}
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Run each of the following web searches with the web_search tool, exactly once each and "
                    "in order. Use the query text as given. Do not summarise the results; when all searches "
                    f"are done reply with the single word DONE.\n\n{numbered}"
                ),
            }
        ]
        hits: list[SearchHit] = []
        query_by_id: dict[str, str] = {}
        for _ in range(4):  # initial call + pause_turn continuations
            message, retries, duration_ms = await self._create(
                stage, "web_search", model, ledger, max_tokens=1024, messages=messages, tools=[tool]
            )
            usage = message.usage
            searches = 0
            server_usage = getattr(usage, "server_tool_use", None)
            if server_usage is not None:
                searches = getattr(server_usage, "web_search_requests", 0) or 0
            ledger.record(
                stage=stage,
                task="web_search",
                provider=self.provider,
                model=model,
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                search_requests=searches,
                request_fee_usd=searches * price_per_search,
                duration_ms=duration_ms,
                retries=retries,
            )
            for block in message.content:
                if block.type == "server_tool_use":
                    query_by_id[block.id] = (getattr(block, "input", {}) or {}).get("query", "")
                elif block.type == "web_search_tool_result":
                    content = block.content
                    if not isinstance(content, list):
                        log.warning("web_search error: %s", getattr(content, "error_code", content))
                        continue
                    query = query_by_id.get(block.tool_use_id, "")
                    for rank, result in enumerate(content, start=1):
                        hits.append(
                            SearchHit(
                                url=result.url,
                                title=result.title or "",
                                query=query,
                                provider="anthropic",
                                rank=rank,
                                page_age=getattr(result, "page_age", None),
                            )
                        )
            if message.stop_reason != "pause_turn":
                break
            messages = [
                messages[0],
                {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in message.content]},
            ]
        return hits

    async def aclose(self) -> None:
        await self.client.close()


# --------------------------------------------------------------------------------------------------
# OpenAI-compatible (OpenAI, GLM, DeepSeek, vLLM, ...)
# --------------------------------------------------------------------------------------------------


class OpenAICompatibleClient(LLMClient):
    provider = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        response_format: str = "json_schema",
        timeout: float = 240.0,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.response_format = response_format
        self.max_retries = max_retries
        self.http = httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    async def json_completion(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
        stage: str,
        task: str,
        ledger: CostLedger,
        schema_name: str = "output",
    ) -> LLMResponse:
        if self.response_format == "json_schema":
            fmt: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        else:
            fmt = {"type": "json_object"}
            system = f"{system}\n\nReturn only JSON matching this schema:\n{json.dumps(schema)}"
        token_field = "max_completion_tokens" if "api.openai.com" in self.base_url else "max_tokens"
        body = {
            "model": model,
            token_field: max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": fmt,
        }
        attempt = 0
        started = time.perf_counter()
        while True:
            try:
                resp = await self.http.post(f"{self.base_url}/chat/completions", json=body)
                if resp.status_code in (408, 409, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                    attempt += 1
                    retry_after = float(resp.headers.get("retry-after", "0") or 0) or None
                    await asyncio.sleep(_backoff(attempt, retry_after))
                    continue
                resp.raise_for_status()
                payload = resp.json()
                break
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                transient = isinstance(exc, httpx.TransportError)
                if transient and attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(_backoff(attempt))
                    continue
                ledger.record(
                    stage=stage,
                    task=task,
                    provider=self.provider,
                    model=model,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    retries=attempt,
                    success=False,
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
                raise LLMError(f"{task} failed: {exc}") from exc
        duration_ms = int((time.perf_counter() - started) * 1000)
        choice = (payload.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        reported = bool(usage)
        input_tokens = int(usage.get("prompt_tokens") or 0)
        cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        if not reported:  # estimate from characters and flag it (spec §12)
            input_tokens = (len(system) + len(user)) // 4
            output_tokens = len(text) // 4
        stop = "max_tokens" if choice.get("finish_reason") == "length" else choice.get("finish_reason")
        response = LLMResponse(
            data=_parse_json(text) if stop != "max_tokens" else None,
            text=text,
            model=payload.get("model") or model,
            stop_reason=stop,
            input_tokens=input_tokens - cached,
            output_tokens=output_tokens,
            cache_read_tokens=cached,
            duration_ms=duration_ms,
            retries=attempt,
        )
        ledger.record(
            stage=stage,
            task=task,
            provider=self.provider,
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached,
            duration_ms=duration_ms,
            retries=attempt,
            success=response.data is not None,
            usage_reported=reported,
            error=None if response.data is not None else "unparseable output",
        )
        return response

    async def aclose(self) -> None:
        await self.http.aclose()


# --------------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------------


class LLMRegistry:
    """Lazily constructs provider clients from config + environment."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._clients: dict[str, LLMClient] = {}

    def available(self, provider: str) -> bool:
        if provider in self._clients:
            return True
        p = self.config.providers
        if provider == "anthropic":
            return bool(self.config.secret(p.anthropic_api_key_env))
        if provider == "openai_compatible":
            return bool(self.config.secret(p.openai_api_key_env))
        return False

    def get(self, provider: str) -> LLMClient:
        if provider in self._clients:
            return self._clients[provider]
        p = self.config.providers
        timeout = self.config.limits.llm_timeout_s
        if provider == "anthropic":
            key = self.config.secret(p.anthropic_api_key_env)
            if not key:
                raise LLMUnavailableError(f"{p.anthropic_api_key_env} is not set")
            client: LLMClient = AnthropicClient(key, timeout=timeout)
        elif provider == "openai_compatible":
            key = self.config.secret(p.openai_api_key_env)
            if not key:
                raise LLMUnavailableError(f"{p.openai_api_key_env} is not set")
            client = OpenAICompatibleClient(key, p.openai_base_url, p.openai_response_format, timeout=timeout)
        else:
            raise LLMUnavailableError(f"unknown provider {provider}")
        self._clients[provider] = client
        return client

    def register(self, provider: str, client: LLMClient) -> None:
        """Inject a client (used by tests and offline replay)."""
        self._clients[provider] = client

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
