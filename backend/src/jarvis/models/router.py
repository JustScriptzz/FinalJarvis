"""Hot-swappable model router.

Responsibilities:
  * hold one provider instance per configured model endpoint;
  * choose candidates (manual or smart routing) with capability filtering;
  * retry, fall back, and trip per-provider circuit breakers;
  * normalize streaming deltas and aggregate tool calls;
  * account usage, latency and cost;
  * persist dashboard model switches without restarting.

In-flight requests keep a reference to the provider they started on. A hot
reload retires old providers after a grace period instead of closing them
under a running stream.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from ..core.config import LLMProviderConfig, ModelsConfig
from ..core.errors import (
    CapabilityError,
    CircuitOpenError,
    ConfigError,
    NoProviderAvailableError,
    ProviderError,
)
from ..core.logging import get_logger
from ..core.runtime_store import RuntimeSelectionStore
from .base import (
    BaseProvider,
    LLMProvider,
    ProviderKind,
    STTProvider,
    TTSProvider,
    elapsed_ms,
)
from .circuit import CircuitBreaker
from .metrics import RouterMetrics, llm_cost, stt_cost, tts_cost
from .registry import build_providers
from .routing import (
    ComplexityAssessment,
    assess_complexity,
    profile_chain,
    provider_capabilities,
    required_capabilities,
    supports,
)
from .streaming import StreamAggregator
from .types import (
    ChatRequest,
    ChatResponse,
    ErrorInfo,
    HealthStatus,
    ModelDescriptor,
    RoutingInfo,
    SpeechRequest,
    SpeechResult,
    StreamEvent,
    TranscriptionRequest,
    TranscriptionResult,
)

log = get_logger(__name__)

_RETIRE_GRACE_S = 30.0


class ModelRouter:
    def __init__(
        self,
        config: ModelsConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        runtime_store: RuntimeSelectionStore | None = None,
        metrics: RouterMetrics | None = None,
    ) -> None:
        self._config = config
        self._http = http_client or httpx.AsyncClient(
            follow_redirects=True,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._owns_http = http_client is None
        self._store = runtime_store
        self._providers = build_providers(
            config.llm.providers,
            config.stt.providers,
            config.tts.providers,
            self._http,
        )
        self._selection: dict[ProviderKind, str] = {
            ProviderKind.LLM: config.llm.active,
            ProviderKind.STT: config.stt.active,
            ProviderKind.TTS: config.tts.active,
        }
        self._apply_runtime_selection()
        self._breakers: dict[str, CircuitBreaker] = self._make_breakers()
        self._metrics = metrics or RouterMetrics()
        self._lock = asyncio.Lock()
        self._retired: list[BaseProvider] = []
        self._retire_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> ModelRouter:
        return self

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in list(self._retire_tasks):
            task.cancel()
        if self._retire_tasks:
            await asyncio.gather(*self._retire_tasks, return_exceptions=True)
        providers = [
            provider for group in self._providers.values() for provider in group.values()
        ] + self._retired
        await asyncio.gather(*(provider.aclose() for provider in providers), return_exceptions=True)
        if self._owns_http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Introspection / hot swap
    # ------------------------------------------------------------------
    @property
    def config(self) -> ModelsConfig:
        return self._config

    @property
    def selection(self) -> dict[str, str]:
        return {kind.value: provider_id for kind, provider_id in self._selection.items()}

    def available(self, kind: ProviderKind | str) -> list[ModelDescriptor]:
        return [provider.descriptor() for provider in self._providers[ProviderKind(kind)].values()]

    def provider_config(self, kind: ProviderKind | str, provider_id: str) -> Any:
        """The raw config of one provider, for read-only introspection."""
        provider = self._providers[ProviderKind(kind)].get(provider_id)
        return provider.config if provider is not None else None

    async def list_voices(self, kind: ProviderKind | str, provider_id: str) -> list[str]:
        """Ask one provider for the voices it can actually speak with.

        Providers that can enumerate (ElevenLabs, Cartesia) query their own API;
        the rest fall back to the configured list. A failure is reported as an
        empty list rather than raised, because the voice picker must still open.
        """
        provider = self._providers[ProviderKind(kind)].get(provider_id)
        if provider is None or not isinstance(provider, TTSProvider):
            return []
        try:
            return await asyncio.wait_for(provider.list_voices(), timeout=10.0)
        except Exception as exc:
            log.warning(
                "voice listing failed",
                extra={"provider": provider_id, "error": str(exc)},
            )
            return []

    def describe(self) -> dict[str, Any]:
        routing = self._config.llm.routing
        return {
            "routing": {
                "mode": routing.mode,
                "profiles": routing.profiles,
                "fallback": routing.fallback,
                "fast_threshold": routing.fast_threshold,
                "escalate_threshold": routing.escalate_threshold,
            },
            "active": self.selection,
            "providers": {
                kind.value: [provider.descriptor().model_dump() for provider in group.values()]
                for kind, group in self._providers.items()
            },
            "circuit_breakers": {
                key: breaker.snapshot() for key, breaker in self._breakers.items()
            },
            "metrics": self._metrics.snapshot(),
        }

    async def switch(
        self, kind: ProviderKind | str, provider_id: str, *, persist: bool = True
    ) -> ModelDescriptor:
        provider_kind = ProviderKind(kind)
        async with self._lock:
            provider = self._providers[provider_kind].get(provider_id)
            if provider is None:
                raise ConfigError(f"Unknown {provider_kind.value} provider {provider_id!r}")
            if not provider.config.enabled:
                raise ConfigError(f"Provider {provider_id!r} is disabled")
            self._selection[provider_kind] = provider_id
            if persist and self._store is not None:
                self._store.save({item.value: value for item, value in self._selection.items()})
        log.info(
            "model switched",
            extra={"kind": provider_kind.value, "provider": provider_id},
        )
        return provider.descriptor()

    async def reload(self, config: ModelsConfig) -> None:
        """Replace provider definitions without restarting the process."""
        new_providers = build_providers(
            config.llm.providers,
            config.stt.providers,
            config.tts.providers,
            self._http,
        )
        old_breakers = self._breakers
        async with self._lock:
            old_providers = self._providers
            self._config = config
            self._providers = new_providers
            for kind in ProviderKind:
                current = self._selection.get(kind, "")
                candidate = new_providers[kind].get(current)
                if candidate is None or not candidate.config.enabled:
                    self._selection[kind] = getattr(config, kind.value).active
            self._breakers = {
                f"{kind.value}:{provider_id}": old_breakers.get(
                    f"{kind.value}:{provider_id}",
                    CircuitBreaker(config.circuit_breaker),
                )
                for kind, group in new_providers.items()
                for provider_id in group
            }
        retired = [provider for group in old_providers.values() for provider in group.values()]
        self._retire(retired)
        log.info("model configuration reloaded")

    def _retire(self, providers: list[BaseProvider]) -> None:
        if not providers:
            return
        self._retired.extend(providers)

        async def close_later() -> None:
            await asyncio.sleep(_RETIRE_GRACE_S)
            await asyncio.gather(
                *(provider.aclose() for provider in providers),
                return_exceptions=True,
            )
            for provider in providers:
                if provider in self._retired:
                    self._retired.remove(provider)

        task = asyncio.create_task(close_later())
        self._retire_tasks.add(task)
        task.add_done_callback(self._retire_tasks.discard)

    def _apply_runtime_selection(self) -> None:
        if self._store is None:
            return
        saved = self._store.load()
        for kind in ProviderKind:
            provider_id = saved.get(kind.value)
            if not provider_id:
                continue
            provider = self._providers[kind].get(provider_id)
            if provider is not None and provider.config.enabled:
                self._selection[kind] = provider_id
            else:
                log.warning(
                    "ignoring stale runtime selection",
                    extra={"kind": kind.value, "provider": provider_id},
                )

    def _make_breakers(self) -> dict[str, CircuitBreaker]:
        return {
            f"{kind.value}:{provider_id}": CircuitBreaker(self._config.circuit_breaker)
            for kind, group in self._providers.items()
            for provider_id in group
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _resolve_llm(
        self, request: ChatRequest
    ) -> tuple[list[LLMProvider], RoutingInfo, ComplexityAssessment]:
        routing = self._config.llm.routing
        assessment = assess_complexity(request, routing)
        required = required_capabilities(assessment)
        required.add("streaming")
        reasons = list(assessment.reasons)
        explicit = False
        mode: Literal["manual", "smart"] = "manual"

        if request.provider_id:
            explicit = True
            ids = [request.provider_id]
            profile = None
            mode = "manual"
            reasons = [f"explicit provider={request.provider_id}"]
        elif routing.mode == "smart":
            if request.profile and request.profile not in routing.profiles:
                raise ConfigError(f"Unknown routing profile {request.profile!r}")
            profile = request.profile or assessment.profile
            ids = profile_chain(routing, profile)
            mode = "smart"
            reasons.append(f"profile={profile}")
        else:
            profile = None
            ids = [self._selection[ProviderKind.LLM], *routing.fallback]
            mode = "manual"
            reasons = ["manual routing"]

        ids = list(dict.fromkeys([*ids, self._selection[ProviderKind.LLM]]))
        candidates: list[LLMProvider] = []
        skipped: list[str] = []
        for provider_id in ids:
            provider = self._providers[ProviderKind.LLM].get(provider_id)
            if provider is None:
                skipped.append(f"{provider_id}: unknown")
                continue
            if not provider.config.enabled:
                skipped.append(f"{provider_id}: disabled")
                continue
            if not supports(provider.config, required):
                skipped.append(f"{provider_id}: missing {sorted(required - _caps(provider))}")
                continue
            candidates.append(provider)  # type: ignore[arg-type]

        if skipped:
            reasons.extend(f"skip {item}" for item in skipped)
        if explicit:
            provider = self._providers[ProviderKind.LLM].get(request.provider_id or "")
            if provider is None:
                raise ConfigError(f"Unknown LLM provider {request.provider_id!r}")
            if not provider.config.enabled:
                raise ConfigError(f"Provider {request.provider_id!r} is disabled")
            if not supports(provider.config, required):
                raise CapabilityError(
                    f"Provider {request.provider_id!r} cannot satisfy "
                    f"{sorted(required)}; it supports {sorted(_caps(provider))}"
                )
        if not candidates:
            raise NoProviderAvailableError(
                f"No LLM provider can handle this request (required: {sorted(required)})",
                details={
                    "required": sorted(required),
                    "candidates": ids,
                    "skipped": skipped,
                },
            )

        info = RoutingInfo(
            mode=mode,
            profile=profile,
            score=assessment.score,
            reasons=reasons,
            candidates=[provider.id for provider in candidates],
        )
        return candidates, info, assessment

    def _resolve_simple(self, kind: ProviderKind, provider_id: str | None) -> list[BaseProvider]:
        group = self._providers[kind]
        if provider_id:
            provider = group.get(provider_id)
            if provider is None:
                raise ConfigError(f"Unknown {kind.value} provider {provider_id!r}")
            if not provider.config.enabled:
                raise ConfigError(f"Provider {provider_id!r} is disabled")
            return [provider]
        section = getattr(self._config, kind.value)
        ids = [self._selection[kind], *section.fallback]
        providers = [
            group[pid] for pid in dict.fromkeys(ids) if pid in group and group[pid].config.enabled
        ]
        if not providers:
            raise NoProviderAvailableError(
                f"No {kind.value} provider is enabled", details={"candidates": ids}
            )
        return providers

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResponse:
        candidates, routing, _ = self._resolve_llm(request)
        errors: list[dict[str, Any]] = []

        for provider in candidates:
            breaker = self._breakers[f"llm:{provider.id}"]
            if not breaker.allow():
                errors.append(
                    {
                        "provider": provider.id,
                        "error": "circuit open",
                        "retryable": True,
                    }
                )
                continue
            for attempt in range(1, provider.config.max_attempts + 1):
                started = time.perf_counter()
                try:
                    async with provider._slot():
                        aggregator = StreamAggregator(provider, request, routing)
                        async for chunk in provider.stream(request):
                            aggregator.feed(chunk)
                    latency = elapsed_ms(started)
                    cost = llm_cost(provider.config.pricing, aggregator.usage)
                    final_routing = routing.model_copy(
                        update={
                            "attempts": attempt,
                            "selected_provider": provider.id,
                        }
                    )
                    response = aggregator.response(latency_ms=latency, cost_usd=cost).model_copy(
                        update={"routing": final_routing}
                    )
                    breaker.record_success()
                    self._metrics.record_success(
                        "llm",
                        provider.id,
                        latency_ms=latency,
                        cost_usd=cost,
                        usage=response.usage,
                    )
                    log.info(
                        "llm response",
                        extra={
                            "provider": provider.id,
                            "model": response.model,
                            "latency_ms": latency,
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "tool_calls": len(response.tool_calls),
                        },
                    )
                    return response
                except ProviderError as exc:
                    latency = elapsed_ms(started)
                    breaker.record_failure()
                    self._metrics.record_failure("llm", provider.id, exc.message)
                    errors.append(_error_record(provider.id, exc, attempt, latency))
                    self._annotate(exc, provider, routing)
                    if not exc.retryable:
                        raise
                    if attempt < provider.config.max_attempts:
                        await self._backoff(attempt, exc.retry_after_s)
                        continue
                    break

        raise NoProviderAvailableError(
            "All LLM providers failed",
            details={"errors": errors, "candidates": [p.id for p in candidates]},
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        candidates, routing, _ = self._resolve_llm(request)
        errors: list[dict[str, Any]] = []

        for provider in candidates:
            breaker = self._breakers[f"llm:{provider.id}"]
            if not breaker.allow():
                errors.append({"provider": provider.id, "error": "circuit open", "retryable": True})
                continue
            for attempt in range(1, provider.config.max_attempts + 1):
                aggregator = StreamAggregator(provider, request, routing)
                yield aggregator.start_event(attempt)
                emitted = False
                started = time.perf_counter()
                try:
                    async with provider._slot():
                        async for chunk in provider.stream(request):
                            events = aggregator.feed(chunk)
                            if events:
                                emitted = True
                            for event in events:
                                yield event
                    latency = elapsed_ms(started)
                    cost = llm_cost(provider.config.pricing, aggregator.usage)
                    final_routing = routing.model_copy(
                        update={
                            "attempts": attempt,
                            "selected_provider": provider.id,
                        }
                    )
                    response = aggregator.response(latency_ms=latency, cost_usd=cost).model_copy(
                        update={"routing": final_routing}
                    )
                    for call in response.tool_calls:
                        yield StreamEvent(
                            type="tool_call",
                            tool_call=call,
                            provider_id=provider.id,
                            model=response.model,
                        )
                    breaker.record_success()
                    self._metrics.record_success(
                        "llm",
                        provider.id,
                        latency_ms=latency,
                        cost_usd=cost,
                        usage=response.usage,
                    )
                    yield StreamEvent(
                        type="done",
                        provider_id=provider.id,
                        model=response.model,
                        routing=final_routing,
                        usage=response.usage,
                        finish_reason=response.finish_reason,
                        response=response,
                    )
                    return
                except ProviderError as exc:
                    latency = elapsed_ms(started)
                    breaker.record_failure()
                    self._metrics.record_failure("llm", provider.id, exc.message)
                    errors.append(_error_record(provider.id, exc, attempt, latency))
                    self._annotate(exc, provider, routing)
                    if emitted or not exc.retryable:
                        yield StreamEvent(
                            type="error",
                            provider_id=provider.id,
                            error=_error_info(exc),
                            routing=routing,
                        )
                        raise
                    if attempt < provider.config.max_attempts:
                        await self._backoff(attempt, exc.retry_after_s)
                        continue
                    break

        raise NoProviderAvailableError(
            "All LLM providers failed",
            details={"errors": errors, "candidates": [p.id for p in candidates]},
        )

    # ------------------------------------------------------------------
    # Speech to text
    # ------------------------------------------------------------------
    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        candidates = self._resolve_simple(ProviderKind.STT, request.provider_id)
        errors: list[dict[str, Any]] = []

        for provider in candidates:
            stt = provider
            assert isinstance(stt, STTProvider)
            breaker = self._breakers[f"stt:{stt.id}"]
            if not breaker.allow():
                errors.append({"provider": stt.id, "error": "circuit open"})
                continue
            for attempt in range(1, stt.config.max_attempts + 1):
                started = time.perf_counter()
                try:
                    async with stt._slot():
                        result = await stt.transcribe(request)
                    latency = elapsed_ms(started)
                    cost = stt_cost(stt.config.pricing, result.duration_s)
                    result = result.model_copy(
                        update={
                            "provider_id": stt.id,
                            "model": request.model or stt.model_name,
                            "latency_ms": latency,
                            "cost_usd": cost,
                        }
                    )
                    breaker.record_success()
                    self._metrics.record_success(
                        "stt",
                        stt.id,
                        latency_ms=latency,
                        cost_usd=cost,
                        audio_seconds=result.duration_s,
                    )
                    return result
                except ProviderError as exc:
                    latency = elapsed_ms(started)
                    breaker.record_failure()
                    self._metrics.record_failure("stt", stt.id, exc.message)
                    errors.append(_error_record(stt.id, exc, attempt, latency))
                    if not exc.retryable:
                        raise
                    if attempt < stt.config.max_attempts:
                        await self._backoff(attempt, exc.retry_after_s)
                        continue
                    break

        raise NoProviderAvailableError(
            "All STT providers failed",
            details={"errors": errors, "candidates": [p.id for p in candidates]},
        )

    # ------------------------------------------------------------------
    # Text to speech
    # ------------------------------------------------------------------
    async def synthesize(self, request: SpeechRequest) -> SpeechResult:
        candidates = self._resolve_simple(ProviderKind.TTS, request.provider_id)
        errors: list[dict[str, Any]] = []

        for provider in candidates:
            tts = provider
            assert isinstance(tts, TTSProvider)
            breaker = self._breakers[f"tts:{tts.id}"]
            if not breaker.allow():
                errors.append({"provider": tts.id, "error": "circuit open"})
                continue
            for attempt in range(1, tts.config.max_attempts + 1):
                started = time.perf_counter()
                try:
                    async with tts._slot():
                        result = await tts.synthesize(request)
                    latency = elapsed_ms(started)
                    cost = tts_cost(tts.config.pricing, len(request.text), result.duration_s)
                    result = result.model_copy(
                        update={
                            "provider_id": tts.id,
                            "model": tts.model_for(request),
                            "latency_ms": latency,
                            "cost_usd": cost,
                        }
                    )
                    breaker.record_success()
                    self._metrics.record_success(
                        "tts",
                        tts.id,
                        latency_ms=latency,
                        cost_usd=cost,
                        audio_seconds=result.duration_s,
                        characters=len(request.text),
                    )
                    return result
                except ProviderError as exc:
                    latency = elapsed_ms(started)
                    breaker.record_failure()
                    self._metrics.record_failure("tts", tts.id, exc.message)
                    errors.append(_error_record(tts.id, exc, attempt, latency))
                    if not exc.retryable:
                        raise
                    if attempt < tts.config.max_attempts:
                        await self._backoff(attempt, exc.retry_after_s)
                        continue
                    break

        raise NoProviderAvailableError(
            "All TTS providers failed",
            details={"errors": errors, "candidates": [p.id for p in candidates]},
        )

    # ------------------------------------------------------------------
    # Health / metrics
    # ------------------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for kind, group in self._providers.items():
            results[kind.value] = {}
            tasks = []
            ids = []
            for provider_id, provider in group.items():
                if not provider.config.enabled:
                    results[kind.value][provider_id] = {
                        "health": HealthStatus(
                            status="disabled", detail="provider disabled"
                        ).model_dump(),
                        "circuit": self._breakers[f"{kind.value}:{provider_id}"].snapshot(),
                    }
                    continue
                ids.append(provider_id)
                tasks.append(
                    asyncio.wait_for(
                        provider.health(),
                        timeout=min(10.0, provider.config.timeout_s),
                    )
                )
            if tasks:
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                for provider_id, outcome in zip(ids, outcomes, strict=True):
                    if isinstance(outcome, HealthStatus):
                        health = outcome
                    elif isinstance(outcome, TimeoutError):
                        health = HealthStatus(status="error", detail="health check timed out")
                    elif isinstance(outcome, Exception):
                        health = HealthStatus(
                            status="error", detail=f"{type(outcome).__name__}: {outcome}"
                        )
                    else:
                        health = HealthStatus(status="error", detail="invalid health response")
                    results[kind.value][provider_id] = {
                        "health": health.model_dump(),
                        "circuit": self._breakers[f"{kind.value}:{provider_id}"].snapshot(),
                    }
        return results

    def metrics(self) -> dict[str, Any]:
        return self._metrics.snapshot()

    async def list_llm_models(self, provider_id: str | None = None) -> dict[str, list[str]]:
        providers = self._providers[ProviderKind.LLM]
        selected = (
            {provider_id: providers[provider_id]}
            if provider_id and provider_id in providers
            else providers
        )
        if provider_id and provider_id not in providers:
            raise ConfigError(f"Unknown LLM provider {provider_id!r}")

        async def fetch(item: tuple[str, BaseProvider]) -> tuple[str, list[str]]:
            pid, provider = item
            assert isinstance(provider, LLMProvider)
            try:
                models = await asyncio.wait_for(
                    provider.list_models(), timeout=min(15.0, provider.config.timeout_s)
                )
            except Exception:
                models = [provider.model_name]
            return pid, models

        outcomes = await asyncio.gather(*(fetch(item) for item in selected.items()))
        return dict(outcomes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _backoff(self, attempt: int, retry_after_s: float | None) -> None:
        if retry_after_s is not None:
            delay = min(30.0, retry_after_s)
        else:
            delay = min(8.0, 0.4 * (2 ** (attempt - 1)))
            delay *= 1.0 + random.uniform(-0.25, 0.25)
        await asyncio.sleep(max(0.0, delay))

    @staticmethod
    def _annotate(exc: ProviderError, provider: BaseProvider, routing: RoutingInfo) -> None:
        if not exc.provider_id:
            exc.provider_id = provider.id
        exc.details.setdefault("routing", routing.model_dump())


def _caps(provider: BaseProvider) -> set[str]:
    if not isinstance(provider.config, LLMProviderConfig):
        return set()
    return provider_capabilities(provider.config)


def _error_record(
    provider_id: str, exc: ProviderError, attempt: int, latency_ms: float
) -> dict[str, Any]:
    return {
        "provider": provider_id,
        "attempt": attempt,
        "code": exc.code,
        "message": exc.message,
        "status_code": exc.status_code,
        "retryable": exc.retryable,
        "latency_ms": latency_ms,
    }


def _error_info(exc: ProviderError) -> ErrorInfo:
    return ErrorInfo(
        code=exc.code,
        message=exc.message,
        retryable=exc.retryable,
        provider_id=exc.provider_id,
        status_code=exc.status_code,
    )


__all__ = ["CircuitOpenError", "ModelRouter"]
