"""Per-provider metrics and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..core.config import PricingConfig
from .types import Usage


def llm_cost(pricing: PricingConfig, usage: Usage) -> float:
    return round(
        (usage.input_tokens / 1_000_000.0) * pricing.input_per_mtok
        + (usage.output_tokens / 1_000_000.0) * pricing.output_per_mtok
        + pricing.per_request,
        8,
    )


def stt_cost(pricing: PricingConfig, duration_s: float | None) -> float:
    minutes = (duration_s or 0.0) / 60.0
    return round(minutes * pricing.per_audio_minute + pricing.per_request, 8)


def tts_cost(pricing: PricingConfig, characters: int, duration_s: float | None) -> float:
    return round(
        (characters / 1000.0) * pricing.per_1k_chars
        + ((duration_s or 0.0) / 60.0) * pricing.per_audio_minute
        + pricing.per_request,
        8,
    )


@dataclass
class ProviderStats:
    requests: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    audio_seconds: float = 0.0
    characters: int = 0
    cost_usd: float = 0.0
    latency_total_ms: float = 0.0
    last_latency_ms: float = 0.0
    last_error: str | None = None
    last_used_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        average = self.latency_total_ms / self.requests if self.requests else 0.0
        return {
            "requests": self.requests,
            "failures": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "audio_seconds": round(self.audio_seconds, 3),
            "characters": self.characters,
            "cost_usd": round(self.cost_usd, 6),
            "avg_latency_ms": round(average, 2),
            "last_latency_ms": round(self.last_latency_ms, 2),
            "last_error": self.last_error,
            "last_used_at": self.last_used_at,
        }


@dataclass
class RouterMetrics:
    providers: dict[str, ProviderStats] = field(default_factory=dict)

    def _stats(self, kind: str, provider_id: str) -> ProviderStats:
        return self.providers.setdefault(f"{kind}:{provider_id}", ProviderStats())

    def record_success(
        self,
        kind: str,
        provider_id: str,
        *,
        latency_ms: float,
        cost_usd: float = 0.0,
        usage: Usage | None = None,
        audio_seconds: float | None = None,
        characters: int = 0,
    ) -> None:
        stats = self._stats(kind, provider_id)
        stats.requests += 1
        stats.latency_total_ms += latency_ms
        stats.last_latency_ms = latency_ms
        stats.cost_usd += cost_usd
        stats.last_used_at = datetime.now(UTC).isoformat()
        if usage:
            stats.input_tokens += usage.input_tokens
            stats.output_tokens += usage.output_tokens
        if audio_seconds:
            stats.audio_seconds += audio_seconds
        if characters:
            stats.characters += characters

    def record_failure(self, kind: str, provider_id: str, error: str) -> None:
        stats = self._stats(kind, provider_id)
        stats.requests += 1
        stats.failures += 1
        stats.last_error = error[:500]
        stats.last_used_at = datetime.now(UTC).isoformat()

    def snapshot(self) -> dict[str, object]:
        return {key: stats.to_dict() for key, stats in sorted(self.providers.items())}


__all__ = [
    "ProviderStats",
    "RouterMetrics",
    "llm_cost",
    "stt_cost",
    "tts_cost",
]
