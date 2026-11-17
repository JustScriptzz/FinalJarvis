"""Shared test fixtures: small, deterministic model configs and mock HTTP."""

from __future__ import annotations

import json
from typing import Any

import httpx


def llm_provider(
    *,
    base_url: str = "http://primary/v1",
    model: str = "test-llm",
    **overrides: Any,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "enabled": True,
        "type": "openai_compatible",
        "base_url": base_url,
        "api_key": "test-key",
        "model": model,
        "timeout_s": 5,
        "connect_timeout_s": 1,
        "max_concurrency": 2,
        "max_attempts": 1,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "pricing": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
    }
    data.update(overrides)
    return data


def stt_provider(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "enabled": True,
        "type": "openai_compatible",
        "base_url": "http://stt/v1",
        "api_key": "test-key",
        "model": "whisper-1",
        "timeout_s": 5,
        "max_attempts": 1,
        "pricing": {"per_audio_minute": 0.006},
    }
    data.update(overrides)
    return data


def tts_provider(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "enabled": True,
        "type": "openai_compatible",
        "base_url": "http://tts/v1",
        "api_key": "test-key",
        "model": "tts-1",
        "voice": "alloy",
        "format": "wav",
        "sample_rate": 24000,
        "timeout_s": 5,
        "max_attempts": 1,
        "pricing": {"per_1k_chars": 0.015},
    }
    data.update(overrides)
    return data


def models_config(
    *,
    llm: dict[str, Any] | None = None,
    stt: dict[str, Any] | None = None,
    tts: dict[str, Any] | None = None,
    active_llm: str = "primary",
    fallback_llm: list[str] | None = None,
    routing: dict[str, Any] | None = None,
    active_stt: str = "stt",
    active_tts: str = "tts",
) -> dict[str, Any]:
    llm_providers = llm or {"primary": llm_provider()}
    routing_config: dict[str, Any] = {
        "mode": "manual",
        "fallback": fallback_llm or [],
        "profiles": {},
    }
    if routing:
        routing_config.update(routing)
    return {
        "llm": {
            "active": active_llm,
            "routing": routing_config,
            "providers": llm_providers,
        },
        "stt": {
            "active": active_stt,
            "fallback": [],
            "providers": stt or {"stt": stt_provider()},
        },
        "tts": {
            "active": active_tts,
            "fallback": [],
            "providers": tts or {"tts": tts_provider()},
        },
        "circuit_breaker": {"failure_threshold": 5, "reset_timeout_s": 30},
    }


def sse_response(events: list[dict[str, Any]], *, include_done: bool = True) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    if include_done:
        body += "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
