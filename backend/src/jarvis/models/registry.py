"""Provider type registry and construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import httpx

from ..core.config import (
    LLMProviderConfig,
    STTProviderConfig,
    TTSProviderConfig,
)
from ..core.errors import ConfigError
from .base import BaseProvider, LLMProvider, ProviderKind, STTProvider, TTSProvider
from .llm.anthropic import AnthropicLLM
from .llm.gemini import GeminiLLM
from .llm.openai_compatible import OpenAICompatibleLLM
from .stt.deepgram import DeepgramSTT
from .stt.faster_whisper import FasterWhisperSTT
from .stt.openai_compatible import OpenAICompatibleSTT
from .stt.parakeet import ParakeetSTT
from .tts.cartesia import CartesiaTTS
from .tts.elevenlabs import ElevenLabsTTS
from .tts.f5_tts import F5TTSProvider
from .tts.openai_compatible import OpenAICompatibleTTS
from .tts.piper import PiperTTS

LLM_REGISTRY: dict[str, type[LLMProvider]] = {
    OpenAICompatibleLLM.type_name: OpenAICompatibleLLM,
    AnthropicLLM.type_name: AnthropicLLM,
    GeminiLLM.type_name: GeminiLLM,
}

STT_REGISTRY: dict[str, type[STTProvider]] = {
    FasterWhisperSTT.type_name: FasterWhisperSTT,
    ParakeetSTT.type_name: ParakeetSTT,
    OpenAICompatibleSTT.type_name: OpenAICompatibleSTT,
    DeepgramSTT.type_name: DeepgramSTT,
}

TTS_REGISTRY: dict[str, type[TTSProvider]] = {
    OpenAICompatibleTTS.type_name: OpenAICompatibleTTS,
    ElevenLabsTTS.type_name: ElevenLabsTTS,
    CartesiaTTS.type_name: CartesiaTTS,
    PiperTTS.type_name: PiperTTS,
    F5TTSProvider.type_name: F5TTSProvider,
}


def build_llm_providers(
    configs: Mapping[str, LLMProviderConfig], http: httpx.AsyncClient
) -> dict[str, LLMProvider]:
    return {
        provider_id: cast(LLMProvider, _build(provider_id, config, LLM_REGISTRY, http))
        for provider_id, config in configs.items()
    }


def build_stt_providers(
    configs: Mapping[str, STTProviderConfig], http: httpx.AsyncClient
) -> dict[str, STTProvider]:
    return {
        provider_id: cast(STTProvider, _build(provider_id, config, STT_REGISTRY, http))
        for provider_id, config in configs.items()
    }


def build_tts_providers(
    configs: Mapping[str, TTSProviderConfig], http: httpx.AsyncClient
) -> dict[str, TTSProvider]:
    return {
        provider_id: cast(TTSProvider, _build(provider_id, config, TTS_REGISTRY, http))
        for provider_id, config in configs.items()
    }


def build_providers(
    llm: Mapping[str, LLMProviderConfig],
    stt: Mapping[str, STTProviderConfig],
    tts: Mapping[str, TTSProviderConfig],
    http: httpx.AsyncClient,
) -> dict[ProviderKind, dict[str, BaseProvider]]:
    return {
        ProviderKind.LLM: cast(dict[str, BaseProvider[Any]], build_llm_providers(llm, http)),
        ProviderKind.STT: cast(dict[str, BaseProvider[Any]], build_stt_providers(stt, http)),
        ProviderKind.TTS: cast(dict[str, BaseProvider[Any]], build_tts_providers(tts, http)),
    }


def _build(
    provider_id: str,
    config: LLMProviderConfig | STTProviderConfig | TTSProviderConfig,
    registry: Mapping[str, type[BaseProvider[Any]]],
    http: httpx.AsyncClient,
) -> BaseProvider[Any]:
    provider_type = registry.get(config.type)
    if provider_type is None:
        raise ConfigError(
            f"Provider {provider_id!r} has unsupported type {config.type!r}; "
            f"known types: {sorted(registry)}"
        )
    return provider_type(provider_id, config, http)


__all__ = [
    "LLM_REGISTRY",
    "STT_REGISTRY",
    "TTS_REGISTRY",
    "build_llm_providers",
    "build_providers",
    "build_stt_providers",
    "build_tts_providers",
]
