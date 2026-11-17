"""OpenAI-compatible ``/audio/speech``.

Works with OpenAI, Kokoro-FastAPI, openedai-speech, Chatterbox servers and any
other local TTS exposed through the OpenAI audio API.
"""

from __future__ import annotations

import time
from typing import Any

from ...core.errors import ProviderError
from ..audio import (
    estimate_duration,
    is_wav,
    media_type_for_format,
    wav_info,
)
from ..base import TTSProvider, elapsed_ms
from ..types import HealthStatus, SpeechRequest, SpeechResult


class OpenAICompatibleTTS(TTSProvider):
    type_name = "openai_compatible"

    async def synthesize(self, request: SpeechRequest) -> SpeechResult:
        audio_format = (request.format or self.config.format).lower()
        voice = request.voice or self.config.voice or "alloy"
        payload: dict[str, Any] = {
            "model": self.model_for(request),
            "input": request.text,
            "voice": voice,
            "response_format": audio_format,
        }
        speed = request.speed if request.speed is not None else self.config.speed
        if speed and abs(speed - 1.0) > 0.001:
            payload["speed"] = speed
        if request.instructions and self.config.options.get("supports_instructions", False):
            payload["instructions"] = request.instructions
        language = request.language or self.config.language
        if language and self.config.options.get("supports_language", False):
            payload["language"] = language

        response = await self._request(
            "POST",
            f"{self.base_url}/audio/speech",
            headers=self._bearer_headers(),
            json=payload,
            timeout_s=request.timeout_s,
        )
        audio = response.content
        if not audio:
            raise ProviderError(f"{self.id} returned empty audio", provider_id=self.id)
        sample_rate = request.sample_rate or self.config.sample_rate
        duration: float | None = None
        if is_wav(audio):
            info = wav_info(audio)
            sample_rate = info.sample_rate
            duration = info.duration_s
        elif audio_format == "pcm":
            sample_rate = sample_rate or 24000
            duration = estimate_duration(audio, sample_rate)
        media_type = response.headers.get("content-type") or media_type_for_format(audio_format)
        return SpeechResult(
            audio=audio,
            media_type=media_type,
            format=audio_format,
            sample_rate=sample_rate,
            duration_s=duration,
            voice=voice,
            provider_id=self.id,
            model=self.model_for(request),
        )

    async def list_voices(self) -> list[str]:
        return list(self.config.voices)

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        started = time.perf_counter()
        try:
            await self._request(
                "GET",
                f"{self.base_url}/models",
                headers=self._bearer_headers(),
                timeout_s=min(5.0, self.config.timeout_s),
            )
        except ProviderError as exc:
            return HealthStatus(status="error", detail=f"{exc.code}: {exc.message}")
        return HealthStatus(
            status="ok",
            detail=f"{self.model_name} reachable",
            latency_ms=elapsed_ms(started),
        )


__all__ = ["OpenAICompatibleTTS"]
