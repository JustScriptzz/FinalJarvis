"""Cartesia Sonic text-to-speech."""

from __future__ import annotations

import time
from typing import Any

from ...core.errors import ProviderError
from ..audio import estimate_duration, is_wav, media_type_for_format, wav_info
from ..base import TTSProvider, elapsed_ms
from ..types import HealthStatus, SpeechRequest, SpeechResult


class CartesiaTTS(TTSProvider):
    type_name = "cartesia"
    DEFAULT_BASE_URL = "https://api.cartesia.ai"

    def _auth_headers(self) -> dict[str, str]:
        key = self.config.secret_key
        if not key:
            raise ProviderError(f"{self.id} requires CARTESIA_API_KEY", provider_id=self.id)
        return {
            **self.config.headers,
            "X-API-Key": key,
            "Cartesia-Version": self.config.api_version or "2024-06-10",
            "content-type": "application/json",
            "accept": "audio/*",
        }

    def _output_format(
        self, requested: str | None, requested_rate: int | None
    ) -> tuple[dict[str, Any], str, int]:
        value = (requested or self.config.format or "wav").lower()
        rate = requested_rate or self.config.sample_rate or 24000
        if value == "wav":
            return (
                {"container": "wav", "encoding": "pcm_s16le", "sample_rate": rate},
                "wav",
                rate,
            )
        if value in {"pcm", "raw"}:
            return (
                {"container": "raw", "encoding": "pcm_s16le", "sample_rate": rate},
                "pcm",
                rate,
            )
        if value == "mp3":
            return (
                {"container": "mp3", "encoding": "mp3", "sample_rate": 44100},
                "mp3",
                44100,
            )
        if value in {"ulaw", "mulaw"}:
            return (
                {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000},
                "ulaw",
                8000,
            )
        return (
            {"container": "wav", "encoding": "pcm_s16le", "sample_rate": rate},
            "wav",
            rate,
        )

    async def synthesize(self, request: SpeechRequest) -> SpeechResult:
        voice = request.voice or self.config.voice
        if not voice:
            raise ProviderError(f"{self.id} requires a voice id", provider_id=self.id)
        output_format, result_format, sample_rate = self._output_format(
            request.format, request.sample_rate
        )
        payload: dict[str, Any] = {
            "model_id": self.model_for(request),
            "transcript": request.text,
            "voice": {"mode": "id", "id": voice},
            "output_format": output_format,
            "language": request.language or self.config.language or "en",
        }
        speed = request.speed if request.speed is not None else self.config.speed
        if speed and abs(speed - 1.0) > 0.001:
            payload["generation_config"] = {"speed": speed}
        response = await self._request(
            "POST",
            f"{self.base_url}/tts/bytes",
            headers=self._auth_headers(),
            json=payload,
            timeout_s=request.timeout_s,
        )
        audio = response.content
        if not audio:
            raise ProviderError(f"{self.id} returned empty audio", provider_id=self.id)
        duration: float | None = None
        if is_wav(audio):
            info = wav_info(audio)
            sample_rate = info.sample_rate
            duration = info.duration_s
        elif result_format == "pcm":
            duration = estimate_duration(audio, sample_rate)
        return SpeechResult(
            audio=audio,
            media_type=media_type_for_format(result_format),
            format=result_format,
            sample_rate=sample_rate,
            duration_s=duration,
            voice=voice,
            provider_id=self.id,
            model=self.model_for(request),
        )

    async def list_voices(self) -> list[str]:
        try:
            response = await self._request(
                "GET",
                f"{self.base_url}/voices?limit=100",
                headers=self._auth_headers(),
                timeout_s=min(10.0, self.config.timeout_s),
            )
            payload = response.json()
        except (ProviderError, ValueError):
            return list(self.config.voices)
        return [
            voice["id"]
            for voice in payload.get("data", [])
            if isinstance(voice, dict) and voice.get("id")
        ] or list(self.config.voices)

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        started = time.perf_counter()
        try:
            await self._request(
                "GET",
                f"{self.base_url}/voices?limit=1",
                headers=self._auth_headers(),
                timeout_s=min(5.0, self.config.timeout_s),
            )
        except ProviderError as exc:
            return HealthStatus(status="error", detail=f"{exc.code}: {exc.message}")
        return HealthStatus(
            status="ok",
            detail=f"{self.model_name} reachable",
            latency_ms=elapsed_ms(started),
        )


__all__ = ["CartesiaTTS"]
