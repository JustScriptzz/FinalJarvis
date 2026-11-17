"""ElevenLabs text-to-speech."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

from ...core.errors import ProviderError
from ..audio import estimate_duration, is_wav, media_type_for_format, wav_info
from ..base import TTSProvider, elapsed_ms
from ..types import HealthStatus, SpeechRequest, SpeechResult


class ElevenLabsTTS(TTSProvider):
    type_name = "elevenlabs"
    DEFAULT_BASE_URL = "https://api.elevenlabs.io"

    def _v1(self, path: str) -> str:
        base = self.base_url
        if base.endswith("/v1"):
            return f"{base}{path}"
        return f"{base}/v1{path}"

    def _auth_headers(self) -> dict[str, str]:
        key = self.config.secret_key
        if not key:
            raise ProviderError(f"{self.id} requires ELEVENLABS_API_KEY", provider_id=self.id)
        return {
            **self.config.headers,
            "xi-api-key": key,
            "content-type": "application/json",
        }

    def _output_format(self, requested: str | None) -> tuple[str, str, int | None]:
        """Return (elevenlabs_format, result_format, sample_rate)."""
        configured = str(
            self.config.options.get("output_format") or ("mp3_44100_128" if not requested else "")
        )
        if requested:
            lowered = requested.lower()
            if "_" in lowered:
                configured = lowered
            elif lowered == "pcm":
                configured = "pcm_24000"
            elif lowered in {"mp3", "wav"}:
                configured = "mp3_44100_128" if lowered == "mp3" else "pcm_24000"
        if not configured:
            configured = "mp3_44100_128"
        if configured.startswith("pcm_"):
            try:
                rate = int(configured.split("_", 1)[1])
            except ValueError:
                rate = 24000
            return configured, "pcm", rate
        if configured.startswith("ulaw_"):
            return configured, "ulaw", 8000
        return configured, "mp3", None

    async def synthesize(self, request: SpeechRequest) -> SpeechResult:
        voice = request.voice or self.config.voice
        if not voice:
            raise ProviderError(f"{self.id} requires a voice id", provider_id=self.id)
        output_format, result_format, sample_rate = self._output_format(request.format)
        payload: dict[str, Any] = {
            "text": request.text,
            "model_id": self.model_for(request),
            "voice_settings": {
                "stability": float(self.config.options.get("stability", 0.5)),
                "similarity_boost": float(self.config.options.get("similarity_boost", 0.75)),
                "style": float(self.config.options.get("style", 0.0)),
                "use_speaker_boost": bool(self.config.options.get("use_speaker_boost", True)),
            },
        }
        language = request.language or self.config.language
        if language:
            payload["language_code"] = language
        response = await self._request(
            "POST",
            self._v1(f"/text-to-speech/{quote(voice, safe='')}"),
            headers=self._auth_headers(),
            params={"output_format": output_format},
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
        elif result_format in {"pcm", "ulaw"}:
            sample_rate = sample_rate or 24000
            duration = estimate_duration(
                audio, sample_rate, sample_width=1 if result_format == "ulaw" else 2
            )
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
                self._v1("/voices"),
                headers=self._auth_headers(),
                timeout_s=min(10.0, self.config.timeout_s),
            )
            payload = response.json()
        except (ProviderError, ValueError):
            return list(self.config.voices)
        return [
            voice["voice_id"]
            for voice in payload.get("voices", [])
            if isinstance(voice, dict) and voice.get("voice_id")
        ] or list(self.config.voices)

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        started = time.perf_counter()
        try:
            await self._request(
                "GET",
                self._v1("/user"),
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


__all__ = ["ElevenLabsTTS"]
