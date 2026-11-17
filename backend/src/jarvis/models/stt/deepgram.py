"""Deepgram pre-recorded transcription (Nova family)."""

from __future__ import annotations

import json
import time
from urllib.parse import urlencode

from ...core.errors import ProviderError
from ..base import STTProvider, elapsed_ms
from ..types import (
    HealthStatus,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
)


class DeepgramSTT(STTProvider):
    type_name = "deepgram"
    DEFAULT_BASE_URL = "https://api.deepgram.com"

    def _auth_headers(self) -> dict[str, str]:
        key = self.config.secret_key
        if not key:
            raise ProviderError(f"{self.id} requires DEEPGRAM_API_KEY", provider_id=self.id)
        return {
            **self.config.headers,
            "Authorization": f"Token {key}",
        }

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        language = request.language or self.config.language
        params: dict[str, str] = {
            "model": request.model or self.model_name,
            "smart_format": "true",
            "punctuate": "true",
            "utterances": "true",
        }
        if language:
            params["language"] = language
        else:
            params["detect_language"] = "true"
        if self.config.diarize:
            params["diarize"] = "true"
        if self.config.options.get("numerals"):
            params["numerals"] = "true"

        url = f"{self.base_url}/v1/listen?{urlencode(params)}"
        response = await self._request(
            "POST",
            url,
            headers={
                **self._auth_headers(),
                "Content-Type": request.media_type,
            },
            content=request.audio,
            timeout_s=request.timeout_s,
        )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{self.id} returned invalid JSON for transcription",
                provider_id=self.id,
            ) from exc

        results = payload.get("results") or {}
        channels = results.get("channels") or []
        channel = channels[0] if channels else {}
        alternatives = channel.get("alternatives") or []
        alternative = alternatives[0] if alternatives else {}
        metadata = payload.get("metadata") or {}

        segments: list[TranscriptSegment] = []
        utterances = results.get("utterances") or []
        if utterances:
            for index, utterance in enumerate(utterances):
                segments.append(
                    TranscriptSegment(
                        id=index,
                        start=float(utterance.get("start") or 0.0),
                        end=float(utterance.get("end") or 0.0),
                        text=str(utterance.get("transcript") or "").strip(),
                        speaker=(
                            f"speaker_{utterance.get('speaker')}"
                            if utterance.get("speaker") is not None
                            else None
                        ),
                        confidence=(
                            float(utterance["confidence"])
                            if utterance.get("confidence") is not None
                            else None
                        ),
                    )
                )
        elif alternative.get("transcript"):
            duration = float(metadata.get("duration") or 0.0)
            segments.append(
                TranscriptSegment(
                    id=0,
                    start=0.0,
                    end=duration,
                    text=str(alternative.get("transcript") or "").strip(),
                    confidence=(
                        float(alternative["confidence"])
                        if alternative.get("confidence") is not None
                        else None
                    ),
                )
            )

        return TranscriptionResult(
            text=str(alternative.get("transcript") or "").strip(),
            language=channel.get("detected_language") or metadata.get("language") or language,
            duration_s=(
                float(metadata["duration"]) if metadata.get("duration") is not None else None
            ),
            segments=segments,
            provider_id=self.id,
            model=request.model or self.model_name,
        )

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        started = time.perf_counter()
        try:
            await self._request(
                "GET",
                f"{self.base_url}/v1/projects",
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


__all__ = ["DeepgramSTT"]
