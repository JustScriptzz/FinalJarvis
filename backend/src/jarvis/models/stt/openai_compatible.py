"""OpenAI-compatible ``/audio/transcriptions`` (OpenAI, Groq, Speaches...)."""

from __future__ import annotations

import json
import time
from typing import Any

from ...core.errors import ProviderError
from ...core.logging import get_logger
from ..base import STTProvider, elapsed_ms
from ..types import (
    HealthStatus,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
)

log = get_logger(__name__)


class OpenAICompatibleSTT(STTProvider):
    type_name = "openai_compatible"

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        url = f"{self.base_url}/audio/transcriptions"
        data: dict[str, Any] = {
            "model": request.model or self.model_name,
            "response_format": str(self.config.options.get("response_format", "verbose_json")),
        }
        language = request.language or self.config.language
        if language:
            data["language"] = language
        if request.prompt:
            data["prompt"] = request.prompt
        if request.temperature is not None:
            data["temperature"] = str(request.temperature)
        if self.config.options.get("send_timestamp_granularities", False):
            data["timestamp_granularities[]"] = "segment"

        response = await self._request(
            "POST",
            url,
            headers=self._bearer_headers(),
            files={
                "file": (
                    request.filename,
                    request.audio,
                    request.media_type,
                )
            },
            data=data,
            timeout_s=request.timeout_s,
        )

        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return TranscriptionResult(
                text=response.text.strip(),
                language=language,
                provider_id=self.id,
                model=self.model_name,
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{self.id} returned invalid JSON for transcription",
                provider_id=self.id,
            ) from exc

        segments = [
            TranscriptSegment(
                id=int(segment.get("id", index)),
                start=float(segment.get("start") or 0.0),
                end=float(segment.get("end") or 0.0),
                text=str(segment.get("text") or "").strip(),
                speaker=(str(segment["speaker"]) if segment.get("speaker") is not None else None),
            )
            for index, segment in enumerate(payload.get("segments") or [])
            if isinstance(segment, dict)
        ]
        duration = payload.get("duration")
        if duration is None and segments:
            duration = segments[-1].end
        return TranscriptionResult(
            text=str(payload.get("text") or "").strip(),
            language=payload.get("language") or language,
            duration_s=float(duration) if duration is not None else None,
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


__all__ = ["OpenAICompatibleSTT"]
