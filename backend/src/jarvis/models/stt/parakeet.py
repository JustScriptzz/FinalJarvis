"""Local NVIDIA Parakeet transcription through NeMo."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

from ...core.errors import AudioError, ProviderUnavailableError
from ...core.logging import get_logger
from ..audio import wav_info
from ..base import STTProvider
from ..types import (
    HealthStatus,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
)

log = get_logger(__name__)


class ParakeetSTT(STTProvider):
    type_name = "parakeet"

    def __init__(self, provider_id, config, http) -> None:
        super().__init__(provider_id, config, http)
        self._model: Any = None
        self._load_lock = asyncio.Lock()

    def _load_sync(self) -> Any:
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailableError(
                "NeMo is not installed; run: pip install 'jarvis-core[parakeet]'",
                provider_id=self.id,
            ) from exc
        log.info(
            "loading parakeet model",
            extra={"provider": self.id, "model": self.config.model},
        )
        model = ASRModel.from_pretrained(model_name=self.config.model)
        if self.config.device != "auto":
            model = model.to(self.config.device)
        model.eval()
        return model

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_sync)
        return self._model

    def _transcribe_sync(self, path: str) -> Any:
        result = self._model.transcribe(
            [path],
            batch_size=int(self.config.options.get("batch_size", 1)),
            verbose=False,
        )
        if isinstance(result, tuple):
            result = result[0]
        return result[0] if result else None

    @staticmethod
    def _duration(path: str) -> float | None:
        try:
            with open(path, "rb") as handle:
                return wav_info(handle.read()).duration_s
        except (OSError, AudioError):
            return None

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        await self._ensure_model()
        suffix = Path(request.filename).suffix or ".wav"
        fd, path = tempfile.mkstemp(prefix="jarvis-parakeet-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(request.audio)
            hypothesis = await asyncio.to_thread(self._transcribe_sync, path)
            duration = self._duration(path)
        finally:
            await asyncio.to_thread(Path(path).unlink, missing_ok=True)

        text = str(getattr(hypothesis, "text", "") or "").strip()
        segments: list[TranscriptSegment] = []
        timestamp = getattr(hypothesis, "timestamp", None)
        if isinstance(timestamp, dict):
            for index, segment in enumerate(timestamp.get("segment") or []):
                if not isinstance(segment, dict):
                    continue
                segments.append(
                    TranscriptSegment(
                        id=index,
                        start=float(segment.get("start") or 0.0),
                        end=float(segment.get("end") or 0.0),
                        text=str(segment.get("segment") or "").strip(),
                    )
                )
        if not segments and text:
            segments.append(TranscriptSegment(id=0, start=0.0, end=duration or 0.0, text=text))

        return TranscriptionResult(
            text=text,
            language=request.language or self.config.language or "en",
            duration_s=duration,
            segments=segments,
            provider_id=self.id,
            model=self.model_name,
        )

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        if importlib.util.find_spec("nemo") is None:
            return HealthStatus(
                status="error",
                detail="NeMo not installed; pip install 'jarvis-core[parakeet]'",
            )
        if self._model is None:
            return HealthStatus(status="ok", detail="NeMo available; model loads on first use")
        return HealthStatus(status="ok", detail=f"{self.model_name} loaded")

    async def aclose(self) -> None:
        self._model = None


__all__ = ["ParakeetSTT"]
