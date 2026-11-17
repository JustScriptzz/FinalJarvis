"""Local Faster-Whisper transcription (CTranslate2).

The model is loaded lazily on first use and kept resident. Inference runs in a
worker thread so the asyncio event loop stays responsive.

The same weights run on a CPU or an NVIDIA GPU, but the right arithmetic differs:
``int8`` is the useful quantization on a CPU, while a GPU wants ``float16``.
``device: auto`` and ``compute_type: auto`` therefore resolve together, so one
config file behaves correctly on a laptop and on a desktop with a card.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

from ...core.errors import ProviderUnavailableError
from ...core.logging import get_logger
from ..base import STTProvider
from ..types import (
    HealthStatus,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptSegment,
)

log = get_logger(__name__)

# CTranslate2 quantization that suits each device. A CPU has no fast half
# precision path, and int8 halves its memory traffic; a GPU is the opposite.
CPU_COMPUTE_TYPE = "int8"
GPU_COMPUTE_TYPE = "float16"
AUTO_COMPUTE_TYPES = frozenset({"auto", "default"})


def _cuda_device_count() -> int:
    """GPUs visible to CTranslate2, or 0 when the runtime cannot say."""
    try:
        import ctranslate2
    except ImportError:  # pragma: no cover - ctranslate2 ships with faster-whisper
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # pragma: no cover - defensive: a broken driver is not fatal
        return 0


def _repo_name(model: str) -> str:
    """Hub repository for a model name, so the cache layout can be inspected."""
    if "/" in model:
        return model
    try:
        from faster_whisper.utils import _MODELS
    except ImportError:  # pragma: no cover - depends on the optional extra
        return model
    return _MODELS.get(model, model)


class FasterWhisperSTT(STTProvider):
    type_name = "faster_whisper"

    def __init__(self, provider_id, config, http) -> None:
        super().__init__(provider_id, config, http)
        self._model: Any = None
        self._load_lock = asyncio.Lock()
        self._resolved: tuple[str, str] | None = None

    def _resolve_device(self) -> str:
        configured = (self.config.device or "auto").strip()
        if configured != "auto":
            return configured
        return "cuda" if _cuda_device_count() > 0 else "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        configured = (self.config.compute_type or "auto").strip()
        if configured not in AUTO_COMPUTE_TYPES:
            return configured
        return GPU_COMPUTE_TYPE if device.startswith("cuda") else CPU_COMPUTE_TYPE

    def _weights_cached(self) -> bool:
        """Whether the configured model is already on disk for offline use."""
        download_root = self.config.options.get("download_root")
        model = self.config.model
        if not download_root or not model:
            return False
        base = Path(str(download_root))
        if not base.is_dir():
            return False
        snapshots = base / f"models--{_repo_name(model).replace('/', '--')}" / "snapshots"
        if not snapshots.is_dir():
            return False
        return any(
            (entry / "model.bin").is_file() for entry in snapshots.iterdir() if entry.is_dir()
        )

    def _load_sync(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderUnavailableError(
                "faster-whisper is not installed; run: "
                "uv pip install --python .venv/bin/python 'jarvis-core[local]'",
                provider_id=self.id,
            ) from exc
        device = self._resolve_device()
        if device.startswith("cuda") and _cuda_device_count() == 0:
            raise ProviderUnavailableError(
                "local-whisper is configured for device 'cuda' but ctranslate2 reports no "
                "CUDA device; set device: auto to use the best available device",
                provider_id=self.id,
            )
        compute_type = self._resolve_compute_type(device)
        kwargs: dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
        }
        download_root = self.config.options.get("download_root")
        if download_root:
            kwargs["download_root"] = str(download_root)
        cpu_threads = self.config.options.get("cpu_threads")
        if cpu_threads:
            kwargs["cpu_threads"] = int(cpu_threads)
        model_source = self.config.model_path or self.config.model
        log.info(
            "loading faster-whisper model",
            extra={
                "provider": self.id,
                "model": model_source,
                "device": device,
                "compute_type": compute_type,
            },
        )
        model = WhisperModel(model_source, **kwargs)
        self._resolved = (device, compute_type)
        return model

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_sync)
        return self._model

    def _transcribe_sync(
        self, path: str, request: TranscriptionRequest
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        model = self._model
        segments, info = model.transcribe(
            path,
            language=request.language or self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            word_timestamps=bool(self.config.options.get("word_timestamps", False)),
            initial_prompt=request.prompt,
            condition_on_previous_text=bool(
                self.config.options.get("condition_on_previous_text", True)
            ),
        )
        collected = [
            {
                "id": index,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
            for index, segment in enumerate(segments)
        ]
        metadata = {
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
            "language_probability": getattr(info, "language_probability", None),
        }
        return collected, metadata

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        await self._ensure_model()
        suffix = Path(request.filename).suffix or ".wav"
        fd, path = tempfile.mkstemp(prefix="jarvis-stt-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(request.audio)
            collected, metadata = await asyncio.to_thread(self._transcribe_sync, path, request)
        finally:
            await asyncio.to_thread(Path(path).unlink, missing_ok=True)

        segments = [
            TranscriptSegment(
                id=int(segment["id"]),
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=str(segment["text"]),
            )
            for segment in collected
        ]
        text = " ".join(segment.text for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=metadata.get("language") or request.language or self.config.language,
            duration_s=(
                float(metadata["duration"]) if metadata.get("duration") is not None else None
            ),
            segments=segments,
            provider_id=self.id,
            model=self.model_name,
        )

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        if importlib.util.find_spec("faster_whisper") is None:
            return HealthStatus(
                status="error",
                detail=(
                    "faster-whisper not installed; run: "
                    "uv pip install --python .venv/bin/python 'jarvis-core[local]'"
                ),
            )
        if self._model is not None:
            device, compute_type = self._resolved or ("auto", "auto")
            return HealthStatus(
                status="ok",
                detail=f"{self.model_name} loaded on {device} ({compute_type})",
            )
        if self._weights_cached():
            return HealthStatus(
                status="ok",
                detail=f"{self.model_name} cached; loads into memory on first use",
            )
        return HealthStatus(
            status="ok",
            detail=(
                f"faster-whisper available; {self.model_name} downloads on first use "
                "(pre-fetch it with scripts/fetch_whisper_model.py)"
            ),
        )

    async def aclose(self) -> None:
        # CTranslate2 releases native memory when the object is collected.
        self._model = None


__all__ = ["FasterWhisperSTT"]
