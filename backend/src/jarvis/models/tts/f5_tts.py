"""Local F5-TTS synthesis with voice cloning.

Requires the optional ``f5-tts`` package plus a reference clip and its exact
transcript. The model loads lazily on first use and inference runs in a worker
thread, because it is CPU-bound and would otherwise stall the event loop.

Reference decoding deliberately does **not** go through ``torchaudio.load``.
Recent torchaudio versions delegate to TorchCodec, whose wheels link against
ffmpeg shared libraries (``libavutil.so.58/60/61``) that are frequently absent
from a host, and a decode failure there looks like a synthesis failure. The clip
is plain WAV, so it is read with ``soundfile`` instead: no ffmpeg, no ABI
gamble, and the same result on every machine.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...core.errors import AudioError, ProviderUnavailableError
from ..audio import float32_to_pcm16, media_type_for_format, pcm16_to_wav
from ..base import TTSProvider, elapsed_ms
from ..types import HealthStatus, SpeechRequest, SpeechResult

# Frames this short are not worth trimming; below it the clip may be silence-free.
_TRIM_FRAME_S = 0.02
# Below this per-frame RMS a frame counts as silence, in dBFS.
_SILENCE_DBFS = -42.0


def _load_reference_samples(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV reference as float32 ``(channels, samples)`` plus its rate."""
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    # torchaudio yields (channels, samples); F5's batch path expects that layout.
    return np.ascontiguousarray(data.T), int(sample_rate)


class _SoundFileTorchAudio:
    """Proxy that keeps torchaudio's API but decodes audio with soundfile.

    Only ``load`` is replaced. Everything else (``transforms.Resample`` and
    friends) is delegated untouched, so resampling still uses torch.
    """

    def __init__(self, module: Any) -> None:
        self._module = module

    def load(self, filepath: str, *args: Any, **kwargs: Any) -> tuple[Any, int]:
        import torch

        samples, sample_rate = _load_reference_samples(str(filepath))
        return torch.from_numpy(samples), sample_rate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


def _install_reference_decoder(utils_infer: Any) -> None:
    """Route F5's reference decode through soundfile. Idempotent."""
    current = getattr(utils_infer, "torchaudio", None)
    if isinstance(current, _SoundFileTorchAudio):
        return
    utils_infer.torchaudio = _SoundFileTorchAudio(current)


def _trim_silence(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Drop leading and trailing silence, keeping at least the speech itself.

    Implemented with plain numpy so it needs neither pydub nor ffmpeg. A clip
    that is silent throughout is returned unchanged rather than emptied.
    """
    if samples.size == 0 or sample_rate <= 0:
        return samples
    frame = max(1, int(sample_rate * _TRIM_FRAME_S))
    usable = samples.size - (samples.size % frame)
    if usable <= 0:
        return samples
    frames = samples[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    threshold = 10.0 ** (_SILENCE_DBFS / 20.0)
    loud = np.flatnonzero(rms > threshold)
    if loud.size == 0:
        return samples
    start = max(0, (int(loud[0]) - 1) * frame)
    end = min(samples.size, (int(loud[-1]) + 2) * frame)
    return samples[start:end]


class F5TTSProvider(TTSProvider):
    type_name = "f5_tts"

    def __init__(self, provider_id, config, http) -> None:
        super().__init__(provider_id, config, http)
        self._model: Any = None
        self._load_lock = asyncio.Lock()
        # Inference gets its own lock: the router already holds the
        # concurrency semaphore around synthesize(), so re-acquiring
        # ``_slot()`` here would block forever on a max_concurrency of 1.
        # This lock only guards calls made directly on the provider.
        self._infer_lock = asyncio.Lock()

    def _load_sync(self) -> Any:
        try:
            from f5_tts.api import F5TTS
            from f5_tts.infer import utils_infer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailableError(
                "f5-tts is not installed; install the voice extra (see docs/voices.md)",
                provider_id=self.id,
            ) from exc
        if importlib.util.find_spec("soundfile") is None:  # pragma: no cover - optional
            raise ProviderUnavailableError(
                "soundfile is required to read the reference clip; "
                "install the voice extra (see docs/voices.md)",
                provider_id=self.id,
            )
        _install_reference_decoder(utils_infer)
        kwargs: dict[str, Any] = {"model": self.config.model or "F5TTS_v1_Base"}
        if self.config.device != "auto":
            kwargs["device"] = self.config.device
        return F5TTS(**kwargs)

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_sync)
        return self._model

    def _reference(self, request: SpeechRequest) -> str:
        reference = self.config.reference_audio
        if request.voice and self.config.options.get("allow_dynamic_reference", False):
            reference = request.voice
        if not reference or not Path(reference).is_file():
            raise ProviderUnavailableError(
                f"{self.id} reference voice not found: {reference}", provider_id=self.id
            )
        return reference

    def _infer_sync(
        self, model: Any, request: SpeechRequest, reference: str
    ) -> tuple[np.ndarray, int]:
        desired: dict[str, Any] = {
            "ref_file": reference,
            "ref_text": self.config.reference_text,
            "gen_text": request.text,
            "speed": request.speed if request.speed is not None else self.config.speed,
            "nfe_step": int(self.config.options.get("nfe_step", 32)),
            "cfg_strength": float(self.config.options.get("cfg_strength", 2.0)),
            "cross_fade_duration": float(self.config.options.get("cross_fade_duration", 0.15)),
            "sway_sampling_coef": float(self.config.options.get("sway_sampling_coef", -1)),
        }
        # ``remove_silence`` is handled below: F5 only applies it when it writes
        # the wav itself, and this provider keeps audio in memory.
        # Only pass what this f5-tts version accepts, so a version that
        # renames or drops an option still works. A signature that ends in
        # ``**kwargs`` accepts everything, and filtering on named parameters
        # alone would silently drop ``gen_text``.
        parameters = inspect.signature(model.infer).parameters
        accepts_any = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
        )
        kwargs = (
            dict(desired)
            if accepts_any
            else {key: value for key, value in desired.items() if key in parameters}
        )
        try:
            result = model.infer(**kwargs)
        except Exception as exc:
            raise AudioError(
                f"{self.id} synthesis failed: {type(exc).__name__}: {exc}",
                details={"provider_id": self.id},
            ) from exc
        if not isinstance(result, tuple) or len(result) < 2:
            raise ProviderUnavailableError(
                f"{self.id} returned an unexpected inference result", provider_id=self.id
            )
        wav, sample_rate = result[0], int(result[1])
        if wav is None:
            raise AudioError(
                f"{self.id} produced no audio", details={"provider_id": self.id}
            )
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        samples = np.asarray(wav, dtype=np.float32).reshape(-1)
        if bool(self.config.options.get("remove_silence", False)):
            samples = _trim_silence(samples, sample_rate)
        return samples, sample_rate

    async def synthesize(self, request: SpeechRequest) -> SpeechResult:
        model = await self._ensure_model()
        reference = self._reference(request)
        # One inference at a time: the model is not reentrant, and CPU cores
        # are the scarce resource here. The router's semaphore already limits
        # concurrency when it drives the provider; this lock only serializes
        # direct calls.
        async with self._infer_lock:
            samples, sample_rate = await asyncio.to_thread(
                self._infer_sync, model, request, reference
            )
        if samples.size == 0:
            raise AudioError(
                "F5-TTS produced empty audio", details={"provider_id": self.id}
            )
        pcm = float32_to_pcm16(samples)
        audio_format = (request.format or self.config.format or "pcm").lower()
        if audio_format == "wav":
            audio = pcm16_to_wav(pcm, sample_rate)
            media_type = "audio/wav"
        else:
            audio = pcm
            audio_format = "pcm"
            media_type = media_type_for_format("pcm")
        return SpeechResult(
            audio=audio,
            media_type=media_type,
            format=audio_format,
            sample_rate=sample_rate,
            duration_s=len(samples) / sample_rate if sample_rate else None,
            voice=Path(reference).name,
            provider_id=self.id,
            model=self.config.model or "F5TTS_v1_Base",
        )

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        started = time.perf_counter()
        if importlib.util.find_spec("f5_tts") is None:
            return HealthStatus(
                status="error",
                detail="f5-tts not installed (install the voice extra; see docs/voices.md)",
            )
        if importlib.util.find_spec("soundfile") is None:
            return HealthStatus(
                status="error",
                detail="soundfile not installed; cannot read the reference clip",
            )
        reference = self.config.reference_audio
        if not reference or not await asyncio.to_thread(Path(reference).is_file):
            return HealthStatus(status="error", detail=f"reference voice not found: {reference}")
        if not self.config.reference_text.strip():
            return HealthStatus(
                status="error",
                detail="reference_text is empty; it must match the clip word for word",
            )
        if self._model is None:
            return HealthStatus(status="ok", detail="f5-tts available; model loads on first use")
        return HealthStatus(
            status="ok", detail="F5-TTS model loaded", latency_ms=elapsed_ms(started)
        )

    async def aclose(self) -> None:
        self._model = None


__all__ = ["F5TTSProvider"]
