"""Small, dependency-light audio helpers (PCM16 / WAV).

The realtime voice pipeline (Phase 3) uses LiveKit's resamplers and frame
buffers. These helpers exist for provider boundaries: wrapping raw TTS output
in WAV, reading sample rates, and converting float32 model output to PCM16.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

from ..core.errors import AudioError

_FORMAT_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
    "ulaw": "audio/basic",
}


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration_s: float


def media_type_for_format(audio_format: str) -> str:
    return _FORMAT_MEDIA_TYPES.get(audio_format.lower(), "application/octet-stream")


def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def wav_info(data: bytes) -> WavInfo:
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            sample_rate = reader.getframerate()
            frames = reader.getnframes()
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"Invalid WAV payload: {exc}") from exc
    return WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frames=frames,
        duration_s=frames / sample_rate if sample_rate else 0.0,
    )


def wav_to_pcm16(data: bytes) -> tuple[bytes, int, int]:
    """Return ``(pcm_s16le, sample_rate, channels)`` from a PCM16 WAV."""
    info = wav_info(data)
    if info.sample_width != 2:
        raise AudioError(
            f"Only 16-bit PCM WAV is supported, got {info.sample_width * 8}-bit samples"
        )
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            frames = reader.readframes(info.frames)
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"Could not decode WAV payload: {exc}") from exc
    return frames, info.sample_rate, info.channels


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    if sample_rate <= 0:
        raise AudioError("sample_rate must be positive")
    if channels < 1:
        raise AudioError("channels must be >= 1")
    if len(pcm) % (2 * channels) != 0:
        raise AudioError("PCM byte length is not aligned to 16-bit frames")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


def float32_to_pcm16(samples: np.ndarray | list[float]) -> bytes:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1:
        array = array.reshape(-1)
    clipped = np.clip(array, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        raise AudioError("PCM byte length is not a multiple of 2")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def estimate_duration(
    pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2
) -> float | None:
    if sample_rate <= 0 or channels <= 0 or sample_width <= 0:
        return None
    frame_bytes = channels * sample_width
    if len(pcm) % frame_bytes:
        return None
    return (len(pcm) / frame_bytes) / sample_rate


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int, channels: int = 1) -> bytes:
    """Linear resampling for provider-boundary conversions.

    This is intentionally simple and deterministic. Realtime voice uses
    LiveKit's audio processing, which has proper anti-aliasing filters.
    """
    if source_rate == target_rate:
        return pcm
    if source_rate <= 0 or target_rate <= 0:
        raise AudioError("sample rates must be positive")
    samples = pcm16_to_float32(pcm)
    if samples.size % channels:
        raise AudioError("PCM sample count is not aligned to channels")
    frames = samples.reshape(-1, channels)
    source_length = frames.shape[0]
    target_length = max(1, round(source_length * target_rate / source_rate))
    source_positions = np.arange(source_length, dtype=np.float64)
    target_positions = np.linspace(0, source_length - 1, target_length, dtype=np.float64)
    output = np.empty((target_length, channels), dtype=np.float32)
    for channel in range(channels):
        output[:, channel] = np.interp(
            target_positions, source_positions, frames[:, channel]
        ).astype(np.float32)
    return float32_to_pcm16(output.reshape(-1))
