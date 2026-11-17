from __future__ import annotations

import pytest

from jarvis.core.errors import AudioError
from jarvis.models.audio import (
    estimate_duration,
    float32_to_pcm16,
    pcm16_to_float32,
    pcm16_to_wav,
    resample_pcm16,
    wav_info,
    wav_to_pcm16,
)


def test_wav_roundtrip() -> None:
    pcm = b"\x00\x01\x02\x03" * 1200
    wav = pcm16_to_wav(pcm, 24000)
    decoded, sample_rate, channels = wav_to_pcm16(wav)
    assert decoded == pcm
    assert sample_rate == 24000
    assert channels == 1
    info = wav_info(wav)
    assert info.duration_s == pytest.approx(0.1, abs=1e-4)


def test_resample_changes_length_and_preserves_pitch() -> None:
    pcm = b"\x10\x00" * 16000
    resampled = resample_pcm16(pcm, 16000, 8000)
    assert len(resampled) == 16000  # 8000 frames * 2 bytes
    assert estimate_duration(resampled, 8000) == pytest.approx(1.0)


def test_float_conversion_clips() -> None:
    pcm = float32_to_pcm16([0.0, 0.5, 1.0, -1.0, 2.0])
    samples = pcm16_to_float32(pcm)
    assert samples[0] == 0.0
    assert samples[2] == pytest.approx(1.0, abs=1e-4)
    assert samples[4] == pytest.approx(1.0, abs=1e-4)


def test_rejects_misaligned_pcm() -> None:
    with pytest.raises(AudioError):
        pcm16_to_wav(b"\x00", 16000)
