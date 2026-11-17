"""The F5-TTS provider's audio plumbing, without loading the model.

Synthesis itself needs a multi-gigabyte model and minutes of CPU, so it is not
exercised here. What *is* covered is the part that has failed in practice: the
reference clip must decode without ffmpeg, and trailing silence must be trimmed
with numpy rather than pydub.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from jarvis.models.tts import f5_tts

soundfile = pytest.importorskip("soundfile")


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 24000) -> None:
    soundfile.write(str(path), samples, sample_rate, subtype="PCM_16")


def test_reference_samples_are_channels_by_time(tmp_path: Path) -> None:
    path = tmp_path / "ref.wav"
    mono = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    _write_wav(path, mono)

    samples, sample_rate = f5_tts._load_reference_samples(str(path))

    assert sample_rate == 24000
    assert samples.shape == (1, 2400)
    assert samples.dtype == np.float32
    assert math.isclose(float(samples.max()), 0.5, abs_tol=1e-3)


def test_reference_samples_keep_every_channel(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    left = np.full(1200, 0.25, dtype=np.float32)
    right = np.full(1200, -0.25, dtype=np.float32)
    _write_wav(path, np.stack([left, right], axis=1))

    samples, _ = f5_tts._load_reference_samples(str(path))

    assert samples.shape == (2, 1200)
    assert samples[0].mean() > 0
    assert samples[1].mean() < 0


def test_trim_silence_removes_quiet_edges() -> None:
    sample_rate = 24000
    speech = (0.4 * np.sin(np.linspace(0, 200 * np.pi, sample_rate))).astype(np.float32)
    samples = np.concatenate(
        [
            np.zeros(sample_rate // 2, dtype=np.float32),
            speech,
            np.zeros(sample_rate // 2, dtype=np.float32),
        ]
    )

    trimmed = f5_tts._trim_silence(samples, sample_rate)

    assert trimmed.size < samples.size
    # The speech survives; only the padding goes.
    assert trimmed.size >= speech.size
    assert np.abs(trimmed).max() > 0.3


def test_trim_silence_leaves_silence_only_audio_alone() -> None:
    silent = np.zeros(4800, dtype=np.float32)

    trimmed = f5_tts._trim_silence(silent, 24000)

    assert trimmed.size == silent.size


def test_trim_silence_keeps_short_clips_intact() -> None:
    samples = np.full(10, 0.5, dtype=np.float32)

    trimmed = f5_tts._trim_silence(samples, 24000)

    assert trimmed.size == samples.size


def test_reference_decoder_is_installed_once() -> None:
    fake = types.SimpleNamespace(transforms="original")
    original = sys.modules.get("torchaudio")
    sys.modules.pop("torchaudio", None)
    try:
        utils_infer = types.SimpleNamespace(torchaudio=fake)
        f5_tts._install_reference_decoder(utils_infer)
        installed = utils_infer.torchaudio
        assert isinstance(installed, f5_tts._SoundFileTorchAudio)
        # Unchanged attributes still come from the wrapped module.
        assert installed.transforms == "original"

        f5_tts._install_reference_decoder(utils_infer)
        assert utils_infer.torchaudio is installed
    finally:
        if original is not None:
            sys.modules["torchaudio"] = original


def test_health_reports_missing_reference_text(tmp_path: Path) -> None:
    reference = tmp_path / "ref.wav"
    _write_wav(reference, np.zeros(2400, dtype=np.float32))
    provider = f5_tts.F5TTSProvider(
        "voice",
        _config(str(reference), reference_text=""),
        http=None,  # health() does not touch the network
    )

    import asyncio

    status = asyncio.run(provider.health())

    assert status.status == "error"
    assert "reference_text" in (status.detail or "")


def _config(reference: str, *, reference_text: str = "hello") -> object:
    from jarvis.core.config import TTSProviderConfig

    return TTSProviderConfig(
        enabled=True,
        type="f5_tts",
        local=True,
        reference_audio=reference,
        reference_text=reference_text,
        format="wav",
        sample_rate=24000,
    )


async def test_router_driven_synthesis_does_not_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The router holds the provider's concurrency slot; synthesize() must not.

    Regression: an earlier version re-acquired ``_slot()`` inside ``synthesize``
    while the router already held it with ``max_concurrency: 1``, so every
    request through the real server path hung forever. The model is stubbed out
    here, so this checks locking and result plumbing, not audio quality.
    """
    import asyncio

    from jarvis.core.config import ModelsConfig
    from jarvis.models.router import ModelRouter
    from jarvis.models.types import SpeechRequest

    reference = tmp_path / "ref.wav"
    _write_wav(reference, np.zeros(2400, dtype=np.float32))

    config = ModelsConfig.model_validate(
        {
            "llm": {"active": "x", "providers": {"x": {"type": "openai_compatible",
                    "base_url": "http://x/v1", "model": "m"}}},
            "stt": {"active": "stt", "providers": {"stt": {"type": "openai_compatible",
                    "base_url": "http://s/v1", "model": "m"}}},
            "tts": {
                "active": "voice",
                "providers": {
                    "voice": {
                        "enabled": True,
                        "type": "f5_tts",
                        "local": True,
                        "reference_audio": str(reference),
                        "reference_text": "hello",
                        "format": "wav",
                        "sample_rate": 24000,
                        "max_concurrency": 1,
                    }
                },
            },
        }
    )

    class _Stub:
        """Stands in for the loaded F5TTS model."""

        def infer(self, **kwargs: object) -> tuple[np.ndarray, int]:
            assert kwargs["gen_text"] == "hello there"
            return np.zeros(2400, dtype=np.float32), 24000

    async def _fake_ensure(self: f5_tts.F5TTSProvider) -> _Stub:
        return _Stub()

    monkeypatch.setattr(f5_tts.F5TTSProvider, "_ensure_model", _fake_ensure)

    router = ModelRouter(config)
    try:
        result = await asyncio.wait_for(
            router.synthesize(
                SpeechRequest(text="hello there", provider_id="voice", format="wav")
            ),
            timeout=10,
        )
    finally:
        await router.aclose()

    assert result.provider_id == "voice"
    assert result.audio[:4] == b"RIFF"
    assert result.sample_rate == 24000
