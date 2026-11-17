"""The Faster-Whisper STT provider: device/precision resolution and health.

Loading real weights is a multi-gigabyte download and minutes of CPU, so it is
not done here. The behaviour that matters and that can go wrong silently is:

* ``device: auto`` must pick CUDA only when CTranslate2 actually sees a card.
* ``compute_type: auto`` must follow that decision — ``int8`` on a CPU, but
  ``float16`` on a GPU, where int8 throws away accuracy for no speed gain.
* Health must distinguish "installed but weights not fetched" from "ready", so
  the first spoken sentence is not a surprise download (or, offline, a failure).
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from pathlib import Path
from typing import Any

import pytest

from jarvis.core.config import STTProviderConfig
from jarvis.core.errors import ProviderUnavailableError
from jarvis.models.stt import faster_whisper as fw
from jarvis.models.stt.faster_whisper import FasterWhisperSTT

HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None


def _config(**overrides: Any) -> STTProviderConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "type": "faster_whisper",
        "local": True,
        "model": "large-v3",
    }
    values.update(overrides)
    return STTProviderConfig(**values)


def _provider(**overrides: Any) -> FasterWhisperSTT:
    return FasterWhisperSTT("local-whisper", _config(**overrides), http=None)


def _snapshot(root: Path, repo: str, name: str = "abc123") -> Path:
    """Create the cache layout faster-whisper's Hub download leaves behind."""
    path = root / f"models--{repo.replace('/', '--')}" / "snapshots" / name
    path.mkdir(parents=True)
    return path


def test_auto_device_is_cpu_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fw, "_cuda_device_count", lambda: 0)

    provider = _provider(device="auto", compute_type="auto")

    assert provider._resolve_device() == "cpu"
    assert provider._resolve_compute_type("cpu") == fw.CPU_COMPUTE_TYPE == "int8"


def test_auto_device_follows_a_visible_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fw, "_cuda_device_count", lambda: 1)

    provider = _provider(device="auto", compute_type="auto")

    assert provider._resolve_device() == "cuda"
    # int8 on a GPU costs accuracy without the speed benefit it buys on a CPU.
    assert provider._resolve_compute_type("cuda") == fw.GPU_COMPUTE_TYPE == "float16"


def test_explicit_device_beats_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fw, "_cuda_device_count", lambda: 2)

    assert _provider(device="cpu")._resolve_device() == "cpu"
    assert _provider(device="cuda")._resolve_device() == "cuda"


def test_explicit_compute_type_is_never_overridden() -> None:
    provider = _provider(compute_type="int8_float16")

    assert provider._resolve_compute_type("cuda") == "int8_float16"
    assert provider._resolve_compute_type("cpu") == "int8_float16"


@pytest.mark.parametrize("value", ["auto", "default"])
def test_auto_and_default_mean_the_same_thing(value: str) -> None:
    provider = _provider(compute_type=value)

    assert provider._resolve_compute_type("cpu") == "int8"
    assert provider._resolve_compute_type("cuda") == "float16"


@pytest.mark.skipif(not HAS_FASTER_WHISPER, reason="faster-whisper extra not installed")
def test_cuda_without_a_gpu_fails_with_an_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for CUDA on a CPU-only box must not surface as a raw runtime error."""
    monkeypatch.setattr(fw, "_cuda_device_count", lambda: 0)

    with pytest.raises(ProviderUnavailableError) as excinfo:
        _provider(device="cuda")._load_sync()

    message = str(excinfo.value)
    assert "no CUDA device" in message
    assert "device: auto" in message


def test_weights_cached_recognises_the_hub_cache_layout(tmp_path: Path) -> None:
    root = tmp_path / "whisper"
    snapshot = _snapshot(root, "Systran/faster-whisper-large-v3")
    (snapshot / "model.bin").write_bytes(b"weights")

    assert _provider(options={"download_root": str(root)})._weights_cached() is True


def test_weights_cached_rejects_a_partial_download(tmp_path: Path) -> None:
    root = tmp_path / "whisper"
    _snapshot(root, "Systran/faster-whisper-large-v3")  # directory, no model.bin

    assert _provider(options={"download_root": str(root)})._weights_cached() is False


def test_weights_cached_does_not_match_another_model(tmp_path: Path) -> None:
    root = tmp_path / "whisper"
    snapshot = _snapshot(root, "Systran/faster-whisper-small")
    (snapshot / "model.bin").write_bytes(b"weights")

    assert _provider(options={"download_root": str(root)})._weights_cached() is False


def test_weights_cached_is_false_without_a_download_root() -> None:
    assert _provider()._weights_cached() is False


def test_weights_cached_follows_a_repository_id(tmp_path: Path) -> None:
    """A custom Hub id is used verbatim, so its cache name keeps the same rule."""
    root = tmp_path / "whisper"
    snapshot = _snapshot(root, "acme/custom-whisper")
    (snapshot / "model.bin").write_bytes(b"weights")

    provider = _provider(model="acme/custom-whisper", options={"download_root": str(root)})

    assert provider._weights_cached() is True


def test_health_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fw.importlib.util, "find_spec", lambda _name: None)

    status = asyncio.run(_provider().health())

    assert status.status == "error"
    assert "faster-whisper not installed" in (status.detail or "")
    # The venv has no pip on some installs; the message must name a command that works.
    assert "uv pip install" in (status.detail or "")


def test_health_reports_cached_weights_before_they_are_loaded() -> None:
    provider = _provider()
    provider._weights_cached = lambda: True  # type: ignore[method-assign]

    status = asyncio.run(provider.health())

    assert status.status == "ok"
    assert "cached" in (status.detail or "")
    assert "downloads on first use" not in (status.detail or "")


def test_health_warns_that_weights_download_on_first_use() -> None:
    provider = _provider()
    provider._weights_cached = lambda: False  # type: ignore[method-assign]

    status = asyncio.run(provider.health())

    assert status.status == "ok"
    assert "downloads on first use" in (status.detail or "")
    assert "fetch_whisper_model" in (status.detail or "")


def test_health_reports_the_resolved_device_once_loaded() -> None:
    provider = _provider()
    provider._model = object()
    provider._resolved = ("cpu", "int8")

    status = asyncio.run(provider.health())

    assert status.status == "ok"
    assert "loaded on cpu (int8)" in (status.detail or "")


def test_snapshot_age_does_not_confuse_cache_detection(tmp_path: Path) -> None:
    """Two snapshots (an interrupted update) must still count as cached."""
    root = tmp_path / "whisper"
    old = _snapshot(root, "Systran/faster-whisper-large-v3", "old")
    (old / "model.bin").write_bytes(b"weights")
    new = _snapshot(root, "Systran/faster-whisper-large-v3", "new")
    (new / "model.bin").write_bytes(b"weights")
    time.sleep(0.01)

    assert _provider(options={"download_root": str(root)})._weights_cached() is True
