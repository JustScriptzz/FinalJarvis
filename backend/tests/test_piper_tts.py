"""The Piper TTS provider: binary discovery, arguments, and health.

Synthesis needs the ``piper`` executable and a multi-megabyte Onnx voice, so it
is not run here. What *is* covered is the part that broke in practice: ``pip
install piper-tts`` installs the executable inside the environment's ``bin``
directory, which is *not* on ``PATH`` when the server is started through the
interpreter (``.venv/bin/python -m jarvis...``). A PATH-only lookup therefore
reported the shipped default voice as missing on a machine where it was
correctly installed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from jarvis.core.config import TTSProviderConfig
from jarvis.core.errors import ProviderUnavailableError
from jarvis.models.tts.piper import PiperTTS


def _config(
    *, binary: str = "piper", model: Path | None = None, **options: object
) -> TTSProviderConfig:
    return TTSProviderConfig(
        enabled=True,
        type="piper",
        local=True,
        binary=binary,
        model_path=str(model) if model else "/nonexistent/voice.onnx",
        sample_rate=22050,
        format="wav",
        options=dict(options),
    )


@pytest.fixture()
def venv_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pretend the running interpreter sits in a venv that has ``piper``.

    ``_binary`` looks beside ``sys.executable``, so the interpreter path is
    redirected rather than touching the real environment.
    """
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("")
    binary = tmp_path / "bin" / "piper"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(interpreter))
    return binary


def test_binary_is_found_beside_the_interpreter(
    venv_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: ``piper`` lives in the venv, not on PATH."""
    monkeypatch.setattr("shutil.which", lambda _: None)

    provider = PiperTTS("voice", _config(), http=None)

    assert provider._binary() == str(venv_binary)


def test_binary_is_found_as_exe_beside_a_windows_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows names it ``piper.exe``; the adjacent lookup must name it exactly."""
    interpreter = tmp_path / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("")
    binary = tmp_path / "Scripts" / "piper.exe"
    binary.write_text("")
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setattr("shutil.which", lambda _: None)

    provider = PiperTTS("voice", _config(), http=None)

    assert provider._binary() == str(binary)


def test_binary_falls_back_to_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A system-wide install still works when nothing sits beside the interpreter."""
    interpreter = tmp_path / "python"
    interpreter.write_text("")
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/piper")

    provider = PiperTTS("voice", _config(), http=None)

    assert provider._binary() == "/usr/local/bin/piper"


def test_missing_binary_names_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing executable must say how to fix it, not just that it is missing."""
    interpreter = tmp_path / "python"
    interpreter.write_text("")
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setattr("shutil.which", lambda _: None)

    provider = PiperTTS("voice", _config(), http=None)

    with pytest.raises(ProviderUnavailableError) as excinfo:
        provider._args()

    message = str(excinfo.value)
    assert "pip install piper-tts" in message
    assert "piper" in message


def test_args_forward_volume_and_length_scale(venv_binary: Path, tmp_path: Path) -> None:
    """``volume`` must reach the CLI: the JARVIS model clips at full scale."""
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"onnx")
    provider = PiperTTS("voice", _config(model=model, volume=0.8, length_scale=1.15), http=None)

    args = provider._args()

    assert "--volume" in args
    assert args[args.index("--volume") + 1] == "0.8"
    assert args[args.index("--length_scale") + 1] == "1.15"
    # Raw PCM on stdout is how the provider captures audio.
    assert "--output-raw" in args


def test_health_reports_a_missing_model(tmp_path: Path, venv_binary: Path) -> None:
    """Binary present but no voice model is an error with the path in it."""
    provider = PiperTTS("voice", _config(model=tmp_path / "absent.onnx"), http=None)

    status = asyncio.run(provider.health())

    assert status.status == "error"
    assert "absent.onnx" in (status.detail or "")


def test_health_is_ok_when_binary_and_model_exist(tmp_path: Path, venv_binary: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"onnx")
    provider = PiperTTS("voice", _config(model=model), http=None)

    status = asyncio.run(provider.health())

    assert status.status == "ok"
