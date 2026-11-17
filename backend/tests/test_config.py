from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from jarvis.core.config import (
    JarvisConfig,
    ModelsConfig,
    _expand_env,
    load_config,
)
from jarvis.core.errors import ConfigError

from .conftest import models_config

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def test_loads_repository_config() -> None:
    config = load_config(REPO_CONFIG)
    assert config.app.name == "JARVIS"
    assert config.models.llm.active == "local-qwen"
    assert config.models.stt.active == "local-whisper"
    assert config.models.tts.active == "local-jarvis"
    assert config.app.data_dir.is_absolute()
    assert "local-qwen" in config.models.llm.providers


def test_default_voice_is_the_jarvis_piper_model() -> None:
    """The shipped default is the Piper JARVIS voice, and it must be usable.

    ``local-jarvis`` runs a Piper Onnx voice trained on the J.A.R.V.I.S.
    character voice (``jgkawell/jarvis`` on Hugging Face, MIT licensed). The
    model files live in ``data/`` and are fetched by
    ``scripts/fetch_piper_voice.py``, so a fresh checkout skips the existence
    checks rather than failing before the setup step has been run.
    """
    config = load_config(REPO_CONFIG)
    voice = config.models.tts.providers["local-jarvis"]

    assert voice.type == "piper"
    assert voice.enabled
    assert voice.model_path.endswith("jarvis-medium.onnx")
    assert voice.config_path, "Piper needs the .onnx.json beside the model"
    assert voice.config_path.endswith("jarvis-medium.onnx.json")

    model = Path(voice.model_path)
    if not model.is_file():
        pytest.skip("voice model not fetched; run backend/scripts/fetch_piper_voice.py")
    assert model.stat().st_size > 40 * 1024 * 1024, "model file looks truncated"
    assert Path(voice.config_path).is_file()


def test_local_whisper_core_matches_the_fetched_model_layout() -> None:
    """The default STT core must agree with what `fetch_whisper_model.py` writes.

    The provider passes ``options.download_root`` to faster-whisper as its Hub
    cache directory, and the fetcher writes into the same one. A mismatch means a
    multi-gigabyte download the provider never finds, then a second one at the
    first spoken sentence.
    """
    config = load_config(REPO_CONFIG)
    stt = config.models.stt.providers["local-whisper"]

    assert stt.type == "faster_whisper"
    assert stt.enabled
    assert stt.model == "large-v3"
    # auto resolves to int8 on a CPU and float16 when ctranslate2 sees a GPU.
    assert stt.compute_type == "auto"
    assert stt.device in {"auto", "cpu", "cuda"}

    download_root = Path(str(stt.options["download_root"]))
    if not download_root.is_absolute():
        download_root = REPO_CONFIG.parents[1] / download_root
    snapshots = download_root / "models--Systran--faster-whisper-large-v3" / "snapshots"
    if not snapshots.is_dir():
        pytest.skip("whisper model not fetched; run backend/scripts/fetch_whisper_model.py")
    if any((entry / "model.bin").is_file() for entry in snapshots.iterdir()):
        return
    # A snapshot directory appears before its blobs finish arriving.
    if any(download_root.rglob("*.incomplete")):
        pytest.skip("whisper download in progress")
    pytest.fail("cached whisper snapshot has no model.bin")


def test_f5_voice_core_stays_usable_as_an_alternative() -> None:
    """The F5 clone is no longer the default, but it must not rot.

    F5-TTS is slow on CPU, so it is an opt-in choice for machines with a GPU.
    Keeping it correct means it still points at a real clip and its transcript:
    F5 silently produces slurred speech when ``reference_text`` does not match
    the clip, and the only symptom is bad audio.

    The clip itself lives in ``data/``, which is not committed, so a fresh
    checkout skips the file checks until ``scripts/fetch_voice.py`` has run.
    """
    config = load_config(REPO_CONFIG)
    voice = config.models.tts.providers["local-f5"]

    assert voice.type == "f5_tts"
    assert voice.enabled
    assert voice.reference_audio
    assert voice.reference_text.strip()
    # Cloning on a machine without a GPU would otherwise select CUDA and fail.
    assert voice.device in {"cpu", "cuda", "mps", "auto"}

    if not Path(voice.reference_audio).is_file():
        pytest.skip("voice reference not fetched; run backend/scripts/fetch_voice.py")

    transcript = Path(voice.reference_audio).with_suffix(".txt")
    assert transcript.is_file(), f"{transcript} is missing; rerun fetch_voice.py"
    assert transcript.read_text(encoding="utf-8").strip() == voice.reference_text.strip()



def test_env_expansion_supports_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JARVIS_TEST_VALUE", raising=False)
    expanded = _expand_env(
        {
            "plain": "${JARVIS_TEST_VALUE}",
            "default": "${JARVIS_TEST_VALUE:-fallback}",
            "nested": ["${JARVIS_TEST_VALUE:-nested}"],
        }
    )
    assert expanded == {
        "plain": "",
        "default": "fallback",
        "nested": ["nested"],
    }


def test_env_expansion_required_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JARVIS_MISSING_SECRET", raising=False)
    with pytest.raises(ConfigError, match="JARVIS_MISSING_SECRET"):
        _expand_env({"key": "${JARVIS_MISSING_SECRET:?set it}"})


def test_unknown_provider_reference_is_rejected() -> None:
    payload = models_config()
    payload["llm"]["active"] = "missing"
    with pytest.raises(ValidationError, match="unknown provider"):
        ModelsConfig.model_validate(payload)


def test_disabled_active_provider_is_rejected() -> None:
    payload = models_config()
    payload["llm"]["providers"]["primary"]["enabled"] = False
    with pytest.raises(ValidationError, match="is disabled"):
        ModelsConfig.model_validate(payload)


def test_secrets_are_redacted_in_dumps() -> None:
    config = JarvisConfig.model_validate(
        {
            "app": {"name": "JARVIS", "data_dir": "/tmp/jarvis-test"},
            "server": {"api_token": "super-secret"},
            "models": models_config(),
        }
    )
    dumped = config.model_dump(mode="json")
    assert dumped["server"]["api_token"] == "**********"
    assert config.server.api_token is not None
    assert config.server.api_token.get_secret_value() == "super-secret"


def test_api_key_is_secret() -> None:
    config = ModelsConfig.model_validate(models_config())
    key = config.llm.providers["primary"].api_key
    assert isinstance(key, SecretStr)
    assert key.get_secret_value() == "test-key"


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("app: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)
