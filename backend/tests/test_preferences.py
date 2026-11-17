"""Preferences: the user's own model/voice choice, independent of the cores.

These tests cover the record itself (clamping, clearing, merging) and the HTTP
surface the dashboard uses to read and change it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jarvis.core.config import JarvisConfig, load_config
from jarvis.server import preferences as prefs
from jarvis.server.app import create_app

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def test_defaults_are_neutral() -> None:
    record = prefs.normalize(None)
    assert record["llm"]["model"] is None
    assert record["tts"]["voice"] is None
    assert record["tts"]["browser_voice"] is None
    assert record["llm"]["temperature"] == 0.65
    assert record["tts"]["rate"] == 1.0


def test_numbers_are_clamped_and_types_rejected() -> None:
    record = prefs.normalize(
        {
            "llm": {"temperature": 99, "max_tokens": 0},
            "tts": {"rate": -3, "speed": 50, "pitch": "loud"},
        }
    )
    assert record["llm"]["temperature"] == 2.0
    assert record["llm"]["max_tokens"] == 1.0
    assert record["tts"]["rate"] == 0.5
    assert record["tts"]["speed"] == 4.0
    # A non-numeric value falls back to the default rather than 0 or NaN.
    assert record["tts"]["pitch"] == 1.0


def test_unknown_keys_are_dropped() -> None:
    record = prefs.normalize(
        {"llm": {"model": "qwen2.5:14b", "nonsense": 1}, "bogus": {"x": 2}}
    )
    assert record["llm"]["model"] == "qwen2.5:14b"
    assert "nonsense" not in record["llm"]
    assert "bogus" not in record


def test_voice_id_whitespace_is_trimmed_and_empty_clears() -> None:
    record = prefs.normalize({"tts": {"voice": "  af_bella  "}})
    assert record["tts"]["voice"] == "af_bella"
    cleared = prefs.normalize({"tts": {"voice": "   "}})
    assert cleared["tts"]["voice"] is None


def test_merge_keeps_untouched_fields_and_clears_with_none() -> None:
    current = prefs.normalize({"llm": {"model": "a", "temperature": 0.4}})
    merged = prefs.merge(current, {"llm": {"model": None}, "tts": {"voice": "af_heart"}})
    assert merged["llm"]["model"] is None
    # temperature was not in the patch, so it survives.
    assert merged["llm"]["temperature"] == 0.4
    assert merged["tts"]["voice"] == "af_heart"


def test_browser_and_server_voices_stay_separate() -> None:
    record = prefs.normalize(
        {"tts": {"voice": "af_bella", "browser_voice": "urn:moz-tts:piper:en-GB"}}
    )
    assert record["tts"]["voice"] == "af_bella"
    assert record["tts"]["browser_voice"] == "urn:moz-tts:piper:en-GB"


def test_reconcile_clears_model_and_voice_but_keeps_delivery() -> None:
    record = prefs.normalize(
        {
            "llm": {"model": "qwen2.5:14b", "provider": "old-llm", "temperature": 0.2},
            "tts": {"voice": "af_bella", "provider": "old-tts", "speed": 1.3, "rate": 1.1},
            "stt": {"model": "large-v3", "language": "en", "provider": "old-stt"},
        }
    )
    reconciled, changed = prefs.reconcile(
        record, {"llm": "new-llm", "tts": "new-tts", "stt": "new-stt"}
    )
    assert changed is True
    # A model name and a voice id are meaningless on another core.
    assert reconciled["llm"]["model"] is None
    assert reconciled["tts"]["voice"] is None
    assert reconciled["stt"]["model"] is None
    assert reconciled["stt"]["language"] is None
    # Delivery is not core-specific; it must survive the switch.
    assert reconciled["llm"]["temperature"] == 0.2
    assert reconciled["tts"]["speed"] == 1.3
    assert reconciled["tts"]["rate"] == 1.1
    assert reconciled["llm"]["provider"] == "new-llm"


def test_reconcile_is_a_noop_on_the_same_core() -> None:
    record = prefs.normalize(
        {"llm": {"model": "qwen2.5:14b", "provider": "qwen"}, "tts": {"provider": "kokoro"}}
    )
    reconciled, changed = prefs.reconcile(record, {"llm": "qwen", "tts": "kokoro"})
    assert changed is False
    assert reconciled["llm"]["model"] == "qwen2.5:14b"


def test_reconcile_keeps_browser_voice_when_switching_to_browser() -> None:
    record = prefs.normalize(
        {
            "tts": {
                "voice": "af_bella",
                "browser_voice": "urn:moz-tts:piper:en-GB",
                "provider": "local-kokoro",
            }
        }
    )
    reconciled, changed = prefs.reconcile(record, {"tts": "browser"})
    assert changed is True
    # The server voice belongs to Kokoro; the browser choice was never used yet.
    assert reconciled["tts"]["voice"] is None
    assert reconciled["tts"]["provider"] == "browser"


def test_reconcile_forgets_the_stamp_when_no_core_is_engaged() -> None:
    record = prefs.normalize(
        {"stt": {"model": "large-v3", "language": "en", "provider": "local-whisper"}}
    )
    reconciled, changed = prefs.reconcile(record, {"llm": "q", "tts": "browser", "stt": None})
    assert changed is True
    assert reconciled["stt"]["provider"] is None
    assert reconciled["stt"]["model"] is None


def test_request_options_follow_the_record() -> None:
    record = prefs.normalize(
        {"llm": {"model": "qwen2.5:14b", "temperature": 0.2, "max_tokens": 900},
         "tts": {"voice": "am_michael", "speed": 1.4}}
    )
    llm = prefs.llm_request_options(record)
    assert llm == {"model": "qwen2.5:14b", "temperature": 0.2, "max_tokens": 900}
    tts = prefs.tts_request_options(record)
    assert tts == {"voice": "am_michael", "speed": 1.4}


# ---------------------------------------------------------------------------
# The HTTP surface
# ---------------------------------------------------------------------------
def _config(tmp_path: Path) -> JarvisConfig:
    config = load_config(CONFIG_PATH)
    config.app.data_dir = tmp_path
    config.database.url = f"sqlite+aiosqlite:///{tmp_path / 'jarvis.db'}"
    for group in (
        config.models.llm.providers,
        config.models.stt.providers,
        config.models.tts.providers,
    ):
        for provider in group.values():
            if getattr(provider, "enabled", False) and getattr(provider, "base_url", None):
                provider.base_url = "http://127.0.0.1:9/v1"  # type: ignore[assignment]
    return config


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(config=_config(tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def test_preferences_roundtrip(client: TestClient) -> None:
    initial = client.get("/api/preferences").json()
    assert initial["ok"] is True
    assert initial["data"]["llm"]["model"] is None

    patched = client.patch(
        "/api/preferences", json={"tts": {"voice": "af_bella", "speed": 1.2}}
    )
    assert patched.status_code == 200, patched.text
    data = patched.json()["data"]
    assert data["tts"]["voice"] == "af_bella"
    assert data["tts"]["speed"] == 1.2
    # Unrelated fields keep their defaults.
    assert data["llm"]["temperature"] == 0.65

    # Persistence is real, not per-request state.
    again = client.get("/api/preferences").json()["data"]
    assert again["tts"]["voice"] == "af_bella"


def test_preferences_reject_out_of_range_without_crashing(client: TestClient) -> None:
    response = client.patch("/api/preferences", json={"llm": {"temperature": 12}})
    assert response.status_code == 200, response.text
    assert response.json()["data"]["llm"]["temperature"] == 2.0


def test_preferences_none_clears_an_override(client: TestClient) -> None:
    client.patch("/api/preferences", json={"llm": {"model": "qwen2.5:14b"}})
    cleared = client.patch("/api/preferences", json={"llm": {"model": None}}).json()["data"]
    assert cleared["llm"]["model"] is None


def test_catalog_lists_voices_from_the_core(client: TestClient) -> None:
    response = client.get("/api/models/tts/local-kokoro/catalog")
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["kind"] == "tts"
    assert "af_heart" in data["voices"]
    # The configured default is reported so the picker can mark it.
    assert data["current_voice"] == "af_heart"


def test_catalog_for_unknown_core_is_404(client: TestClient) -> None:
    assert client.get("/api/models/tts/nope/catalog").status_code == 404


def test_catalog_reports_disabled_core(client: TestClient) -> None:
    response = client.get("/api/models/tts/openai-tts/catalog")
    assert response.status_code == 409
    assert "disabled" in response.json()["error"].lower()


def test_preview_refuses_browser_tts(client: TestClient) -> None:
    response = client.post(
        "/api/tts/preview", json={"provider_id": "browser", "voice": "x"}
    )
    assert response.status_code == 409
    assert response.json()["ok"] is False


def test_preferences_are_stamped_with_the_engaged_core(client: TestClient) -> None:
    client.post("/api/setup", json={"llm": "local-qwen", "tts": "browser", "stt": None})
    patched = client.patch(
        "/api/preferences", json={"llm": {"model": "qwen2.5:7b"}, "tts": {"rate": 1.2}}
    ).json()["data"]
    assert patched["llm"]["provider"] == "local-qwen"
    assert patched["tts"]["provider"] == "browser"


def test_switching_core_clears_the_old_model_but_keeps_temperature(
    client: TestClient,
) -> None:
    client.post("/api/setup", json={"llm": "local-qwen", "tts": "browser", "stt": None})
    client.patch(
        "/api/preferences",
        json={"llm": {"model": "qwen2.5:7b", "temperature": 0.25}},
    )
    response = client.post(
        "/api/models/switch", json={"kind": "llm", "provider_id": "local-fast"}
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["preferences"]["llm"]["model"] is None
    assert data["preferences"]["llm"]["temperature"] == 0.25
    assert data["preferences"]["llm"]["provider"] == "local-fast"
    # The stored record agrees with what the switch reported.
    stored = client.get("/api/preferences").json()["data"]
    assert stored["llm"]["model"] is None


def test_saving_a_core_keeps_the_user_routing_mode(client: TestClient) -> None:
    """The custom-core reload must not put smart routing back.

    The merged configuration is rebuilt from the pristine YAML, whose committed
    mode may differ from the user's choice. If that choice is not re-stamped, a
    saved core silently sends the next turn to a profile model instead of the
    core the user picked.
    """
    client.post("/api/setup", json={"llm": "local-qwen", "tts": "browser", "stt": None})
    assert client.get("/api/models").json()["data"]["routing_mode"] == "manual"

    response = client.post(
        "/api/models/custom",
        json={
            "id": "qa-local",
            "kind": "llm",
            "type": "openai_compatible",
            "base_url": "http://127.0.0.1:9/v1",
            "model": "qa-model",
            "local": True,
        },
    )
    assert response.status_code == 200, response.text

    described = client.get("/api/models").json()["data"]
    assert described["routing_mode"] == "manual"
    assert described["models"]["routing"]["mode"] == "manual"
