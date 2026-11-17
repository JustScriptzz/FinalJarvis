"""End-to-end tests for the web server: REST, WebSocket and persistence."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jarvis.core.config import JarvisConfig, load_config
from jarvis.server.app import create_app

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _config(tmp_path: Path) -> JarvisConfig:
    config = load_config(CONFIG_PATH)
    config.app.data_dir = tmp_path
    config.database.url = f"sqlite+aiosqlite:///{tmp_path / 'jarvis.db'}"
    # Point every enabled provider at a closed port so health probes fail fast
    # and no test depends on a running model server.
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


def test_hud_and_status(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert "J.A.R.V.I.S." in index.text
    assert "/js/app.js" in index.text

    status = client.get("/api/status").json()
    assert status["ok"] is True
    assert status["data"]["setup_complete"] is False
    assert status["data"]["providers"]["llm"] > 0
    assert status["data"]["host"]["hostname"]

    setup = client.get("/api/setup").json()["data"]
    assert setup["required"] is True
    assert setup["completed"] is False
    assert any(item["id"] == "browser" for item in setup["providers"]["tts"])
    assert any(item["id"] == "browser" for item in setup["providers"]["stt"])


def test_chat_requires_explicit_core_selection(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "setup_required"
    assert body["setup"]["required"] is True


def test_setup_persists_local_or_cloud_choice(client: TestClient) -> None:
    response = client.post(
        "/api/setup",
        json={"llm": "local-qwen", "stt": "browser", "tts": "browser"},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["setup"]["completed"] is True
    assert data["setup"]["selection"]["llm"] == "local-qwen"
    assert data["setup"]["selection"]["tts"] == "browser"
    assert data["status"]["routing_mode"] == "manual"

    saved = client.get("/api/setup").json()["data"]
    assert saved["completed"] is True
    assert saved["selection"]["stt"] == "browser"

    models = client.get("/api/models").json()["data"]
    assert models["setup"]["completed"] is True


def test_setup_rejects_unknown_or_disabled_provider(client: TestClient) -> None:
    unknown = client.post(
        "/api/setup",
        json={"llm": "does-not-exist", "tts": "browser"},
    )
    assert unknown.status_code == 400

    disabled = client.post(
        "/api/setup",
        json={"llm": "anthropic-sonnet", "tts": "browser"},
    )
    assert disabled.status_code == 400
    assert "disabled" in disabled.json()["error"].lower()


def test_switch_rejects_unknown_provider(client: TestClient) -> None:
    response = client.post(
        "/api/models/switch",
        json={"kind": "llm", "provider_id": "nope"},
    )
    assert response.status_code == 400
    assert response.json()["ok"] is False


def test_commands_and_conversations(client: TestClient) -> None:
    status = client.post("/api/commands", json={"command": "/status"}).json()
    assert status["ok"] is True
    assert "All systems report" in status["data"]["text"]

    fresh = client.post("/api/conversations", json={"title": "Test thread"}).json()["data"]
    assert fresh["title"] == "Test thread"

    listed = client.get("/api/conversations").json()["data"]
    assert any(item["id"] == fresh["id"] for item in listed)

    renamed = client.patch(
        f"/api/conversations/{fresh['id']}", json={"title": "Renamed"}
    )
    assert renamed.status_code == 200
    detail = client.get(f"/api/conversations/{fresh['id']}").json()["data"]
    assert detail["conversation"]["title"] == "Renamed"

    deleted = client.delete(f"/api/conversations/{fresh['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/api/conversations/{fresh['id']}").status_code == 404


def test_memory_roundtrip(client: TestClient) -> None:
    created = client.post(
        "/api/memory", json={"text": "Prefers tea over coffee", "tags": ["preference"]}
    ).json()["data"]
    assert created["text"].startswith("Prefers tea")

    items = client.get("/api/memory").json()["data"]
    assert any(item["id"] == created["id"] for item in items)

    deleted = client.delete(f"/api/memory/{created['id']}")
    assert deleted.status_code == 200
    assert all(item["id"] != created["id"] for item in client.get("/api/memory").json()["data"])


def test_websocket_hello_and_command(client: TestClient) -> None:
    with client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
        assert hello["type"] == "hello"
        assert hello["data"]["version"] == "0.2.0"
        assert hello["data"]["setup"]["completed"] is False
        assert hello["data"]["system"]["host"]["hostname"]

        socket.send_json(
            {"type": "command", "data": {"request_id": "test", "command": "/models"}}
        )
        while True:
            frame = socket.receive_json()
            if frame["type"] == "command.result":
                break
        assert frame["data"]["error"] is False
        assert "Model matrix" in frame["data"]["text"]

        socket.send_json({"type": "telemetry.request", "data": {}})
        while True:
            frame = socket.receive_json()
            if frame["type"] == "telemetry":
                break
        assert frame["data"]["cpu"]["count"] >= 1


def test_provider_fault_is_persisted_in_the_transcript(client: TestClient) -> None:
    setup = client.post(
        "/api/setup",
        json={"llm": "local-qwen", "stt": "browser", "tts": "browser"},
    )
    assert setup.status_code == 200, setup.text

    with client.websocket_connect("/ws") as socket:
        assert socket.receive_json()["type"] == "hello"
        socket.send_json(
            {"type": "chat", "data": {"request_id": "fault", "message": "Are you there?"}}
        )
        for _ in range(400):
            frame = socket.receive_json()
            if frame["type"] in {"chat.error", "chat.done"}:
                break
        assert frame["type"] == "chat.error", frame
        data = frame["data"]
        assert data["conversation_id"]
        assert data["message_id"].startswith("msg_")
        persisted = data["error_message"]
        assert persisted["role"] == "error"
        assert persisted["id"] == data["message_id"]
        assert "Language core unavailable." in persisted["content"]

    detail = client.get(f"/api/conversations/{data['conversation_id']}").json()["data"]
    roles = [message["role"] for message in detail["messages"]]
    assert roles == ["user", "error"], roles
    assert detail["messages"][1]["id"] == data["message_id"]


def test_websocket_cancel_unknown_request_is_silent(client: TestClient) -> None:
    with client.websocket_connect("/ws") as socket:
        assert json.loads(json.dumps(socket.receive_json()))["type"] == "hello"
        socket.send_json({"type": "cancel", "data": {"request_id": "missing"}})
        socket.send_json({"type": "ping", "data": {}})
        while True:
            frame = socket.receive_json()
            if frame["type"] == "pong":
                break


def test_custom_core_can_be_added_selected_and_removed(client: TestClient, tmp_path: Path) -> None:
    payload = {
        "kind": "llm",
        "id": "my-vllm",
        "type": "openai_compatible",
        "model": "qwen2.5:72b",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "sk-test-key",
        "context_window": 131072,
    }
    created = client.post("/api/models/custom", json=payload)
    assert created.status_code == 200, created.text
    core = created.json()["data"]["core"]
    assert core["id"] == "my-vllm"
    assert core["api_key_set"] is True
    # The secret is never echoed back.
    assert "sk-test-key" not in created.text

    setup = created.json()["data"]["setup"]
    llm_ids = [item["id"] for item in setup["providers"]["llm"]]
    assert "my-vllm" in llm_ids
    # The shipped cores are still listed alongside it.
    assert any(item["id"] == "local-qwen" for item in setup["providers"]["llm"])

    # The new core is immediately selectable.
    chosen = client.post(
        "/api/setup", json={"llm": "my-vllm", "stt": "browser", "tts": "browser"}
    )
    assert chosen.status_code == 200, chosen.text
    assert chosen.json()["data"]["setup"]["selection"]["llm"] == "my-vllm"

    # And it survives a reload of the model list.
    models = client.get("/api/models").json()["data"]
    assert any(item["id"] == "my-vllm" for item in models["models"]["providers"]["llm"])

    # Removing it puts the server back into setup instead of silently
    # routing through a different core.
    removed = client.delete("/api/models/custom/llm/my-vllm")
    assert removed.status_code == 200, removed.text
    assert removed.json()["data"]["setup"]["completed"] is False

    listing = client.get("/api/models").json()["data"]
    assert not any(
        item["id"] == "my-vllm" for item in listing["models"]["providers"]["llm"]
    )
    assert not any(item["id"] == "my-vllm" for item in listing["setup"]["custom"])


def test_custom_core_rejects_an_id_that_shadows_config(client: TestClient) -> None:
    response = client.post(
        "/api/models/custom",
        json={
            "kind": "llm",
            "id": "local-qwen",
            "type": "openai_compatible",
            "model": "other",
            "base_url": "http://127.0.0.1:9/v1",
        },
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["error"]


def test_custom_core_rejects_a_bad_definition(client: TestClient) -> None:
    response = client.post(
        "/api/models/custom",
        json={
            "kind": "llm",
            "id": "broken-core",
            "type": "openai_compatible",
            "model": "m",
            "base_url": "",
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["ok"] is False


def test_discover_reports_an_unreachable_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/models/discover", json={"base_url": "http://127.0.0.1:9/v1"}
    )
    assert response.status_code == 400
    assert "No models found" in response.json()["error"]


def test_custom_core_update_keeps_the_stored_key(client: TestClient) -> None:
    created = client.post(
        "/api/models/custom",
        json={
            "kind": "llm",
            "id": "my-lab",
            "type": "openai_compatible",
            "model": "qwen2.5:14b",
            "base_url": "http://127.0.0.1:9/v1",
            "api_key": "sk-original",
        },
    )
    assert created.status_code == 200, created.text
    # The dialog needs the editable fields back, minus the secret.
    editable = created.json()["data"]["core"]["editable"]
    assert editable["base_url"] == "http://127.0.0.1:9/v1"
    assert "api_key" not in editable

    updated = client.put(
        "/api/models/custom/llm/my-lab",
        json={
            "kind": "llm",
            "id": "my-lab",
            "type": "openai_compatible",
            "model": "llama3.3:70b",
            "base_url": "http://127.0.0.1:9/v1",
            "context_window": 131072,
        },
    )
    assert updated.status_code == 200, updated.text
    core = updated.json()["data"]["core"]
    assert core["model"] == "llama3.3:70b"
    assert core["editable"]["context_window"] == 131072
    # Blank key on update must not wipe the stored credential.
    assert core["api_key_set"] is True

    removed = client.delete("/api/models/custom/llm/my-lab")
    assert removed.status_code == 200


def test_custom_core_update_rejects_an_unknown_core(client: TestClient) -> None:
    response = client.put(
        "/api/models/custom/llm/not-there",
        json={
            "kind": "llm",
            "id": "not-there",
            "type": "openai_compatible",
            "model": "m",
            "base_url": "http://127.0.0.1:9/v1",
        },
    )
    assert response.status_code == 404
