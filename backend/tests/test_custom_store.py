"""Unit tests for the custom-core store and endpoint discovery."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jarvis.core.config import JarvisConfig, load_config
from jarvis.core.custom_store import (
    CustomProviderStore,
    build_provider,
    is_private_endpoint,
    merge_custom_providers,
    slugify,
)
from jarvis.core.discovery import discover_models, normalize_base_url
from jarvis.core.errors import ConfigError

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@pytest.fixture(scope="module")
def config() -> JarvisConfig:
    return load_config(CONFIG_PATH)


def test_slugify_makes_safe_ids() -> None:
    assert slugify("My vLLM Box") == "my-vllm-box"
    assert slugify("  Qwen 2.5 / 72b ") == "qwen-2-5-72b"
    assert slugify("!!!") == "core"


def test_build_provider_validates_and_expands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_CORE_KEY", "sk-test")
    llm = build_provider(
        "llm",
        {
            "type": "openai_compatible",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "${TEST_CORE_KEY}",
            "model": "qwen2.5:72b",
            "context_window": 131072,
        },
    )
    assert llm.secret_key == "sk-test"
    assert llm.context_window == 131072
    assert llm.is_local is True

    with pytest.raises(ConfigError):
        build_provider("llm", {"type": "nonsense", "model": "x"})
    with pytest.raises(ConfigError):
        # openai_compatible needs a base_url
        build_provider("llm", {"type": "openai_compatible", "model": "x"})


def test_store_keeps_the_key_but_never_returns_it(tmp_path: Path) -> None:
    store = CustomProviderStore.in_data_dir(tmp_path)
    store.save_provider(
        "llm",
        "my-vllm",
        {
            "type": "openai_compatible",
            "base_url": "http://192.168.1.9:8000/v1",
            "api_key": "sk-secret",
            "model": "qwen2.5:72b",
        },
    )
    described = store.describe()
    assert len(described) == 1
    assert described[0]["api_key_set"] is True
    assert "sk-secret" not in json.dumps(described)

    # An update without a key keeps the stored one.
    store.save_provider(
        "llm",
        "my-vllm",
        {
            "type": "openai_compatible",
            "base_url": "http://192.168.1.9:8000/v1",
            "model": "qwen2.5:72b",
        },
    )
    assert store.get("llm", "my-vllm")["api_key"] == "sk-secret"

    assert store.remove_provider("llm", "my-vllm") is True
    assert store.remove_provider("llm", "my-vllm") is False
    assert store.describe() == []


def test_store_rejects_bad_ids(tmp_path: Path) -> None:
    store = CustomProviderStore.in_data_dir(tmp_path)
    with pytest.raises(ConfigError):
        store.save_provider(
            "llm",
            "Bad Id With Spaces",
            {"type": "openai_compatible", "base_url": "http://x/v1", "model": "m"},
        )


def test_merge_adds_and_removes_cores(config: JarvisConfig) -> None:
    merged = merge_custom_providers(
        config.models,
        [
            {
                "id": "remote-lab",
                "kind": "llm",
                "type": "openai_compatible",
                "base_url": "http://10.0.0.4:8080/v1",
                "model": "llama3.3:70b",
            }
        ],
    )
    assert "remote-lab" in merged.llm.providers
    assert "local-qwen" in merged.llm.providers

    # Merging the pristine config again (no custom entries) drops it.
    clean = merge_custom_providers(config.models, [])
    assert "remote-lab" not in clean.llm.providers


def test_merge_skips_a_broken_entry_instead_of_failing(config: JarvisConfig) -> None:
    merged = merge_custom_providers(
        config.models,
        [
            {"id": "broken", "kind": "llm", "type": "not-a-type", "model": "x"},
            {
                "id": "fine",
                "kind": "llm",
                "type": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
                "model": "fine-model",
            },
        ],
    )
    assert "broken" not in merged.llm.providers
    assert "fine" in merged.llm.providers


def test_normalize_base_url() -> None:
    assert normalize_base_url("localhost:11434/v1") == "http://localhost:11434/v1"
    assert normalize_base_url("http://h:1/v1/") == "http://h:1/v1"
    with pytest.raises(ConfigError):
        normalize_base_url("")
    with pytest.raises(ConfigError):
        normalize_base_url("ftp://h/v1")


def test_is_private_endpoint() -> None:
    assert is_private_endpoint("http://127.0.0.1:8000/v1")
    assert is_private_endpoint("http://192.168.1.5/v1")
    assert is_private_endpoint("http://172.20.0.2/v1")
    assert is_private_endpoint("http://localhost:11434/v1")
    assert not is_private_endpoint("https://api.openai.com/v1")
    assert not is_private_endpoint(None)


# ---------------------------------------------------------------------------
# Discovery against a real (local) HTTP server
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    mode: str = "openai"

    def do_GET(self) -> None:
        if self.mode == "openai" and self.path == "/v1/models":
            payload = {"object": "list", "data": [{"id": "qwen2.5:72b"}, {"id": "llama3.1:8b"}]}
        elif self.mode == "ollama" and self.path == "/api/tags":
            payload = {"models": [{"name": "qwen2.5:72b"}, {"name": "nomic-embed-text"}]}
        else:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def server_factory() -> Iterator[object]:
    servers: list[ThreadingHTTPServer] = []

    def start(mode: str) -> str:
        handler = type("_H", (_Handler,), {"mode": mode})
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        servers.append(httpd)
        host, port = httpd.server_address[:2]
        return f"http://{host}:{port}"

    yield start
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def test_discover_openai_models(server_factory: callable) -> None:
    root = server_factory("openai")
    result = asyncio.run(discover_models(f"{root}/v1"))
    assert result["source"] == "openai"
    assert result["models"] == ["llama3.1:8b", "qwen2.5:72b"]


def test_discover_native_ollama(server_factory: callable) -> None:
    root = server_factory("ollama")
    result = asyncio.run(discover_models(f"{root}/v1"))
    assert result["source"] == "ollama"
    assert "qwen2.5:72b" in result["models"]


def test_discover_reports_a_real_failure() -> None:
    with pytest.raises(ConfigError) as caught:
        asyncio.run(discover_models("http://127.0.0.1:9/v1"))
    assert "No models found" in str(caught.value)


def test_describe_exposes_editable_fields_without_the_key(tmp_path: Path) -> None:
    store = CustomProviderStore.in_data_dir(tmp_path)
    store.save_provider(
        "tts",
        "my-kokoro",
        {
            "type": "openai_compatible",
            "base_url": "http://127.0.0.1:8880/v1",
            "api_key": "sk-tts",
            "model": "kokoro",
            "voice": "af_bella",
            "format": "wav",
        },
    )
    described = store.describe()[0]
    assert described["editable"]["voice"] == "af_bella"
    assert described["editable"]["format"] == "wav"
    assert "api_key" not in described["editable"]
    assert "sk-tts" not in json.dumps(described)

@pytest.mark.parametrize(
    ("raw", "file_field"),
    [
        ({"type": "piper", "model_path": "/opt/voices/en_US-lessac-medium.onnx"}, "model_path"),
        ({"type": "f5_tts", "reference_audio": "/opt/voices/ref.wav"}, "reference_audio"),
    ],
)
def test_in_process_tts_cores_are_addable(
    tmp_path: Path, raw: dict[str, str], file_field: str
) -> None:
    store = CustomProviderStore.in_data_dir(tmp_path)
    saved = store.save_provider("tts", "local-voice", raw)
    assert saved["type"] == raw["type"]
    described = store.describe()[0]
    assert described["editable"][file_field] == raw[file_field]
    assert described["local"] is True or described["local"] is False


def test_in_process_tts_cores_still_require_their_file(tmp_path: Path) -> None:
    store = CustomProviderStore.in_data_dir(tmp_path)
    with pytest.raises(ConfigError, match="require 'model_path'"):
        store.save_provider("tts", "bad-piper", {"type": "piper"})
    with pytest.raises(ConfigError, match="reference_audio"):
        store.save_provider("tts", "bad-f5", {"type": "f5_tts"})
