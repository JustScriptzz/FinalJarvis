from __future__ import annotations

import json

import httpx
import pytest

from jarvis.core.config import ModelsConfig
from jarvis.core.errors import ConfigError
from jarvis.core.runtime_store import RuntimeSelectionStore
from jarvis.models.audio import pcm16_to_wav
from jarvis.models.router import ModelRouter
from jarvis.models.types import SpeechRequest, TranscriptionRequest

from .conftest import llm_provider, models_config, stt_provider


async def test_transcribe_openai_compatible() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        assert b'name="model"' in request.content
        return httpx.Response(
            200,
            json={
                "text": "hello world",
                "language": "en",
                "duration": 2.5,
                "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": "hello world"}],
            },
        )

    config = ModelsConfig.model_validate(models_config())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        result = await router.transcribe(
            TranscriptionRequest(audio=b"RIFF....WAVE", filename="note.wav")
        )
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.duration_s == pytest.approx(2.5)
    assert result.segments[0].text == "hello world"
    assert result.provider_id == "stt"
    assert result.cost_usd == pytest.approx(0.00025)
    await router.aclose()


async def test_synthesize_openai_compatible_wav() -> None:
    wav = pcm16_to_wav(b"\x00\x00" * 2400, 24000)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/speech"
        payload = json.loads(request.content)
        assert payload["input"] == "Hello"
        assert payload["voice"] == "alloy"
        assert payload["response_format"] == "wav"
        return httpx.Response(200, content=wav, headers={"content-type": "audio/wav"})

    config = ModelsConfig.model_validate(models_config())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        result = await router.synthesize(SpeechRequest(text="Hello"))
    assert result.audio == wav
    assert result.format == "wav"
    assert result.sample_rate == 24000
    assert result.duration_s == pytest.approx(0.1, abs=1e-4)
    assert result.provider_id == "tts"
    assert result.cost_usd == pytest.approx(0.000075)
    await router.aclose()


async def test_hot_swap_persists_across_router_instances(tmp_path) -> None:
    config = ModelsConfig.model_validate(
        models_config(
            llm={
                "primary": llm_provider(),
                "backup": llm_provider(base_url="http://backup/v1", model="backup-llm"),
            }
        )
    )
    store = RuntimeSelectionStore(tmp_path / "runtime" / "models.json")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    ) as client:
        router = ModelRouter(config, http_client=client, runtime_store=store)
        descriptor = await router.switch("llm", "backup")
        assert descriptor.id == "backup"
        assert store.load()["llm"] == "backup"
        await router.aclose()

        restored = ModelRouter(config, http_client=client, runtime_store=store)
        assert restored.selection["llm"] == "backup"
        await restored.aclose()


async def test_switch_rejects_unknown_and_disabled_providers() -> None:
    config = ModelsConfig.model_validate(
        models_config(
            llm={
                "primary": llm_provider(),
                "disabled": llm_provider(base_url="http://disabled/v1", enabled=False),
            }
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    ) as client:
        router = ModelRouter(config, http_client=client)
        with pytest.raises(ConfigError):
            await router.switch("llm", "missing")
        with pytest.raises(ConfigError):
            await router.switch("llm", "disabled")
    await router.aclose()


async def test_stt_fallback_uses_second_provider() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "stt":
            return httpx.Response(503, json={"error": {"message": "down"}})
        return httpx.Response(
            200,
            json={"text": "backup transcript", "language": "en", "duration": 1.0},
        )

    config = ModelsConfig.model_validate(
        models_config(
            stt={
                "stt": stt_provider(),
                "backup": stt_provider(base_url="http://stt-backup/v1"),
            }
        )
    )
    config.stt.fallback = ["backup"]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        result = await router.transcribe(TranscriptionRequest(audio=b"RIFF....WAVE"))
    assert result.provider_id == "backup"
    assert result.text == "backup transcript"
    await router.aclose()
