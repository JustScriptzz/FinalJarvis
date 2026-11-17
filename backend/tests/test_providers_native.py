from __future__ import annotations

import json

import httpx
import pytest

from jarvis.core.config import ModelsConfig
from jarvis.models.router import ModelRouter
from jarvis.models.types import ChatMessage, ChatRequest, SpeechRequest

from .conftest import models_config, sse_response


async def test_anthropic_native_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "claude-sonnet-4-5-20250929"
        assert payload["stream"] is True
        return sse_response(
            [
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 12}},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " from Claude"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            ],
            include_done=False,
        )

    config = ModelsConfig.model_validate(
        models_config(
            llm={
                "claude": {
                    "enabled": True,
                    "type": "anthropic",
                    "api_key": "test-key",
                    "model": "claude-sonnet-4-5-20250929",
                    "max_attempts": 1,
                    "timeout_s": 5,
                    "supports_json": False,
                    "pricing": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
                }
            },
            active_llm="claude",
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        response = await router.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hi")]))
    assert response.text == "Hello from Claude"
    assert response.provider_id == "claude"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 4
    assert response.cost_usd == pytest.approx(0.000096)
    await router.aclose()


async def test_anthropic_tool_use_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tools"][0]["name"] == "get_weather"
        return sse_response(
            [
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 20}},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"city": "Paris"}',
                    },
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 8},
                },
            ],
            include_done=False,
        )

    config = ModelsConfig.model_validate(
        models_config(
            llm={
                "claude": {
                    "enabled": True,
                    "type": "anthropic",
                    "api_key": "test-key",
                    "model": "claude-sonnet-4-5-20250929",
                    "max_attempts": 1,
                    "timeout_s": 5,
                    "supports_json": False,
                }
            },
            active_llm="claude",
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        response = await router.chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="Weather?")],
                tools=[
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            )
        )
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].id == "toolu_1"
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    await router.aclose()


async def test_gemini_native_streaming() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models/gemini-2.5-pro:streamGenerateContent")
        assert request.url.params["alt"] == "sse"
        assert request.headers["x-goog-api-key"] == "test-key"
        payload = json.loads(request.content)
        assert payload["contents"][0]["parts"][0]["text"] == "Hi"
        return sse_response(
            [
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "Hello"}]},
                            "finishReason": None,
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 7,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 9,
                    },
                },
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": " Gemini"}]},
                            "finishReason": "STOP",
                        }
                    ]
                },
            ],
            include_done=False,
        )

    config = ModelsConfig.model_validate(
        models_config(
            llm={
                "gemini": {
                    "enabled": True,
                    "type": "gemini",
                    "base_url": "http://gemini/v1beta",
                    "api_key": "test-key",
                    "model": "gemini-2.5-pro",
                    "max_attempts": 1,
                    "timeout_s": 5,
                    "pricing": {"input_per_mtok": 1.25, "output_per_mtok": 10.0},
                }
            },
            active_llm="gemini",
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        response = await router.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hi")]))
    assert response.text == "Hello Gemini"
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 2
    assert response.finish_reason == "stop"
    await router.aclose()


async def test_elevenlabs_pcm_synthesis() -> None:
    pcm = b"\x00\x00" * 2400  # 0.1s at 24kHz

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/text-to-speech/voice-1"
        assert request.headers["xi-api-key"] == "test-key"
        payload = json.loads(request.content)
        assert payload["text"] == "Hello"
        assert payload["model_id"] == "eleven_flash_v2_5"
        return httpx.Response(200, content=pcm, headers={"content-type": "audio/pcm"})

    config = ModelsConfig.model_validate(
        models_config(
            tts={
                "eleven": {
                    "enabled": True,
                    "type": "elevenlabs",
                    "base_url": "http://eleven/v1",
                    "api_key": "test-key",
                    "model": "eleven_flash_v2_5",
                    "voice": "voice-1",
                    "format": "pcm",
                    "sample_rate": 24000,
                    "max_attempts": 1,
                    "timeout_s": 5,
                    "options": {"output_format": "pcm_24000"},
                    "pricing": {"per_1k_chars": 0.15},
                }
            },
            active_tts="eleven",
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        result = await router.synthesize(SpeechRequest(text="Hello"))
    assert result.audio == pcm
    assert result.format == "pcm"
    assert result.sample_rate == 24000
    assert result.duration_s == pytest.approx(0.1)
    assert result.cost_usd == pytest.approx(0.00075)
    await router.aclose()
