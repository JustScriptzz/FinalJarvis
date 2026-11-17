from __future__ import annotations

import json

import httpx
import pytest

from jarvis.core.config import ModelsConfig
from jarvis.core.errors import ProviderBadRequestError
from jarvis.models.router import ModelRouter
from jarvis.models.types import ChatMessage, ChatRequest, TextPart

from .conftest import llm_provider, models_config, sse_response


def _text_stream(text: str = "Hello from JARVIS") -> httpx.Response:
    return sse_response(
        [
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        ]
    )


async def test_chat_returns_text_usage_and_cost() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return _text_stream()

    config = ModelsConfig.model_validate(models_config())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        response = await router.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hi")]))

    assert response.text == "Hello from JARVIS"
    assert response.provider_id == "primary"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 2
    assert response.usage.total_tokens == 12
    assert response.cost_usd == pytest.approx(0.000014)
    assert response.routing.selected_provider == "primary"
    await router.aclose()


async def test_stream_chat_emits_normalized_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _text_stream("streamed")

    config = ModelsConfig.model_validate(models_config())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        events = [
            event
            async for event in router.stream_chat(
                ChatRequest(messages=[ChatMessage(role="user", content="Hi")])
            )
        ]

    types = [event.type for event in events]
    assert types[0] == "start"
    assert "text_delta" in types
    assert "usage" in types
    assert types[-1] == "done"
    assert events[-1].response is not None
    assert events[-1].response.text == "streamed"
    await router.aclose()


async def test_tool_call_arguments_are_aggregated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tools"][0]["function"]["name"] == "get_weather"
        return sse_response(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"Paris"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
        )

    config = ModelsConfig.model_validate(models_config())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        response = await router.chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="Weather in Paris?")],
                tools=[
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            )
        )

    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    assert response.tool_calls[0].parse_error is None
    await router.aclose()


async def test_retryable_failure_falls_back() -> None:
    calls = {"primary": 0, "backup": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "primary":
            calls["primary"] += 1
            return httpx.Response(500, json={"error": {"message": "boom"}})
        calls["backup"] += 1
        return _text_stream("fallback worked")

    config = ModelsConfig.model_validate(
        models_config(
            llm={
                "primary": llm_provider(),
                "backup": llm_provider(base_url="http://backup/v1", model="backup-llm"),
            },
            fallback_llm=["backup"],
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        response = await router.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hi")]))

    assert response.provider_id == "backup"
    assert response.text == "fallback worked"
    assert calls == {"primary": 1, "backup": 1}
    await router.aclose()


async def test_reload_swaps_provider_definitions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _text_stream("old" if request.url.host == "primary" else "new")

    first = ModelsConfig.model_validate(models_config())
    second_payload = models_config(
        llm={"primary": llm_provider(base_url="http://secondary/v1", model="new-model")}
    )
    second = ModelsConfig.model_validate(second_payload)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(first, http_client=client)
        before = await router.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hi")]))
        await router.reload(second)
        after = await router.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hi")]))
    assert before.text == "old"
    assert after.text == "new"
    assert after.model == "new-model"
    await router.aclose()


async def test_circuit_breaker_skips_failed_provider() -> None:
    calls = {"primary": 0, "backup": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "primary":
            calls["primary"] += 1
            return httpx.Response(500, json={"error": {"message": "boom"}})
        calls["backup"] += 1
        return _text_stream("backup")

    payload = models_config(
        llm={
            "primary": llm_provider(),
            "backup": llm_provider(base_url="http://backup/v1"),
        },
        fallback_llm=["backup"],
    )
    payload["circuit_breaker"]["failure_threshold"] = 1
    config = ModelsConfig.model_validate(payload)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        first = await router.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hi")]))
        second = await router.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hi")]))
    assert first.provider_id == "backup"
    assert second.provider_id == "backup"
    assert calls == {"primary": 1, "backup": 2}
    await router.aclose()


async def test_bad_request_does_not_fall_back() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid schema"}})

    config = ModelsConfig.model_validate(
        models_config(
            llm={
                "primary": llm_provider(),
                "backup": llm_provider(base_url="http://backup/v1"),
            },
            fallback_llm=["backup"],
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        with pytest.raises(ProviderBadRequestError):
            await router.chat(ChatRequest(messages=[ChatMessage(role="user", content="Hi")]))
    await router.aclose()


async def test_multimodal_content_is_converted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "What is this?"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        return _text_stream("a cat")

    config = ModelsConfig.model_validate(
        models_config(
            llm={
                "primary": llm_provider(
                    supports_vision=True,
                )
            }
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        response = await router.chat(
            ChatRequest(
                messages=[
                    ChatMessage(
                        role="user",
                        content=[
                            TextPart(text="What is this?"),
                            {
                                "type": "image",
                                "data": b"\x89PNG\r\n",
                                "media_type": "image/png",
                            },
                        ],
                    )
                ]
            )
        )
    assert response.text == "a cat"
    await router.aclose()
