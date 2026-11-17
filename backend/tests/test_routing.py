from __future__ import annotations

import httpx
import pytest

from jarvis.core.config import ModelsConfig
from jarvis.core.errors import ConfigError, NoProviderAvailableError
from jarvis.models.router import ModelRouter
from jarvis.models.types import ChatMessage, ChatRequest, ImagePart, TextPart

from .conftest import llm_provider, models_config, sse_response


def _stream(text: str) -> httpx.Response:
    return sse_response(
        [
            {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        ]
    )


def _smart_config() -> ModelsConfig:
    return ModelsConfig.model_validate(
        models_config(
            llm={
                "fast": llm_provider(base_url="http://fast/v1", model="fast"),
                "deep": llm_provider(base_url="http://deep/v1", model="deep"),
            },
            routing={
                "mode": "smart",
                "profiles": {
                    "fast": ["fast"],
                    "balanced": ["fast", "deep"],
                    "deep": ["deep"],
                },
                "fallback": [],
            },
            active_llm="fast",
        )
    )


async def test_smart_routing_escalates_complex_requests() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream("deep answer" if request.url.host == "deep" else "fast answer")

    tools = [
        {
            "name": f"tool_{index}",
            "description": "test",
            "parameters": {"type": "object", "properties": {}},
        }
        for index in range(8)
    ]
    messages = [ChatMessage(role="user", content="Hi") for _ in range(7)] + [
        ChatMessage(
            role="user",
            content=(
                "```python\nprint('x')\n```\n"
                "Implement and refactor this algorithm, debug the race condition, "
                "optimize the SQL and regex, analyze the architecture, prove the "
                "plan, compare strategies step by step.\n"
                "1. First\n2. Second\n3. Third"
            ),
        )
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(_smart_config(), http_client=client)
        response = await router.chat(ChatRequest(messages=messages, tools=tools))
    assert response.provider_id == "deep"
    assert response.routing.profile == "deep"
    assert response.routing.score >= 0.62
    await router.aclose()


async def test_smart_routing_keeps_routine_requests_local() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream("fast answer" if request.url.host == "fast" else "deep answer")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(_smart_config(), http_client=client)
        response = await router.chat(
            ChatRequest(messages=[ChatMessage(role="user", content="What time is it?")])
        )
    assert response.provider_id == "fast"
    assert response.routing.profile == "fast"
    await router.aclose()


async def test_vision_requirement_filters_providers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream("vision answer" if request.url.host == "vision" else "text answer")

    config = ModelsConfig.model_validate(
        models_config(
            llm={
                "fast": llm_provider(base_url="http://fast/v1", supports_vision=False),
                "vision": llm_provider(
                    base_url="http://vision/v1",
                    model="vision-llm",
                    supports_vision=True,
                ),
            },
            routing={
                "mode": "smart",
                "profiles": {
                    "fast": ["fast"],
                    "balanced": ["fast", "vision"],
                    "deep": ["vision"],
                },
                "fallback": [],
            },
            active_llm="fast",
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = ModelRouter(config, http_client=client)
        response = await router.chat(
            ChatRequest(
                profile="balanced",
                messages=[
                    ChatMessage(
                        role="user",
                        content=[
                            TextPart(text="Describe"),
                            ImagePart(data=b"\x89PNG", media_type="image/png"),
                        ],
                    )
                ],
            )
        )
    assert response.provider_id == "vision"
    await router.aclose()


async def test_unknown_profile_is_rejected() -> None:
    config = _smart_config()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: _stream("nope"))
    ) as client:
        router = ModelRouter(config, http_client=client)
        with pytest.raises(ConfigError, match="Unknown routing profile"):
            await router.chat(
                ChatRequest(
                    profile="does-not-exist",
                    messages=[ChatMessage(role="user", content="Hi")],
                )
            )
    await router.aclose()


async def test_no_capable_provider_raises() -> None:
    config = ModelsConfig.model_validate(
        models_config(
            llm={"fast": llm_provider(supports_vision=False)},
            routing={
                "mode": "smart",
                "profiles": {"fast": ["fast"], "balanced": ["fast"], "deep": ["fast"]},
                "fallback": [],
            },
            active_llm="fast",
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: _stream("nope"))
    ) as client:
        router = ModelRouter(config, http_client=client)
        with pytest.raises(NoProviderAvailableError, match="No LLM provider"):
            await router.chat(
                ChatRequest(
                    messages=[
                        ChatMessage(
                            role="user",
                            content=[ImagePart(data=b"\x89PNG", media_type="image/png")],
                        )
                    ]
                )
            )
    await router.aclose()
