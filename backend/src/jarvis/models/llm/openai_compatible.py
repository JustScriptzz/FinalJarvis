"""OpenAI-compatible chat completions.

Covers OpenAI, OpenRouter, Ollama, vLLM, llama.cpp server, LM Studio, Groq and
anything else exposing ``POST /v1/chat/completions`` with SSE streaming.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from ...core.errors import ProviderBadRequestError, ProviderError
from ...core.logging import get_logger
from ..base import LLMProvider, LLMStreamChunk, elapsed_ms
from ..sse import iter_sse
from ..types import (
    ChatMessage,
    ChatRequest,
    HealthStatus,
    ImagePart,
    TextPart,
    ToolCallDelta,
    ToolChoice,
    Usage,
)

log = get_logger(__name__)


def _finish_reason(value: str | None) -> str | None:
    if not value:
        return None
    mapping = {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
        "content_filter": "content_filter",
    }
    return mapping.get(value, "unknown")


class OpenAICompatibleLLM(LLMProvider):
    type_name = "openai_compatible"

    # -- payload ----------------------------------------------------------
    def _content(self, message: ChatMessage) -> str | list[dict[str, Any]] | None:
        content = message.content
        if content is None or isinstance(content, str):
            return content
        parts: list[dict[str, Any]] = []
        only_text = True
        for part in content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                only_text = False
                if part.data:
                    encoded = base64.b64encode(part.data).decode("ascii")
                    url = f"data:{part.media_type};base64,{encoded}"
                else:
                    url = part.url or ""
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": part.detail},
                    }
                )
        if only_text:
            return "\n".join(part["text"] for part in parts if part["type"] == "text")
        return parts

    def _messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in messages:
            item: dict[str, Any] = {
                "role": "system" if message.role == "developer" else message.role
            }
            content = self._content(message)
            if message.role == "tool" and isinstance(content, list):
                content = "\n".join(part["text"] for part in content if part["type"] == "text")
            item["content"] = content
            if message.name and message.role != "tool":
                item["name"] = message.name
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            output.append(item)
        return output

    def _tool_choice(self, choice: Any) -> Any:
        if choice is None:
            return None
        if isinstance(choice, ToolChoice):
            return {"type": "function", "function": {"name": choice.name}}
        return choice

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_for(request),
            "messages": self._messages(request.messages),
            "stream": True,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        max_tokens = request.max_tokens or self.config.max_output_tokens
        max_field = str(self.config.options.get("max_tokens_field", "max_tokens"))
        payload[max_field] = max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.tools and request.tool_choice != "none":
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
            tool_choice = self._tool_choice(request.tool_choice)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        response_format = request.response_format
        if response_format.type == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif response_format.type == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.name or "response",
                    "schema": response_format.json_schema,
                    "strict": response_format.strict,
                },
            }
        if self.config.options.get("stream_usage", self.config.supports_stream_usage):
            payload["stream_options"] = {"include_usage": True}
        return payload

    # -- streaming --------------------------------------------------------
    async def stream(self, request: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(request)
        async with self._stream(
            "POST",
            url,
            json=payload,
            headers=self._bearer_headers(),
            timeout_s=request.timeout_s,
        ) as response:
            async for _event, data in iter_sse(response):
                if not data:
                    continue
                if data.strip() == "[DONE]":
                    break
                try:
                    payload_obj = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ProviderBadRequestError(
                        f"{self.id} returned invalid SSE JSON: {data[:200]}",
                        provider_id=self.id,
                    ) from exc
                if isinstance(payload_obj, dict) and payload_obj.get("error"):
                    error = payload_obj["error"]
                    message = error.get("message") if isinstance(error, dict) else str(error)
                    raise ProviderError(f"{self.id} stream error: {message}", provider_id=self.id)

                usage = payload_obj.get("usage")
                if usage:
                    yield LLMStreamChunk(
                        usage=Usage(
                            input_tokens=int(usage.get("prompt_tokens") or 0),
                            output_tokens=int(usage.get("completion_tokens") or 0),
                            total_tokens=int(usage.get("total_tokens") or 0),
                            cached_input_tokens=int(
                                (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                                or 0
                            ),
                        )
                    )

                for choice in payload_obj.get("choices") or []:
                    if int(choice.get("index", 0)) != 0:
                        continue
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") for part in content if isinstance(part, dict)
                        )
                    if content:
                        yield LLMStreamChunk(text=str(content))
                    for call in delta.get("tool_calls") or []:
                        function = call.get("function") or {}
                        yield LLMStreamChunk(
                            tool_call=ToolCallDelta(
                                index=int(call.get("index", 0)),
                                id=call.get("id"),
                                name=function.get("name"),
                                arguments=function.get("arguments") or "",
                            )
                        )
                    reason = _finish_reason(choice.get("finish_reason"))
                    if reason:
                        yield LLMStreamChunk(finish_reason=reason)

    # -- introspection ----------------------------------------------------
    async def list_models(self) -> list[str]:
        try:
            response = await self._request(
                "GET",
                f"{self.base_url}/models",
                headers=self._bearer_headers(),
                timeout_s=min(15.0, self.config.timeout_s),
            )
        except ProviderError:
            return [self.model_name]
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return [self.model_name]
        models = [
            str(item["id"])
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
        return models or [self.model_name]

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        started = time.perf_counter()
        try:
            await self._request(
                "GET",
                f"{self.base_url}/models",
                headers=self._bearer_headers(),
                timeout_s=min(5.0, self.config.timeout_s),
            )
        except ProviderError as exc:
            return HealthStatus(status="error", detail=f"{exc.code}: {exc.message}")
        return HealthStatus(
            status="ok",
            detail=f"{self.model_name} reachable",
            latency_ms=elapsed_ms(started),
        )


__all__ = ["OpenAICompatibleLLM"]
