"""Anthropic Messages API provider (native streaming + tool use)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from ...core.errors import CapabilityError, ProviderBadRequestError, ProviderError
from ..base import LLMProvider, LLMStreamChunk, elapsed_ms
from ..sse import iter_sse
from ..types import (
    ChatMessage,
    ChatRequest,
    ContentPart,
    HealthStatus,
    ImagePart,
    TextPart,
    ToolCallDelta,
    ToolChoice,
    Usage,
)

_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _text(content: str | list[ContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if isinstance(part, TextPart))


class AnthropicLLM(LLMProvider):
    type_name = "anthropic"
    DEFAULT_BASE_URL = "https://api.anthropic.com"

    def _v1(self, path: str) -> str:
        base = self.base_url
        if base.endswith("/v1"):
            return f"{base}{path}"
        return f"{base}/v1{path}"

    def _auth_headers(self) -> dict[str, str]:
        key = self.config.secret_key
        if not key:
            raise ProviderError(f"{self.id} requires ANTHROPIC_API_KEY", provider_id=self.id)
        return {
            **self.config.headers,
            "x-api-key": key,
            "anthropic-version": self.config.api_version or "2023-06-01",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

    # -- payload ----------------------------------------------------------
    def _blocks(self, content: str | list[ContentPart] | None) -> list[dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, TextPart):
                if part.text:
                    blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                if not part.data:
                    raise CapabilityError(
                        "Anthropic requires inline image bytes; URL images are not supported"
                    )
                import base64

                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": base64.b64encode(part.data).decode("ascii"),
                        },
                    }
                )
        return blocks

    def _messages(self, messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        system_chunks: list[str] = []
        converted: list[dict[str, Any]] = []

        def append(role: str, blocks: list[dict[str, Any]]) -> None:
            if not blocks:
                return
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})

        for message in messages:
            if message.role in ("system", "developer"):
                text = _text(message.content)
                if text:
                    system_chunks.append(text)
                continue
            if message.role == "tool":
                append(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": _text(message.content),
                        }
                    ],
                )
                continue
            if message.role == "assistant":
                blocks: list[dict[str, Any]] = []
                text = _text(message.content)
                if text:
                    blocks.append({"type": "text", "text": text})
                for call in message.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                append("assistant", blocks)
                continue
            append("user", self._blocks(message.content))

        if not converted:
            raise ProviderBadRequestError(
                "Anthropic requests require at least one user/assistant message",
                provider_id=self.id,
            )
        return "\n\n".join(system_chunks), converted

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        system, messages = self._messages(request.messages)
        payload: dict[str, Any] = {
            "model": self.model_for(request),
            "max_tokens": request.max_tokens or self.config.max_output_tokens,
            "messages": messages,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop

        if request.response_format.type != "text":
            raise CapabilityError(
                f"{self.id} does not support response_format={request.response_format.type!r}"
            )

        if request.tools and request.tool_choice != "none":
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters or {"type": "object", "properties": {}},
                }
                for tool in request.tools
            ]
            if isinstance(request.tool_choice, ToolChoice):
                payload["tool_choice"] = {
                    "type": "tool",
                    "name": request.tool_choice.name,
                }
            elif request.tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif request.tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
        return payload

    # -- streaming --------------------------------------------------------
    async def stream(self, request: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        payload = self._payload(request)
        input_tokens = 0
        cached_tokens = 0
        async with self._stream(
            "POST",
            self._v1("/messages"),
            json=payload,
            headers=self._auth_headers(),
            timeout_s=request.timeout_s,
        ) as response:
            async for event, data in iter_sse(response):
                if not data:
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ProviderBadRequestError(
                        f"{self.id} returned invalid SSE JSON: {data[:200]}",
                        provider_id=self.id,
                    ) from exc
                event_type = obj.get("type") or event
                if event_type == "error":
                    error = obj.get("error") or {}
                    raise ProviderError(
                        f"{self.id} stream error: {error.get('message', error)}",
                        provider_id=self.id,
                    )
                if event_type == "message_start":
                    usage = (obj.get("message") or {}).get("usage") or {}
                    input_tokens = int(usage.get("input_tokens") or 0)
                    cached_tokens = int(usage.get("cache_read_input_tokens") or 0)
                    continue
                if event_type == "content_block_start":
                    block = obj.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        yield LLMStreamChunk(
                            tool_call=ToolCallDelta(
                                index=int(obj.get("index", 0)),
                                id=block.get("id"),
                                name=block.get("name"),
                                arguments="",
                            )
                        )
                    continue
                if event_type == "content_block_delta":
                    delta = obj.get("delta") or {}
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        text = delta.get("text") or ""
                        if text:
                            yield LLMStreamChunk(text=text)
                    elif delta_type == "input_json_delta":
                        yield LLMStreamChunk(
                            tool_call=ToolCallDelta(
                                index=int(obj.get("index", 0)),
                                arguments=delta.get("partial_json") or "",
                            )
                        )
                    continue
                if event_type == "message_delta":
                    usage = obj.get("usage") or {}
                    output_tokens = int(usage.get("output_tokens") or 0)
                    yield LLMStreamChunk(
                        usage=Usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_input_tokens=cached_tokens,
                        )
                    )
                    reason = (obj.get("delta") or {}).get("stop_reason")
                    if reason:
                        yield LLMStreamChunk(finish_reason=_STOP_REASONS.get(reason, "unknown"))

    # -- introspection ----------------------------------------------------
    async def list_models(self) -> list[str]:
        try:
            response = await self._request(
                "GET",
                self._v1("/models?limit=100"),
                headers=self._auth_headers(),
                timeout_s=min(15.0, self.config.timeout_s),
            )
            payload = response.json()
        except (ProviderError, json.JSONDecodeError):
            return [self.model_name]
        return [
            item["id"]
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ] or [self.model_name]

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        started = time.perf_counter()
        try:
            await self._request(
                "GET",
                self._v1("/models?limit=1"),
                headers=self._auth_headers(),
                timeout_s=min(5.0, self.config.timeout_s),
            )
        except ProviderError as exc:
            return HealthStatus(status="error", detail=f"{exc.code}: {exc.message}")
        return HealthStatus(
            status="ok",
            detail=f"{self.model_name} reachable",
            latency_ms=elapsed_ms(started),
        )


__all__ = ["AnthropicLLM"]
