"""Google Gemini provider (generateContent streaming + function calling)."""

from __future__ import annotations

import base64
import json
import time
import uuid
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

_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "OTHER": "unknown",
}

# Gemini accepts an OpenAPI subset. Unknown JSON-Schema keywords make the whole
# request fail, so sanitize tool schemas before sending.
_ALLOWED_SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "default",
    "anyOf",
    "oneOf",
    "allOf",
    "propertyOrdering",
}


def sanitize_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [sanitize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(item, dict):
            cleaned[key] = {name: sanitize_schema(schema) for name, schema in item.items()}
        elif key in {"items", "anyOf", "oneOf", "allOf"}:
            cleaned[key] = sanitize_schema(item)
        else:
            cleaned[key] = item
    return cleaned


def _text(content: str | list[ContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if isinstance(part, TextPart))


class GeminiLLM(LLMProvider):
    type_name = "gemini"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def _path(self, path: str) -> str:
        base = self.base_url
        if base.endswith("/v1beta") or base.endswith("/v1"):
            return f"{base}{path}"
        return f"{base}/v1beta{path}"

    def _model_id(self, request: ChatRequest) -> str:
        model = self.model_for(request)
        return model[len("models/") :] if model.startswith("models/") else model

    def _auth_headers(self, accept: str = "text/event-stream") -> dict[str, str]:
        key = self.config.secret_key
        if not key:
            raise ProviderError(f"{self.id} requires GEMINI_API_KEY", provider_id=self.id)
        return {
            **self.config.headers,
            "x-goog-api-key": key,
            "content-type": "application/json",
            "accept": accept,
        }

    # -- payload ----------------------------------------------------------
    def _parts(self, content: str | list[ContentPart] | None) -> list[dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"text": content}] if content else []
        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, TextPart):
                if part.text:
                    parts.append({"text": part.text})
            elif isinstance(part, ImagePart):
                if not part.data:
                    raise CapabilityError(
                        "Gemini requires inline image bytes; URL images are not supported"
                    )
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": part.media_type,
                            "data": base64.b64encode(part.data).decode("ascii"),
                        }
                    }
                )
        return parts

    def _contents(self, messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        system_chunks: list[str] = []
        contents: list[dict[str, Any]] = []
        tool_names: dict[str, str] = {}

        def append(role: str, parts: list[dict[str, Any]]) -> None:
            if not parts:
                return
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        for message in messages:
            if message.role in ("system", "developer"):
                text = _text(message.content)
                if text:
                    system_chunks.append(text)
                continue
            if message.role == "assistant":
                for call in message.tool_calls:
                    tool_names[call.id] = call.name
                parts = self._parts(message.content)
                for call in message.tool_calls:
                    parts.append({"functionCall": {"name": call.name, "args": call.arguments}})
                append("model", parts)
                continue
            if message.role == "tool":
                name = message.name or tool_names.get(message.tool_call_id or "")
                if not name:
                    raise CapabilityError(
                        "Gemini tool responses require the tool name (ChatMessage.name)"
                    )
                append(
                    "user",
                    [
                        {
                            "functionResponse": {
                                "name": name,
                                "response": {"result": _text(message.content)},
                            }
                        }
                    ],
                )
                continue
            append("user", self._parts(message.content))

        if not contents:
            raise ProviderBadRequestError(
                "Gemini requests require at least one user/assistant message",
                provider_id=self.id,
            )
        return "\n\n".join(system_chunks), contents

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        system, contents = self._contents(request.messages)
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        generation: dict[str, Any] = {
            "maxOutputTokens": request.max_tokens or self.config.max_output_tokens
        }
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if request.top_p is not None:
            generation["topP"] = request.top_p
        if request.stop:
            generation["stopSequences"] = request.stop
        if request.response_format.type == "json_object":
            generation["responseMimeType"] = "application/json"
        elif request.response_format.type == "json_schema":
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = sanitize_schema(request.response_format.json_schema)
        payload["generationConfig"] = generation

        if request.tools and request.tool_choice != "none":
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": sanitize_schema(
                                tool.parameters or {"type": "object", "properties": {}}
                            ),
                        }
                        for tool in request.tools
                    ]
                }
            ]
            config: dict[str, Any] = {"mode": "AUTO"}
            if isinstance(request.tool_choice, ToolChoice):
                config = {
                    "mode": "ANY",
                    "allowedFunctionNames": [request.tool_choice.name],
                }
            elif request.tool_choice == "required":
                config = {"mode": "ANY"}
            payload["toolConfig"] = {"functionCallingConfig": config}
        return payload

    # -- streaming --------------------------------------------------------
    async def stream(self, request: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        model = self._model_id(request)
        url = self._path(f"/models/{model}:streamGenerateContent?alt=sse")
        payload = self._payload(request)
        tool_index = 0
        async with self._stream(
            "POST",
            url,
            json=payload,
            headers=self._auth_headers(),
            timeout_s=request.timeout_s,
        ) as response:
            async for _event, data in iter_sse(response):
                if not data:
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ProviderBadRequestError(
                        f"{self.id} returned invalid SSE JSON: {data[:200]}",
                        provider_id=self.id,
                    ) from exc
                if obj.get("error"):
                    error = obj["error"]
                    raise ProviderError(
                        f"{self.id} stream error: {error.get('message', error)}",
                        provider_id=self.id,
                    )
                usage = obj.get("usageMetadata")
                if usage:
                    yield LLMStreamChunk(
                        usage=Usage(
                            input_tokens=int(usage.get("promptTokenCount") or 0),
                            output_tokens=int(usage.get("candidatesTokenCount") or 0),
                            total_tokens=int(usage.get("totalTokenCount") or 0),
                            cached_input_tokens=int(usage.get("cachedContentTokenCount") or 0),
                        )
                    )
                for candidate in obj.get("candidates") or []:
                    content = candidate.get("content") or {}
                    for part in content.get("parts") or []:
                        if part.get("text"):
                            yield LLMStreamChunk(text=part["text"])
                        elif "functionCall" in part:
                            call = part["functionCall"] or {}
                            yield LLMStreamChunk(
                                tool_call=ToolCallDelta(
                                    index=tool_index,
                                    id=f"call_{uuid.uuid4().hex[:12]}",
                                    name=call.get("name"),
                                    arguments=json.dumps(
                                        call.get("args") or {}, ensure_ascii=False
                                    ),
                                )
                            )
                            tool_index += 1
                    reason = candidate.get("finishReason")
                    if reason:
                        yield LLMStreamChunk(finish_reason=_FINISH_REASONS.get(reason, "unknown"))

    # -- introspection ----------------------------------------------------
    async def list_models(self) -> list[str]:
        try:
            response = await self._request(
                "GET",
                self._path("/models?pageSize=100"),
                headers=self._auth_headers(accept="application/json"),
                timeout_s=min(15.0, self.config.timeout_s),
            )
            payload = response.json()
        except (ProviderError, json.JSONDecodeError):
            return [self.model_name]
        models = []
        for item in payload.get("models", []):
            methods = item.get("supportedGenerationMethods") or []
            if "generateContent" in methods and item.get("name"):
                models.append(item["name"].removeprefix("models/"))
        return models or [self.model_name]

    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return await super().health()
        started = time.perf_counter()
        try:
            await self._request(
                "GET",
                self._path("/models?pageSize=1"),
                headers=self._auth_headers(accept="application/json"),
                timeout_s=min(5.0, self.config.timeout_s),
            )
        except ProviderError as exc:
            return HealthStatus(status="error", detail=f"{exc.code}: {exc.message}")
        return HealthStatus(
            status="ok",
            detail=f"{self.model_name} reachable",
            latency_ms=elapsed_ms(started),
        )


__all__ = ["GeminiLLM", "sanitize_schema"]
