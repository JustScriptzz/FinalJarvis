"""Aggregate normalized provider deltas into a ChatResponse."""

from __future__ import annotations

import json
import uuid
from typing import Any

from .base import LLMProvider, LLMStreamChunk
from .types import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    RoutingInfo,
    StreamEvent,
    ToolCall,
    ToolCallDelta,
    Usage,
)

_FINISH_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "error",
    "unknown",
}


class StreamAggregator:
    def __init__(self, provider: LLMProvider, request: ChatRequest, routing: RoutingInfo) -> None:
        self.provider = provider
        self.request = request
        self.routing = routing
        self._text: list[str] = []
        self._tools: dict[int, dict[str, Any]] = {}
        self.usage = Usage()
        self.finish_reason: FinishReason | None = None

    def start_event(self, attempt: int) -> StreamEvent:
        return StreamEvent(
            type="start",
            provider_id=self.provider.id,
            model=self.provider.model_for(self.request),
            attempt=attempt,
            routing=self.routing,
        )

    def feed(self, chunk: LLMStreamChunk) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if chunk.text:
            self._text.append(chunk.text)
            events.append(
                StreamEvent(
                    type="text_delta",
                    text=chunk.text,
                    provider_id=self.provider.id,
                    model=self.provider.model_for(self.request),
                )
            )
        if chunk.tool_call is not None:
            self._accumulate_tool(chunk.tool_call)
            events.append(
                StreamEvent(
                    type="tool_call_delta",
                    tool_call_delta=chunk.tool_call,
                    provider_id=self.provider.id,
                    model=self.provider.model_for(self.request),
                )
            )
        if chunk.usage is not None:
            self._merge_usage(chunk.usage)
            events.append(StreamEvent(type="usage", usage=self.usage))
        if chunk.finish_reason:
            reason = chunk.finish_reason if chunk.finish_reason in _FINISH_REASONS else "unknown"
            self.finish_reason = reason  # type: ignore[assignment]
        return events

    def _accumulate_tool(self, delta: ToolCallDelta) -> None:
        entry = self._tools.setdefault(delta.index, {"id": None, "name": None, "arguments": []})
        if delta.id:
            entry["id"] = delta.id
        if delta.name:
            entry["name"] = delta.name
        if delta.arguments:
            entry["arguments"].append(delta.arguments)

    def _merge_usage(self, usage: Usage) -> None:
        self.usage = Usage(
            input_tokens=max(self.usage.input_tokens, usage.input_tokens),
            output_tokens=max(self.usage.output_tokens, usage.output_tokens),
            total_tokens=max(self.usage.total_tokens, usage.total_tokens),
            cached_input_tokens=max(self.usage.cached_input_tokens, usage.cached_input_tokens),
        )

    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(self._tools):
            entry = self._tools[index]
            raw = "".join(entry["arguments"]).strip()
            arguments: dict[str, Any] = {}
            parse_error: str | None = None
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        arguments = parsed
                    else:
                        parse_error = "tool arguments are not a JSON object"
                except json.JSONDecodeError as exc:
                    parse_error = str(exc)
            calls.append(
                ToolCall(
                    id=entry["id"] or f"call_{uuid.uuid4().hex[:12]}",
                    name=entry["name"] or "unknown_tool",
                    arguments=arguments,
                    raw_arguments=raw or None,
                    parse_error=parse_error,
                )
            )
        return calls

    def response(self, *, latency_ms: float, cost_usd: float) -> ChatResponse:
        calls = self.tool_calls()
        finish = self.finish_reason
        if calls and finish in (None, "stop", "unknown"):
            finish = "tool_calls"
        return ChatResponse(
            text="".join(self._text),
            tool_calls=calls,
            finish_reason=finish or "stop",
            usage=self.usage,
            provider_id=self.provider.id,
            model=self.provider.model_for(self.request),
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            routing=self.routing,
        )


__all__ = ["StreamAggregator"]
