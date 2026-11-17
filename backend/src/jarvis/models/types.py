"""Normalized request/response types shared by every provider.

The router and agent only ever see these types. Provider-specific JSON stays
inside ``jarvis.models.llm.*`` / ``stt.*`` / ``tts.*``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "developer", "user", "assistant", "tool"]
FinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "error",
    "unknown",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Chat content
# ---------------------------------------------------------------------------
class TextPart(_Model):
    type: Literal["text"] = "text"
    text: str


class ImagePart(_Model):
    type: Literal["image"] = "image"
    # Exactly one of data/url should be set. Providers that cannot fetch URLs
    # (Anthropic, Gemini inline data) raise CapabilityError for url-only images.
    data: bytes | None = None
    url: str | None = None
    media_type: str = "image/png"
    detail: Literal["auto", "low", "high"] = "auto"

    @model_validator(mode="after")
    def _one_source(self) -> ImagePart:
        if bool(self.data) == bool(self.url):
            raise ValueError("image parts require exactly one of 'data' or 'url'")
        return self


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class ToolCall(_Model):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None
    parse_error: str | None = None


class ToolCallDelta(_Model):
    index: int = 0
    id: str | None = None
    name: str | None = None
    arguments: str = ""


class ChatMessage(_Model):
    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> ChatMessage:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool_calls")
        return self


class ToolDefinition(_Model):
    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolChoice(_Model):
    name: str
    disable_parallel_tool_use: bool = False


class ResponseFormat(_Model):
    type: Literal["text", "json_object", "json_schema"] = "text"
    name: str | None = None
    json_schema: dict[str, Any] | None = None
    strict: bool = True

    @model_validator(mode="after")
    def _schema_required(self) -> ResponseFormat:
        if self.type == "json_schema" and not self.json_schema:
            raise ValueError("json_schema response format requires 'json_schema'")
        return self


# ---------------------------------------------------------------------------
# Chat requests and responses
# ---------------------------------------------------------------------------
class Usage(_Model):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0

    @model_validator(mode="after")
    def _total(self) -> Usage:
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


class RoutingInfo(_Model):
    mode: Literal["manual", "smart"] = "manual"
    profile: str | None = None
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    attempts: int = 0
    selected_provider: str | None = None


class ChatRequest(_Model):
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] | ToolChoice | None = None
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    max_tokens: int | None = Field(None, ge=1)
    stop: list[str] | None = None
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    # Explicit selection. ``provider_id`` bypasses routing and fallback;
    # ``profile`` pins a named smart-routing profile.
    provider_id: str | None = None
    profile: str | None = None
    model: str | None = None
    timeout_s: float | None = Field(None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ChatRequest:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        if self.provider_id and self.profile:
            raise ValueError("provider_id and profile are mutually exclusive")
        return self

    def text(self) -> str:
        """Best-effort plain text of the conversation, for complexity scoring."""
        chunks: list[str] = []
        for message in self.messages:
            if isinstance(message.content, str):
                chunks.append(message.content)
            elif isinstance(message.content, list):
                chunks.extend(part.text for part in message.content if isinstance(part, TextPart))
            for call in message.tool_calls:
                chunks.append(f"{call.name}({call.arguments})")
        return "\n".join(chunks)

    def requires_vision(self) -> bool:
        return any(
            isinstance(part, ImagePart)
            for message in self.messages
            if isinstance(message.content, list)
            for part in message.content
        )


class ChatResponse(_Model):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "stop"
    usage: Usage = Field(default_factory=Usage)
    provider_id: str = ""
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    routing: RoutingInfo = Field(default_factory=RoutingInfo)


# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------
class TranscriptionRequest(_Model):
    audio: bytes = Field(min_length=1)
    filename: str = "audio.wav"
    media_type: str = "audio/wav"
    # A model name the core also serves, overriding its configured default.
    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    temperature: float | None = Field(None, ge=0, le=1)
    timestamps: bool = True
    provider_id: str | None = None
    timeout_s: float | None = Field(None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptSegment(_Model):
    id: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    speaker: str | None = None
    confidence: float | None = None


class TranscriptionResult(_Model):
    text: str = ""
    language: str | None = None
    duration_s: float | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    provider_id: str = ""
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------
class SpeechRequest(_Model):
    text: str = Field(min_length=1)
    voice: str | None = None
    model: str | None = None
    format: str | None = None
    sample_rate: int | None = Field(None, ge=8000, le=192000)
    speed: float | None = Field(None, gt=0.1, le=4.0)
    instructions: str | None = None
    language: str | None = None
    provider_id: str | None = None
    timeout_s: float | None = Field(None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeechResult(_Model):
    audio: bytes
    media_type: str
    format: str
    sample_rate: int | None = None
    duration_s: float | None = None
    voice: str | None = None
    provider_id: str = ""
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------
class ModelDescriptor(_Model):
    id: str
    kind: Literal["llm", "stt", "tts"]
    type: str
    model: str
    local: bool
    enabled: bool
    capabilities: list[str] = Field(default_factory=list)
    context_window: int | None = None
    max_output_tokens: int | None = None
    voices: list[str] = Field(default_factory=list)


class HealthStatus(_Model):
    status: Literal["ok", "degraded", "error", "disabled"] = "ok"
    detail: str = ""
    latency_ms: float | None = None


class ErrorInfo(_Model):
    code: str
    message: str
    retryable: bool = False
    provider_id: str | None = None
    status_code: int | None = None


class StreamEvent(_Model):
    type: Literal[
        "start",
        "text_delta",
        "tool_call_delta",
        "tool_call",
        "usage",
        "done",
        "error",
    ]
    provider_id: str | None = None
    model: str | None = None
    attempt: int = 0
    routing: RoutingInfo | None = None
    text: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    error: ErrorInfo | None = None
    response: ChatResponse | None = None
