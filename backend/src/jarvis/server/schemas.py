"""Request/response models for the JARVIS HTTP and WebSocket API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ProviderKindName = Literal["llm", "stt", "tts"]
RoutingMode = Literal["manual", "smart"]


class SetupRequest(BaseModel):
    llm: str = Field(min_length=1)
    stt: str | None = None
    tts: str = Field(min_length=1)


class CustomCoreRequest(BaseModel):
    """A user-supplied endpoint. Only these fields are writable from the HUD."""

    kind: ProviderKindName
    id: str = Field(min_length=2, max_length=49, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    type: str = Field(min_length=1, max_length=40)
    # Optional for in-process voices, which take their name from a local file.
    model: str | None = Field(default=None, max_length=200)
    base_url: str | None = Field(default=None, max_length=500)
    api_key: str | None = Field(default=None, max_length=4000)
    local: bool | None = None
    context_window: int | None = Field(default=None, ge=256, le=10_000_000)
    max_output_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_json: bool | None = None
    voice: str | None = Field(default=None, max_length=120)
    format: str | None = Field(default=None, max_length=20)
    sample_rate: int | None = Field(default=None, ge=8000, le=192000)
    language: str | None = Field(default=None, max_length=20)
    # In-process TTS voices: piper needs an .onnx voice, f5_tts a prompt clip.
    model_path: str | None = Field(default=None, max_length=500)
    reference_audio: str | None = Field(default=None, max_length=500)
    timeout_s: float | None = Field(default=None, gt=0, le=3600)


class DiscoverRequest(BaseModel):
    base_url: str = Field(min_length=4, max_length=500)
    api_key: str | None = Field(default=None, max_length=4000)


class SwitchRequest(BaseModel):
    kind: ProviderKindName
    provider_id: str = Field(min_length=1)


class RoutingModeRequest(BaseModel):
    mode: RoutingMode


class PreferencesPatch(BaseModel):
    """Partial tuning update. Omitted sections are left untouched."""

    llm: dict[str, Any] | None = None
    stt: dict[str, Any] | None = None
    tts: dict[str, Any] | None = None


class VoiceListRequest(BaseModel):
    provider_id: str | None = None


class VoicePreviewRequest(BaseModel):
    provider_id: str | None = None
    voice: str | None = Field(default=None, max_length=200)
    text: str | None = Field(default=None, max_length=600)
    speed: float | None = Field(default=None, gt=0.1, le=4.0)


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class MemoryCreate(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=16)


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=32000)
    conversation_id: str | None = None
    provider_id: str | None = None
    profile: str | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0, le=2)


class TTSBody(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    voice: str | None = None
    format: str | None = None
    provider_id: str | None = None
    speed: float | None = Field(default=None, gt=0.1, le=4.0)


class CommandBody(BaseModel):
    command: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class ApiMessage(BaseModel):
    ok: bool = True
    data: Any = None
    error: str | None = None
