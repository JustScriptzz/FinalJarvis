"""Provider contracts, shared HTTP behavior and error mapping."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel

from ..core.config import (
    BaseProviderConfig,
    LLMProviderConfig,
    STTProviderConfig,
    TTSProviderConfig,
)
from ..core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from .types import (
    ChatRequest,
    HealthStatus,
    ModelDescriptor,
    SpeechRequest,
    SpeechResult,
    ToolCallDelta,
    TranscriptionRequest,
    TranscriptionResult,
    Usage,
)


class ProviderKind(StrEnum):
    LLM = "llm"
    STT = "stt"
    TTS = "tts"


class LLMStreamChunk(BaseModel):
    """One normalized provider delta. Text and tool fragments can coexist."""

    text: str = ""
    tool_call: ToolCallDelta | None = None
    finish_reason: str | None = None
    usage: Usage | None = None


class BaseProvider[ConfigT: BaseProviderConfig](ABC):
    kind: ClassVar[ProviderKind]
    DEFAULT_BASE_URL: ClassVar[str | None] = None

    def __init__(
        self,
        provider_id: str,
        config: ConfigT,
        http: httpx.AsyncClient,
    ) -> None:
        self.id = provider_id
        self.config = config
        self._http = http
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    # -- identity ---------------------------------------------------------
    @property
    def base_url(self) -> str:
        value = self.config.base_url or self.DEFAULT_BASE_URL
        if not value:
            raise ProviderError(
                f"Provider {self.id!r} has no base_url",
                provider_id=self.id,
            )
        return value.rstrip("/")

    @property
    def model_name(self) -> str:
        return self.config.model or self.config.type

    def model_for(self, request: ChatRequest | SpeechRequest) -> str:
        return request.model or self.model_name

    def descriptor(self) -> ModelDescriptor:
        capabilities: list[str] = []
        context_window: int | None = None
        max_output_tokens: int | None = None
        voices: list[str] = []
        if isinstance(self.config, LLMProviderConfig):
            if self.config.supports_tools:
                capabilities.append("tools")
            if self.config.supports_vision:
                capabilities.append("vision")
            if self.config.supports_json:
                capabilities.append("json")
            if self.config.supports_streaming:
                capabilities.append("streaming")
            context_window = self.config.context_window
            max_output_tokens = self.config.max_output_tokens
        elif isinstance(self.config, STTProviderConfig):
            if self.config.timestamps:
                capabilities.append("timestamps")
            if self.config.diarize:
                capabilities.append("diarization")
        elif isinstance(self.config, TTSProviderConfig):
            voices = list(self.config.voices)
            if self.config.voice and self.config.voice not in voices:
                voices.insert(0, self.config.voice)
            capabilities.append(f"format:{self.config.format}")
        if self.config.is_local:
            capabilities.append("local")
        return ModelDescriptor(
            id=self.id,
            kind=self.kind.value,
            type=self.config.type,
            model=self.model_name,
            local=self.config.is_local,
            enabled=self.config.enabled,
            capabilities=capabilities,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            voices=voices,
        )

    # -- HTTP helpers -----------------------------------------------------
    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": "JARVIS/0.1",
            **self.config.headers,
        }
        if extra:
            headers.update(extra)
        return headers

    def _bearer_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = self._headers(extra)
        key = self.config.secret_key
        if key and not self.config.options.get("no_auth", False):
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _timeout(self, timeout_s: float | None = None) -> httpx.Timeout:
        total = timeout_s or self.config.timeout_s
        return httpx.Timeout(
            total,
            connect=min(self.config.connect_timeout_s, total),
            read=total,
            write=total,
            pool=total,
        )

    def _slot(self) -> asyncio.Semaphore:
        return self._semaphore

    async def _request(
        self,
        method: str,
        url: str,
        *,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            response = await self._http.request(
                method, url, timeout=self._timeout(timeout_s), **kwargs
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"{self.id} timed out after {timeout_s or self.config.timeout_s:.0f}s",
                provider_id=self.id,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"{self.id} connection failed: {exc}", provider_id=self.id
            ) from exc
        await self._ensure_ok(response)
        return response

    @asynccontextmanager
    async def _stream(
        self,
        method: str,
        url: str,
        *,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[httpx.Response]:
        try:
            async with self._http.stream(
                method, url, timeout=self._timeout(timeout_s), **kwargs
            ) as response:
                await self._ensure_ok(response)
                yield response
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"{self.id} timed out after {timeout_s or self.config.timeout_s:.0f}s",
                provider_id=self.id,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"{self.id} connection failed: {exc}", provider_id=self.id
            ) from exc

    async def _ensure_ok(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body = ""
        try:
            raw = await response.aread()
            body = raw.decode("utf-8", "replace")[:2000]
        except Exception:  # pragma: no cover - body already consumed / network abort
            pass
        message = _extract_error_message(body) or response.reason_phrase or "provider error"
        status = response.status_code
        details = {"body": body} if body else {}
        if status in (401, 403):
            raise ProviderAuthError(
                f"{self.id} rejected credentials (HTTP {status}): {message}",
                provider_id=self.id,
                status_code=status,
                details=details,
            )
        if status == 429:
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            raise ProviderRateLimitError(
                f"{self.id} rate limited: {message}",
                provider_id=self.id,
                status_code=status,
                retry_after_s=retry_after,
                details=details,
            )
        if status in (408, 504):
            raise ProviderTimeoutError(
                f"{self.id} timed out (HTTP {status}): {message}",
                provider_id=self.id,
                status_code=status,
                details=details,
            )
        if status >= 500:
            raise ProviderUnavailableError(
                f"{self.id} unavailable (HTTP {status}): {message}",
                provider_id=self.id,
                status_code=status,
                details=details,
            )
        raise ProviderBadRequestError(
            f"{self.id} rejected the request (HTTP {status}): {message}",
            provider_id=self.id,
            status_code=status,
            details=details,
        )

    # -- lifecycle / health ----------------------------------------------
    async def health(self) -> HealthStatus:
        if not self.config.enabled:
            return HealthStatus(status="disabled", detail="provider disabled")
        return HealthStatus(status="ok", detail="configured")

    async def aclose(self) -> None:
        """Release provider-owned resources. The HTTP client is router-owned."""
        return None


class LLMProvider(BaseProvider[LLMProviderConfig]):
    kind = ProviderKind.LLM

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        """Yield normalized deltas. Must not be called concurrently on self."""

    async def list_models(self) -> list[str]:
        return [self.model_name]


class STTProvider(BaseProvider[STTProviderConfig]):
    kind = ProviderKind.STT

    @abstractmethod
    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...


class TTSProvider(BaseProvider[TTSProviderConfig]):
    kind = ProviderKind.TTS

    @abstractmethod
    async def synthesize(self, request: SpeechRequest) -> SpeechResult: ...

    async def list_voices(self) -> list[str]:
        return list(self.config.voices)


def _extract_error_message(body: str) -> str:
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("detail") or error)
        if isinstance(error, str):
            return error
        for key in ("detail", "message", "error_description"):
            if payload.get(key):
                return str(payload[key])
    return body.strip()[:500]


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


__all__ = [
    "BaseProvider",
    "LLMProvider",
    "LLMStreamChunk",
    "ProviderKind",
    "STTProvider",
    "TTSProvider",
    "elapsed_ms",
]
