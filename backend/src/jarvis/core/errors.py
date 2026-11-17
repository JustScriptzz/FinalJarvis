"""Typed errors shared by the router, providers and (later) the API layer."""

from __future__ import annotations

from typing import Any


class JarvisError(Exception):
    """Base class for all expected JARVIS failures."""

    code: str = "jarvis_error"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class ConfigError(JarvisError):
    code = "config_error"


class NotFoundError(JarvisError):
    """The named core, conversation or record does not exist."""

    code = "not_found"


class CapabilityError(JarvisError):
    """The selected model cannot satisfy the request (tools, vision, JSON...)."""

    code = "capability_error"


class AudioError(JarvisError):
    code = "audio_error"


class NoProviderAvailableError(JarvisError):
    code = "no_provider_available"


class ProviderError(JarvisError):
    """A model provider failed. ``retryable`` drives retry/fallback decisions."""

    code = "provider_error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider_id: str | None = None,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.provider_id = provider_id
        self.status_code = status_code
        self.retry_after_s = retry_after_s

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "provider_id": self.provider_id,
                "retryable": self.retryable,
                "status_code": self.status_code,
            }
        )
        return data


class ProviderAuthError(ProviderError):
    code = "provider_auth_error"
    retryable = False


class ProviderBadRequestError(ProviderError):
    code = "provider_bad_request"
    retryable = False


class ProviderRateLimitError(ProviderError):
    code = "provider_rate_limit"
    retryable = True


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout"
    retryable = True


class ProviderUnavailableError(ProviderError):
    code = "provider_unavailable"
    retryable = True


class CircuitOpenError(ProviderUnavailableError):
    code = "circuit_open"
    retryable = False
