"""Configuration loading and validation.

YAML is the source of truth for provider definitions. Secrets are referenced as
``${ENV_VAR}`` and expanded at load time, so the file can be committed.
Runtime model selections live in ``data/runtime/models.json`` (see
``jarvis.core.runtime_store``) and are applied by the router, not here.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

from .errors import ConfigError

# ${VAR} | ${VAR:-default} | ${VAR:?message}
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([-?])([^}]*))?\}")

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _BACKEND_ROOT.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"

LLM_TYPES = frozenset({"openai_compatible", "anthropic", "gemini"})
STT_TYPES = frozenset({"faster_whisper", "openai_compatible", "deepgram", "parakeet"})
TTS_TYPES = frozenset({"openai_compatible", "elevenlabs", "cartesia", "piper", "f5_tts"})


# ---------------------------------------------------------------------------
# Environment expansion
# ---------------------------------------------------------------------------
def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, operator, argument = match.group(1), match.group(2), match.group(3)
        present = name in os.environ and os.environ[name] != ""
        if operator == "?":
            if not present:
                raise ConfigError(
                    f"Required environment variable {name!r} is not set: {argument or name}"
                )
            return os.environ[name]
        if operator == "-":
            return os.environ[name] if present else (argument or "")
        return os.environ.get(name, "")

    return _ENV_RE.sub(replace, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Model provider schema
# ---------------------------------------------------------------------------
class PricingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float = Field(0.0, ge=0)
    output_per_mtok: float = Field(0.0, ge=0)
    per_audio_minute: float = Field(0.0, ge=0)
    per_1k_chars: float = Field(0.0, ge=0)
    per_request: float = Field(0.0, ge=0)


class BaseProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    type: str
    model: str | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    api_version: str | None = None
    local: bool = False
    timeout_s: float = Field(60.0, gt=0)
    connect_timeout_s: float = Field(10.0, gt=0)
    max_concurrency: int = Field(4, ge=1)
    max_attempts: int = Field(3, ge=1, le=10)
    headers: dict[str, str] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    pricing: PricingConfig = Field(default_factory=PricingConfig)

    @property
    def secret_key(self) -> str | None:
        if self.api_key is None:
            return None
        value = self.api_key.get_secret_value()
        return value or None

    @property
    def is_local(self) -> bool:
        if self.local:
            return True
        if not self.base_url:
            return False
        host = (urlsplit(self.base_url).hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


class LLMProviderConfig(BaseProviderConfig):
    context_window: int = Field(32768, ge=256)
    max_output_tokens: int = Field(4096, ge=1)
    supports_tools: bool = True
    supports_vision: bool = False
    supports_json: bool = True
    supports_streaming: bool = True
    supports_stream_usage: bool = True

    @model_validator(mode="after")
    def _validate(self) -> LLMProviderConfig:
        if self.type not in LLM_TYPES:
            raise ValueError(
                f"unknown LLM provider type {self.type!r}; expected one of {sorted(LLM_TYPES)}"
            )
        if not self.model:
            raise ValueError("LLM providers require 'model'")
        if self.type == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible LLM providers require 'base_url'")
        if self.max_output_tokens > self.context_window:
            raise ValueError("max_output_tokens cannot exceed context_window")
        return self


class STTProviderConfig(BaseProviderConfig):
    language: str | None = None
    device: str = "auto"
    compute_type: str = "int8"
    model_path: str | None = None
    beam_size: int = Field(5, ge=1, le=10)
    vad_filter: bool = True
    timestamps: bool = True
    diarize: bool = False
    sample_rate: int | None = Field(None, ge=8000, le=192000)

    @model_validator(mode="after")
    def _validate(self) -> STTProviderConfig:
        if self.type not in STT_TYPES:
            raise ValueError(
                f"unknown STT provider type {self.type!r}; expected one of {sorted(STT_TYPES)}"
            )
        if not self.model:
            raise ValueError("STT providers require 'model'")
        if self.type == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible STT providers require 'base_url'")
        return self


class TTSProviderConfig(BaseProviderConfig):
    voice: str | None = None
    voices: list[str] = Field(default_factory=list)
    format: str = "wav"
    sample_rate: int | None = Field(None, ge=8000, le=192000)
    speed: float = Field(1.0, gt=0.1, le=4.0)
    language: str | None = None
    model_path: str | None = None
    config_path: str | None = None
    speaker: int | None = None
    reference_audio: str | None = None
    reference_text: str = ""
    device: str = "auto"
    binary: str = "piper"

    @model_validator(mode="after")
    def _validate(self) -> TTSProviderConfig:
        if self.type not in TTS_TYPES:
            raise ValueError(
                f"unknown TTS provider type {self.type!r}; expected one of {sorted(TTS_TYPES)}"
            )
        if self.type in {"openai_compatible", "elevenlabs", "cartesia"} and not self.model:
            raise ValueError(f"{self.type} TTS providers require 'model'")
        if self.type == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible TTS providers require 'base_url'")
        if self.type == "piper" and not self.model_path:
            raise ValueError("piper TTS providers require 'model_path'")
        if self.type == "f5_tts" and not self.reference_audio:
            raise ValueError("f5_tts providers require 'reference_audio' (voice prompt)")
        return self


class CircuitBreakerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_threshold: int = Field(5, ge=1)
    reset_timeout_s: float = Field(30.0, gt=0)
    half_open_max_calls: int = Field(1, ge=1)


class LLMRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["manual", "smart"] = "smart"
    profiles: dict[str, list[str]] = Field(default_factory=dict)
    fallback: list[str] = Field(default_factory=list)
    fast_threshold: float = Field(0.32, ge=0, le=1)
    escalate_threshold: float = Field(0.62, ge=0, le=1)

    @model_validator(mode="after")
    def _validate(self) -> LLMRoutingConfig:
        if self.fast_threshold >= self.escalate_threshold:
            raise ValueError("fast_threshold must be lower than escalate_threshold")
        for name, providers in self.profiles.items():
            if not providers:
                raise ValueError(f"routing profile {name!r} is empty")
        return self


class LLMSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: str
    routing: LLMRoutingConfig = Field(default_factory=LLMRoutingConfig)
    providers: dict[str, LLMProviderConfig]

    @model_validator(mode="after")
    def _validate(self) -> LLMSection:
        _require_provider("llm.active", self.active, self.providers)
        if not self.providers[self.active].enabled:
            raise ValueError(f"active LLM provider {self.active!r} is disabled")
        for profile, ids in self.routing.profiles.items():
            for provider_id in ids:
                _require_provider(f"routing profile {profile!r}", provider_id, self.providers)
        for provider_id in self.routing.fallback:
            _require_provider("llm.routing.fallback", provider_id, self.providers)
        return self


class FallbackSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: str
    fallback: list[str] = Field(default_factory=list)

    def validate_against(self, providers: Mapping[str, BaseProviderConfig], section: str) -> None:
        _require_provider(f"{section}.active", self.active, providers)
        if not providers[self.active].enabled:
            raise ValueError(f"active {section} provider {self.active!r} is disabled")
        for provider_id in self.fallback:
            _require_provider(f"{section}.fallback", provider_id, providers)


class STTSection(FallbackSection):
    providers: dict[str, STTProviderConfig]

    @model_validator(mode="after")
    def _validate(self) -> STTSection:
        self.validate_against(self.providers, "stt")
        return self


class TTSSection(FallbackSection):
    providers: dict[str, TTSProviderConfig]

    @model_validator(mode="after")
    def _validate(self) -> TTSSection:
        self.validate_against(self.providers, "tts")
        return self


def _require_provider(section: str, provider_id: str, providers: Mapping[str, Any]) -> None:
    if provider_id not in providers:
        raise ValueError(f"{section} references unknown provider {provider_id!r}")


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMSection
    stt: STTSection
    tts: TTSSection
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)

    @model_validator(mode="after")
    def _unique_ids(self) -> ModelsConfig:
        ids: list[str] = [
            *self.llm.providers,
            *self.stt.providers,
            *self.tts.providers,
        ]
        duplicates = sorted({provider_id for provider_id in ids if ids.count(provider_id) > 1})
        if duplicates:
            raise ValueError(f"provider ids must be unique across kinds: {duplicates}")
        return self


# ---------------------------------------------------------------------------
# Application schema
# ---------------------------------------------------------------------------
class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "JARVIS"
    environment: Literal["development", "production", "test"] = "development"
    timezone: str = "UTC"
    data_dir: Path = Path("./data")
    log_level: str = "INFO"
    log_json: bool = True


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(8080, ge=1, le=65535)
    public_base_url: str = "http://localhost:8080"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_token: SecretStr | None = None


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = "sqlite+aiosqlite:///./data/jarvis.db"
    echo: bool = False


class RedisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    url: str = "redis://localhost:6379/0"
    namespace: str = "jarvis"


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector_backend: Literal["chroma", "qdrant"] = "chroma"
    chroma_path: Path = Path("./data/chroma")
    qdrant_url: str = "http://localhost:6333"
    collection: str = "jarvis_memory"
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_dimensions: int = Field(768, ge=1)


class JarvisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    models: ModelsConfig
    # Owned by later phases; kept as structured mappings for now so one config
    # file can describe the whole system without silently dropping settings.
    telephony: dict[str, Any] = Field(default_factory=dict)
    channels: dict[str, Any] = Field(default_factory=dict)
    permissions: dict[str, Any] = Field(default_factory=dict)
    proactive: dict[str, Any] = Field(default_factory=dict)
    plugins: dict[str, Any] = Field(default_factory=dict)
    observability: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Redacted, JSON-friendly summary for logs and the dashboard."""
        return {
            "app": self.app.name,
            "environment": self.app.environment,
            "data_dir": str(self.app.data_dir),
            "routing_mode": self.models.llm.routing.mode,
            "active": {
                "llm": self.models.llm.active,
                "stt": self.models.stt.active,
                "tts": self.models.tts.active,
            },
            "providers": {
                "llm": {
                    provider_id: {
                        "type": provider.type,
                        "model": provider.model,
                        "enabled": provider.enabled,
                        "local": provider.is_local,
                    }
                    for provider_id, provider in self.models.llm.providers.items()
                },
                "stt": {
                    provider_id: {
                        "type": provider.type,
                        "model": provider.model,
                        "enabled": provider.enabled,
                        "local": provider.is_local,
                    }
                    for provider_id, provider in self.models.stt.providers.items()
                },
                "tts": {
                    provider_id: {
                        "type": provider.type,
                        "model": provider.model,
                        "enabled": provider.enabled,
                        "local": provider.is_local,
                    }
                    for provider_id, provider in self.models.tts.providers.items()
                },
            },
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _resolve_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def _absolutize_sqlite_url(root: Path, url: str) -> str:
    prefix = "sqlite+aiosqlite:///"
    if not url.startswith(prefix):
        return url
    raw_path = url[len(prefix) :]
    if raw_path.startswith("/") or raw_path.startswith(":memory:"):
        return url
    return f"{prefix}{(root / raw_path).resolve()}"


def _resolve_config_paths(config: JarvisConfig, root: Path) -> None:
    config.app.data_dir = _resolve_path(root, config.app.data_dir)
    config.memory.chroma_path = _resolve_path(root, config.memory.chroma_path)
    config.database.url = _absolutize_sqlite_url(root, config.database.url)

    for stt_provider in config.models.stt.providers.values():
        if stt_provider.model_path:
            stt_provider.model_path = str(_resolve_path(root, Path(stt_provider.model_path)))
        if stt_provider.options.get("download_root"):
            stt_provider.options["download_root"] = str(
                _resolve_path(root, Path(str(stt_provider.options["download_root"])))
            )
    for tts_provider in config.models.tts.providers.values():
        for attribute in ("model_path", "config_path", "reference_audio"):
            value = getattr(tts_provider, attribute)
            if value:
                setattr(
                    tts_provider,
                    attribute,
                    str(_resolve_path(root, Path(value))),
                )


def load_config(path: str | Path | None = None) -> JarvisConfig:
    """Load, expand, validate and path-resolve ``config.yaml``.

    Raises:
        ConfigError: for missing files, unset required env vars, invalid YAML,
            unknown providers, dangling references, or schema violations.
    """
    config_path = Path(path or os.environ.get("JARVIS_CONFIG") or DEFAULT_CONFIG_PATH).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")

    root = config_path.parent.parent
    # .env next to the repo root; process environment always wins.
    load_dotenv(root / ".env", override=False)

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    expanded = _expand_env(raw)
    try:
        config = JarvisConfig.model_validate(expanded)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid configuration in {config_path}:\n{_format_validation(exc)}"
        ) from exc

    _resolve_config_paths(config, root)
    return config


def _format_validation(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


def main() -> int:
    """``python -m jarvis.core.config [path]`` validates and prints a summary."""
    path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        config = load_config(path)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(config.summary(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
