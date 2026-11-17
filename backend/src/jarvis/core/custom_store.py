"""User-defined model providers, persisted outside ``config.yaml``.

The YAML file stays the editor's source of truth for the shipped providers, but
a personal assistant has to accept the endpoint the user actually runs. Cores
added from the dashboard live here, are merged into the model configuration on
load and on every change, and survive restarts without rewriting the annotated
YAML file.

API keys are stored in this file (mode 0600) rather than in the repository, and
are never returned by the HTTP API: descriptors report only whether a key is
set.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import SecretStr, ValidationError

from .config import (
    LLM_TYPES,
    STT_TYPES,
    TTS_TYPES,
    BaseProviderConfig,
    LLMProviderConfig,
    ModelsConfig,
    STTProviderConfig,
    TTSProviderConfig,
)
from .errors import ConfigError
from .logging import get_logger

log = get_logger(__name__)

_SCHEMA_VERSION = 1
_KINDS = ("llm", "stt", "tts")
_TYPES: dict[str, frozenset[str]] = {
    "llm": LLM_TYPES,
    "stt": STT_TYPES,
    "tts": TTS_TYPES,
}
# Fields a user is allowed to set from the dashboard. Anything else in
# ``config.yaml`` (pricing, options, headers) keeps its default here; power users
# who need those can still hand-write a YAML provider.
_WRITABLE: dict[str, tuple[str, ...]] = {
    "llm": (
        "type",
        "base_url",
        "api_key",
        "model",
        "context_window",
        "max_output_tokens",
        "supports_tools",
        "supports_vision",
        "supports_json",
        "local",
        "timeout_s",
    ),
    "stt": ("type", "base_url", "api_key", "model", "language", "local", "timeout_s"),
    "tts": (
        "type",
        "base_url",
        "api_key",
        "model",
        "voice",
        "format",
        "sample_rate",
        "language",
        "local",
        "timeout_s",
        # In-process voices need a local file, not a URL.
        "model_path",
        "reference_audio",
    ),
}

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,48}$")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([-?])([^}]*))?\}")
_SECRET_FIELD = "api_key"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)[:48] or "core"


def config_class(kind: str) -> Any:
    """The pydantic config class for a provider kind."""
    return {
        "llm": LLMProviderConfig,
        "stt": STTProviderConfig,
        "tts": TTSProviderConfig,
    }[kind]


def _expand(value: Any) -> Any:
    """Expand ``${VAR}`` inside values typed by the user."""

    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, operator, argument = match.group(1), match.group(2), match.group(3)
            present = name in os.environ and os.environ[name] != ""
            if operator == "?":
                if not present:
                    raise ConfigError(
                        f"Required environment variable {name!r} is not set"
                    )
                return os.environ[name]
            if operator == "-":
                return os.environ[name] if present else (argument or "")
            return os.environ.get(name, "")

        return _ENV_RE.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def build_provider(kind: str, raw: Mapping[str, Any]) -> Any:
    """Validate one provider definition and return its typed config."""
    if kind not in _KINDS:
        raise ConfigError(f"Unknown provider kind {kind!r}; expected one of {list(_KINDS)}")
    payload: dict[str, Any] = {"enabled": True}
    for field in _WRITABLE[kind]:
        if field in raw and raw[field] not in (None, ""):
            payload[field] = raw[field]
    provider_type = str(payload.get("type") or "")
    if provider_type not in _TYPES[kind]:
        raise ConfigError(
            f"{kind.upper()} core type {provider_type!r} is not supported; "
            f"expected one of {sorted(_TYPES[kind])}"
        )
    payload["type"] = provider_type
    payload = _expand(payload)
    try:
        return config_class(kind).model_validate(payload)
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'config'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(f"{kind.upper()} core rejected: {detail}") from exc


@dataclass(slots=True)
class CustomProvider:
    id: str
    kind: str
    type: str
    model: str
    base_url: str | None
    local: bool
    api_key_set: bool
    created_at: str
    updated_at: str

    def public(
        self, raw: Mapping[str, Any], config: BaseProviderConfig | None = None
    ) -> dict[str, Any]:
        """Descriptor-safe view: never leaks the key itself.

        Everything the edit dialog needs to prefill a form is included, but the
        secret is reduced to ``api_key_set``.
        """
        editable: dict[str, Any] = {}
        for field in _WRITABLE[self.kind]:
            value = getattr(config, field, None) if config is not None else None
            if value is None or value == [] or value == {}:
                value = raw.get(field)
            if field == _SECRET_FIELD:
                continue
            if isinstance(value, SecretStr):
                continue
            if isinstance(value, dict | list) or isinstance(value, bool | int | float | str):
                editable[field] = value
        return {
            "id": self.id,
            "kind": self.kind,
            "type": self.type,
            "model": self.model,
            "base_url": self.base_url,
            "local": self.local,
            "api_key_set": self.api_key_set,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "custom": True,
            "health": raw.get("health"),
            "editable": editable,
        }


class CustomProviderStore:
    """Read/write access to ``data/runtime/providers.json``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> CustomProviderStore:
        return cls(Path(data_dir) / "runtime" / "providers.json")

    # -- persistence ------------------------------------------------------
    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"version": _SCHEMA_VERSION, "providers": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("ignoring unreadable custom provider store", extra={"error": str(exc)})
            return {"version": _SCHEMA_VERSION, "providers": []}
        if not isinstance(payload, dict) or payload.get("version") != _SCHEMA_VERSION:
            log.warning("ignoring custom provider store with unknown schema")
            return {"version": _SCHEMA_VERSION, "providers": []}
        items = payload.get("providers")
        if not isinstance(items, list):
            return {"version": _SCHEMA_VERSION, "providers": []}
        return {
            "version": _SCHEMA_VERSION,
            "providers": [item for item in items if isinstance(item, dict)],
        }

    def _write(self, providers: list[dict[str, Any]]) -> None:
        payload = {
            "version": _SCHEMA_VERSION,
            "updated_at": _now(),
            "providers": providers,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".providers-", suffix=".json", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except OSError as exc:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise ConfigError(
                f"Could not persist custom cores to {self.path}: {exc}"
            ) from exc

    # -- queries ----------------------------------------------------------
    def load(self) -> list[dict[str, Any]]:
        return self._read()["providers"]

    def get(self, kind: str, provider_id: str) -> dict[str, Any] | None:
        for item in self.load():
            if item.get("kind") == kind and item.get("id") == provider_id:
                return item
        return None

    # -- mutations --------------------------------------------------------
    def save_provider(
        self, kind: str, provider_id: str, raw: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Create or update one core. An empty api_key keeps the stored one."""
        config = build_provider(kind, raw)
        if not _ID_RE.match(provider_id):
            raise ConfigError(
                "Core id must be 2-49 characters of lower-case letters, digits, "
                "dots, dashes or underscores"
            )
        providers = self.load()
        existing = next(
            (
                item
                for item in providers
                if item.get("kind") == kind and item.get("id") == provider_id
            ),
            None,
        )
        stored_key = str((existing or {}).get("api_key") or "")
        api_key = config.secret_key or stored_key
        entry: dict[str, Any] = {
            "id": provider_id,
            "kind": kind,
            "type": config.type,
            "model": _derive_model(config),
            "base_url": config.base_url,
            # A LAN endpoint is still your machine, not someone's cloud; badge
            # it LOCAL so routing and privacy hints stay truthful.
            "local": config.local or is_private_endpoint(config.base_url),
            "created_at": (existing or {}).get("created_at") or _now(),
            "updated_at": _now(),
        }
        for field in _WRITABLE[kind]:
            if field in {"type", "model", "base_url", "api_key", "local"}:
                continue
            value = getattr(config, field, None)
            if value is not None and value != [] and value != {}:
                entry[field] = value
        if api_key:
            entry["api_key"] = api_key
        providers = [
            item
            for item in providers
            if not (item.get("kind") == kind and item.get("id") == provider_id)
        ]
        providers.append(entry)
        providers.sort(key=lambda item: (str(item.get("kind")), str(item.get("id"))))
        self._write(providers)
        log.info(
            "custom core saved",
            extra={"kind": kind, "provider": provider_id, "type": config.type},
        )
        return entry

    def remove_provider(self, kind: str, provider_id: str) -> bool:
        providers = self.load()
        remaining = [
            item
            for item in providers
            if not (item.get("kind") == kind and item.get("id") == provider_id)
        ]
        if len(remaining) == len(providers):
            return False
        self._write(remaining)
        log.info("custom core removed", extra={"kind": kind, "provider": provider_id})
        return True

    # -- derived views ----------------------------------------------------
    def describe(self) -> list[dict[str, Any]]:
        described = []
        for item in self.load():
            kind = str(item.get("kind") or "")
            try:
                config = build_provider(kind, item)
            except ConfigError as exc:
                log.warning(
                    "ignoring invalid custom core",
                    extra={"provider": item.get("id"), "error": str(exc)},
                )
                continue
            public = CustomProvider(
                id=str(item.get("id")),
                kind=kind,
                type=config.type,
                model=_derive_model(config) or "",
                base_url=config.base_url,
                # Mirror what was persisted: a LAN endpoint counts as local.
                local=bool(item.get("local")) or config.is_local,
                api_key_set=bool(config.secret_key),
                created_at=str(item.get("created_at") or ""),
                updated_at=str(item.get("updated_at") or ""),
            )
            described.append(public.public(item, config))
        return described


def merge_custom_providers(
    models: ModelsConfig, items: Iterable[Mapping[str, Any]]
) -> ModelsConfig:
    """Return a new ``ModelsConfig`` with custom cores merged in.

    Invalid entries are skipped with a warning instead of taking the whole
    configuration down: one broken hand edit must not strand the assistant.
    """
    raw = models.model_dump()
    grid: dict[str, dict[str, Any]] = {
        kind: dict(raw[kind]["providers"]) for kind in _KINDS
    }
    for item in items:
        kind = str(item.get("kind") or "")
        provider_id = str(item.get("id") or "")
        if kind not in _KINDS or not _ID_RE.match(provider_id):
            log.warning("skipping custom core with invalid kind or id", extra={"id": provider_id})
            continue
        try:
            config = build_provider(kind, item)
        except ConfigError as exc:
            log.warning(
                "skipping invalid custom core",
                extra={"provider": provider_id, "error": str(exc)},
            )
            continue
        grid[kind][provider_id] = config.model_dump()
    for kind in _KINDS:
        raw[kind]["providers"] = grid[kind]
    try:
        return ModelsConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Custom cores conflict with the configuration: {exc}") from exc


def load_custom_models(
    config: ModelsConfig, data_dir: str | Path
) -> tuple[ModelsConfig, CustomProviderStore]:
    """Load the store and return a configuration that includes it."""
    load_dotenv(override=False)
    store = CustomProviderStore.in_data_dir(data_dir)
    return merge_custom_providers(config, store.load()), store


def _derive_model(config: BaseProviderConfig) -> str | None:
    """Name an in-process voice after the file or checkpoint it will use."""
    if config.model:
        return config.model
    if config.type == "piper":
        path = getattr(config, "model_path", None)
        if path:
            return Path(str(path)).name.removesuffix(".onnx")
    if config.type == "f5_tts":
        return "F5TTS_v1_Base"
    return None


def is_private_endpoint(base_url: str | None) -> bool:
    """True when the endpoint is loopback or a private/link-local address."""
    if not base_url:
        return False
    host = (urlsplit(base_url).hostname or "").lower()
    if host in {"localhost", "::1", "host.docker.internal"}:
        return True
    if host.startswith("127.") or host.startswith("10.") or host.startswith("192.168."):
        return True
    if re.match(r"^172\.(1[6-9]|2\d|3[01])\.", host):
        return True
    return host.startswith("169.254.")


__all__ = [
    "CustomProvider",
    "CustomProviderStore",
    "build_provider",
    "config_class",
    "is_private_endpoint",
    "load_custom_models",
    "merge_custom_providers",
    "slugify",
]
