"""Shared application state and model-selection setup."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..core.config import JarvisConfig
from ..core.custom_store import (
    CustomProviderStore,
    build_provider,
    merge_custom_providers,
)
from ..core.errors import ConfigError
from ..models.router import ModelRouter
from . import preferences as prefs
from .db import Database
from .telemetry import TelemetryService

SETUP_KEY = "setup"
ROUTING_MODE_KEY = "routing_mode"
BROWSER_TTS = "browser"
BROWSER_STT = "browser"
SETUP_VERSION = 1


@dataclass
class AppState:
    config: JarvisConfig
    router: ModelRouter
    db: Database
    telemetry: TelemetryService
    custom_store: CustomProviderStore | None = None
    # The YAML-only view, kept so a removed custom core really disappears.
    base_models: Any = None
    hub: Any = None
    started_at: float = field(default_factory=time.time)
    version: str = "0.2.0"
    boot_id: str = field(default_factory=lambda: uuid4().hex[:10])
    health: dict[str, Any] = field(default_factory=dict)
    health_updated_at: float = 0.0
    routing_mode: str = "smart"
    setup: dict[str, Any] | None = None
    # Model/voice/persona tuning, independent of which core is engaged.
    preferences: dict[str, Any] = field(default_factory=lambda: prefs.normalize(None))
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    chat_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)

    @property
    def setup_complete(self) -> bool:
        return bool(self.setup and self.setup.get("llm") and self.setup.get("tts"))

    async def publish(self, event: dict[str, Any]) -> None:
        if self.hub is not None:
            await self.hub.broadcast(event)

    def publish_soon(self, event: dict[str, Any]) -> None:
        if self.hub is None:
            return
        task = asyncio.create_task(self.hub.broadcast(event))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def apply_setup(self, setup: dict[str, Any]) -> None:
        """Validate and apply a setup selection to the router."""
        llm = str(setup.get("llm") or "").strip()
        stt = str(setup.get("stt") or "").strip() or None
        tts = str(setup.get("tts") or "").strip()
        if not llm or not tts:
            raise ConfigError("Select both a language model and a voice model")

        await self._assert_selectable("llm", llm)
        if stt and stt != BROWSER_STT:
            await self._assert_selectable("stt", stt)
        if tts != BROWSER_TTS:
            await self._assert_selectable("tts", tts)

        await self.router.switch("llm", llm)
        if stt and stt != BROWSER_STT:
            await self.router.switch("stt", stt)
        if tts != BROWSER_TTS:
            await self.router.switch("tts", tts)

        self.setup = {
            "llm": llm,
            "stt": stt,
            "tts": tts,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "version": SETUP_VERSION,
        }
        await self.db.set_setting(SETUP_KEY, self.setup)

    async def _assert_selectable(self, kind: str, provider_id: str) -> None:
        descriptor = next(
            (item for item in self.router.available(kind) if item.id == provider_id), None
        )
        if descriptor is None:
            raise ConfigError(f"Unknown {kind.upper()} provider {provider_id!r}")
        if not descriptor.enabled:
            raise ConfigError(
                f"{kind.upper()} provider {provider_id!r} is disabled; "
                "add its credentials in .env and enable it in config/config.yaml"
            )

    def effective_selection(self) -> dict[str, str | None]:
        """Router selection with browser cores and keyboard-only input applied.

        The router only knows real providers, so a browser STT/TTS choice — or a
        deliberate ``null`` that means "keyboard only, no transcriber" — is
        tracked by the setup record. Once the user has completed a selection that
        record is authoritative; otherwise the router's own choice is reported.
        """
        if self.setup_complete:
            return {
                kind: (self.setup or {}).get(kind)
                for kind in ("llm", "stt", "tts")
            }
        return dict(self.router.selection)

    # -- preferences ------------------------------------------------------
    async def load_preferences(self) -> None:
        self.preferences = prefs.normalize(await self.db.get_setting(prefs.PREFERENCES_KEY))

    async def apply_preferences(self, patch: Any) -> dict[str, Any]:
        """Validate and persist a partial tuning update.

        ``None`` for a field clears that override; an omitted field is left
        untouched, so the dashboard can send only what the user moved. Each
        section is stamped with the core it was chosen for, which is what lets a
        later core switch drop only the now-meaningless names.
        """
        selection = self.effective_selection()
        merged = prefs.merge(self.preferences, patch)
        for kind in ("llm", "tts", "stt"):
            if isinstance(patch, dict) and isinstance(patch.get(kind), dict):
                merged[kind]["provider"] = selection.get(kind)
        self.preferences = merged
        await self.db.set_setting(prefs.PREFERENCES_KEY, self.preferences)
        return self.preferences

    async def reconcile_preferences(self) -> bool:
        """Clear tuning that belonged to a core which is no longer engaged."""
        record, changed = prefs.reconcile(self.preferences, self.effective_selection())
        if changed:
            self.preferences = record
            await self.db.set_setting(prefs.PREFERENCES_KEY, self.preferences)
        return changed

    async def reload_custom_models(self) -> None:
        """Rebuild the router's provider table from config plus custom cores."""
        if self.custom_store is None:
            return
        base = self.base_models if self.base_models is not None else self.config.models
        merged = merge_custom_providers(base, self.custom_store.load())
        # The merge starts from the pristine YAML, whose routing mode is whatever
        # was committed. The user's own choice lives in the settings table, so it
        # must be re-stamped on every rebuild — otherwise saving a core would
        # quietly put smart routing back and bypass the model they selected.
        merged.llm.routing.mode = (
            "smart" if self.routing_mode == "smart" else "manual"
        )
        self.config.models = merged
        await self.router.reload(merged)
        self.health = {}

    async def add_custom_core(
        self, kind: str, provider_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist a new endpoint and make it usable immediately."""
        if self.custom_store is None:
            raise ConfigError("Custom cores are not available on this server")
        existing = next(
            (item for item in self.router.available(kind) if item.id == provider_id),
            None,
        )
        if existing is not None and not self.custom_store.get(kind, provider_id):
            raise ConfigError(
                f"{kind.upper()} core {provider_id!r} already exists in config/config.yaml; "
                "choose a different id"
            )
        self.custom_store.save_provider(kind, provider_id, payload)
        await self.reload_custom_models()
        saved = next(
            (item for item in self._custom_descriptors() if item["id"] == provider_id),
            None,
        )
        if saved is None:
            raise ConfigError(f"{kind.upper()} core {provider_id!r} did not survive the reload")
        return saved

    async def remove_custom_core(self, kind: str, provider_id: str) -> bool:
        if self.custom_store is None:
            return False
        if not self.custom_store.remove_provider(kind, provider_id):
            return False
        in_use = (self.setup or {}).get(kind) == provider_id or (
            self.router.selection.get(kind) == provider_id
        )
        await self.reload_custom_models()
        if in_use and isinstance(self.setup, dict):
            # The chosen core just disappeared; send the user back to setup
            # rather than silently routing through a different model.
            await self.db.delete_setting(SETUP_KEY)
            self.setup = None
        return True

    def probe_core(self, kind: str, payload: dict[str, Any]) -> Any:
        """Validate a definition without saving it."""
        return build_provider(kind, payload)

    def custom_ids(self) -> set[tuple[str, str]]:
        """``{(kind, id)}`` for every core that came from the store."""
        if self.custom_store is None:
            return set()
        return {(item["kind"], item["id"]) for item in self.custom_store.describe()}

    def _custom_descriptors(self) -> list[dict[str, Any]]:
        """Custom cores with health attached and no secrets in the payload."""
        if self.custom_store is None:
            return []
        described = []
        for item in self.custom_store.describe():
            item["health"] = (
                self.health.get(item["kind"], {}).get(item["id"], {}).get("health")
            )
            described.append(item)
        return described

    def setup_payload(self) -> dict[str, Any]:
        providers: dict[str, list[dict[str, Any]]] = {}
        custom_ids = self.custom_ids()
        for kind in ("llm", "stt", "tts"):
            items = []
            for descriptor in self.router.available(kind):
                item = descriptor.model_dump()
                item["custom"] = (kind, descriptor.id) in custom_ids
                item["health"] = (
                    self.health.get(kind, {}).get(descriptor.id, {}).get("health")
                )
                item["circuit"] = (
                    self.health.get(kind, {}).get(descriptor.id, {}).get("circuit")
                )
                items.append(item)
            if kind == "stt":
                items.append(
                    {
                        "id": BROWSER_STT,
                        "kind": "stt",
                        "type": "browser",
                        "model": "browser speech recognition",
                        "local": False,
                        "enabled": True,
                        "capabilities": ["offline", "zero-setup"],
                        "context_window": None,
                        "max_output_tokens": None,
                        "voices": [],
                        "health": {
                            "status": "ok",
                            "detail": "Uses the browser's speech recognizer; availability varies",
                        },
                        "circuit": None,
                    }
                )
            if kind == "tts":
                items.append(
                    {
                        "id": BROWSER_TTS,
                        "kind": "tts",
                        "type": "browser",
                        "model": "system speech synthesis",
                        "local": True,
                        "enabled": True,
                        "capabilities": ["offline", "zero-setup"],
                        "context_window": None,
                        "max_output_tokens": None,
                        "voices": [],
                        "health": {
                            "status": "ok",
                            "detail": "Runs in the browser; always available",
                        },
                        "circuit": None,
                    }
                )
            providers[kind] = items
        return {
            "required": not self.setup_complete,
            "completed": self.setup_complete,
            "selection": self.setup,
            "providers": providers,
            "custom": self._custom_descriptors(),
            "routing_mode": self.routing_mode,
            "health_updated_at": self.health_updated_at or None,
        }
