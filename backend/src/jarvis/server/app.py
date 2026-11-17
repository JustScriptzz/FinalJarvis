"""FastAPI application factory for JARVIS."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from ..core.config import JarvisConfig, load_config
from ..core.custom_store import load_custom_models
from ..core.logging import configure_logging, get_logger
from ..core.runtime_store import RuntimeSelectionStore
from ..models.router import ModelRouter
from .db import Database
from .routes import router as api_router
from .state import ROUTING_MODE_KEY, SETUP_KEY, AppState
from .telemetry import TelemetryService
from .ws import (
    ConnectionManager,
    build_hello,
    refresh_health,
    start_background_tasks,
    websocket_endpoint,
)

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _database_url(config: JarvisConfig) -> str:
    """Resolve a relative SQLite URL against the configured data directory.

    ``config.database.url`` is authored relative to the repo root, but the CLI
    may be started from anywhere. Anchoring it to ``app.data_dir`` keeps the
    database next to the runtime selection store regardless of the cwd.
    """
    prefix = "sqlite+aiosqlite:///"
    url = config.database.url
    if not url.startswith(prefix):
        return url
    raw = url[len(prefix) :]
    if raw == ":memory:":
        return url
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (config.app.data_dir / path.name).resolve()
    return f"{prefix}{path}"


def create_app(
    config_path: str | Path | None = None,
    *,
    config: JarvisConfig | None = None,
) -> FastAPI:
    """Build the JARVIS web application.

    ``config`` is mainly for tests; production uses ``config_path`` or the
    ``JARVIS_CONFIG`` environment variable.
    """
    application = FastAPI(
        title="J.A.R.V.I.S.",
        description="Just A Rather Very Intelligent System — local-first personal AI",
        version="0.2.0",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=_lifespan,
    )
    application.state.config_path = config_path
    application.state.config_override = config
    application.state.jarvis = None
    application.state.tasks = set()

    application.add_middleware(GZipMiddleware, minimum_size=1024)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/__identity", include_in_schema=False)
    async def identity() -> dict[str, Any]:
        state: AppState | None = application.state.jarvis
        if state is None:
            return {"ok": False, "data": None}
        return {"ok": True, "data": await build_hello(state)}

    application.include_router(api_router)
    application.websocket("/ws")(websocket_endpoint)

    if STATIC_DIR.is_dir():
        application.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="hud")
    else:  # pragma: no cover - packaging guard
        @application.get("/", include_in_schema=False)
        async def missing_hud() -> dict[str, str]:
            return {"error": "Dashboard assets are missing from the package"}

    return application


@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    config_override: JarvisConfig | None = getattr(application.state, "config_override", None)
    config_path = getattr(application.state, "config_path", None)
    config = config_override or load_config(config_path)
    configure_logging(config.app.log_level, json_output=config.app.log_json)

    config.app.data_dir.mkdir(parents=True, exist_ok=True)

    db = Database(_database_url(config))
    await db.connect()

    runtime_store = RuntimeSelectionStore.in_data_dir(config.app.data_dir)
    # Cores added from the dashboard live beside the runtime selection; they are
    # merged before the router is built so they are first-class from boot.
    base_models = config.models
    models, custom_store = load_custom_models(base_models, config.app.data_dir)
    config.models = models
    router = ModelRouter(config.models, runtime_store=runtime_store)
    await router.start()

    state = AppState(
        config=config,
        router=router,
        db=db,
        telemetry=TelemetryService(),
        hub=ConnectionManager(),
        custom_store=custom_store,
        base_models=base_models,
    )
    application.state.jarvis = state

    # Routing mode survives restarts.
    mode = await db.get_setting(ROUTING_MODE_KEY, config.models.llm.routing.mode)
    if mode not in {"smart", "manual"}:
        mode = config.models.llm.routing.mode
    state.routing_mode = mode
    router.config.llm.routing.mode = mode

    # Model/voice tuning is independent of which core is engaged, so it loads
    # even when no setup has been completed yet.
    await state.load_preferences()

    # A saved setup is re-applied and re-validated. If a provider was removed
    # or disabled, the HUD asks the user to choose again instead of silently
    # falling back to a different core.
    saved_setup = await db.get_setting(SETUP_KEY)
    if isinstance(saved_setup, dict):
        try:
            await state.apply_setup(saved_setup)
            state.setup = {
                **saved_setup,
                "llm": saved_setup.get("llm"),
                "tts": saved_setup.get("tts"),
                "stt": saved_setup.get("stt"),
            }
        except Exception as exc:
            log.warning("saved model selection is no longer valid: %s", exc)
            state.setup = None
            await db.delete_setting(SETUP_KEY)

    # Now that the engaged cores are known, drop any tuning that belonged to a
    # core which was removed or replaced while the server was down.
    await state.reconcile_preferences()

    primary = await db.ensure_primary_conversation()
    await db.add_event(
        "J.A.R.V.I.S. core online",
        kind="boot",
        data={
            "host": state.telemetry._host["hostname"],
            "boot_id": state.boot_id,
            "primary_conversation": primary["id"],
        },
    )

    tasks = await start_background_tasks(state)
    application.state.tasks = tasks
    state.tasks = tasks

    # Prime telemetry and health without delaying readiness for more than a
    # moment; the HUD shows a "sweeping" state until the first probe returns.
    with contextlib.suppress(Exception):
        await asyncio.wait_for(refresh_health(state), timeout=12)

    log.info(
        "JARVIS online",
        extra={
            "host": config.server.host,
            "port": config.server.port,
            "setup": state.setup,
            "routing_mode": state.routing_mode,
        },
    )

    try:
        yield
    finally:
        for task in list(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await router.aclose()
        await db.close()
        application.state.jarvis = None
        log.info("JARVIS offline")


__all__ = ["STATIC_DIR", "create_app"]
