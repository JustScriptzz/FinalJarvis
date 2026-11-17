"""REST API for the JARVIS dashboard."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ..core.discovery import discover_models
from ..core.errors import ConfigError, JarvisError, NotFoundError
from ..models.base import ProviderKind
from ..models.types import ModelDescriptor, SpeechRequest, TranscriptionRequest
from . import preferences
from .chat import ChatService, diagnostic
from .commands import CommandProcessor
from .preferences import PREVIEW_TEXT
from .schemas import (
    ChatBody,
    CommandBody,
    ConversationCreate,
    ConversationRename,
    CustomCoreRequest,
    DiscoverRequest,
    MemoryCreate,
    PreferencesPatch,
    ProviderKindName,
    RoutingModeRequest,
    SetupRequest,
    SwitchRequest,
    TTSBody,
    VoicePreviewRequest,
)
from .state import BROWSER_STT, BROWSER_TTS, ROUTING_MODE_KEY, AppState
from .ws import build_status, refresh_health

router = APIRouter(prefix="/api")


def get_state(request: Request) -> AppState:
    return request.app.state.jarvis


async def require_auth(request: Request) -> None:
    state = get_state(request)
    secret = state.config.server.api_token
    if secret is None or not secret.get_secret_value():
        return
    expected = secret.get_secret_value()
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
    token = (
        token
        or request.headers.get("x-jarvis-token", "")
        or request.query_params.get("token", "")
    )
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def _fail(exc: Exception, status: int = 400) -> JSONResponse:
    if isinstance(exc, JarvisError):
        return JSONResponse(
            status_code=status,
            content={"ok": False, "error": exc.message, "code": exc.code, "details": exc.details},
        )
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": str(exc), "code": type(exc).__name__},
    )


def _browser_descriptor() -> dict[str, Any]:
    return {
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
    }


# ---------------------------------------------------------------------------
# Status / system
# ---------------------------------------------------------------------------
@router.get("/status")
async def status(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    return {"ok": True, "data": await build_status(state)}


@router.get("/system")
async def system(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    snapshot = await state.telemetry.sample()
    return {"ok": True, "data": {"snapshot": snapshot, "history": state.telemetry.history()}}


@router.get("/config")
async def config_summary(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    return {"ok": True, "data": state.config.summary()}


@router.get("/version")
async def version(state: AppState = Depends(get_state)) -> dict[str, Any]:
    return {
        "ok": True,
        "data": {
            "name": state.config.app.name,
            "version": state.version,
            "boot_id": state.boot_id,
        },
    }


# ---------------------------------------------------------------------------
# Setup / models
# ---------------------------------------------------------------------------
@router.get("/setup")
async def get_setup(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    return {"ok": True, "data": state.setup_payload()}


@router.post("/setup")
async def save_setup(
    body: SetupRequest,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    first_completion = (
        state.setup is None
        and await state.db.get_setting(ROUTING_MODE_KEY) is None
    )
    try:
        await state.apply_setup(body.model_dump())
    except Exception as exc:
        return _fail(exc)
    if first_completion:
        # An explicit core choice governs the first session. Smart routing is
        # one toggle away in the core matrix, but it must not silently route
        # around the model the user just picked.
        state.routing_mode = "manual"
        state.router.config.llm.routing.mode = "manual"
        await state.db.set_setting(ROUTING_MODE_KEY, "manual")
    event = await state.db.add_event(
        f"Core selection engaged · {body.llm} / {body.tts}",
        kind="setup",
        data=body.model_dump(),
    )
    await state.publish(
        {
            "type": "setup.complete",
            "data": {
                "setup": state.setup_payload(),
                "status": await build_status(state),
                "selection": state.effective_selection(),
                "event": event,
            },
        }
    )
    await refresh_health(state)
    await state.publish(
        {"type": "health", "data": {"health": state.health, "at": state.health_updated_at}}
    )
    return JSONResponse(
        content={
            "ok": True,
            "data": {
                "setup": state.setup_payload(),
                "selection": state.effective_selection(),
                "status": await build_status(state),
            },
        }
    )


@router.get("/models")
async def models(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    described = state.router.describe()
    custom_ids = state.custom_ids()
    for kind, items in described["providers"].items():
        for item in items:
            item["custom"] = (kind, item["id"]) in custom_ids
    return {
        "ok": True,
        "data": {
            "models": described,
            "health": state.health,
            "health_updated_at": state.health_updated_at or None,
            "routing_mode": state.routing_mode,
            "setup": state.setup_payload(),
        },
    }


@router.post("/models/refresh")
async def models_refresh(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    await refresh_health(state)
    return {
        "ok": True,
        "data": {"health": state.health, "at": state.health_updated_at},
    }


@router.post("/models/switch")
async def models_switch(
    body: SwitchRequest,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    if body.kind not in {"llm", "stt", "tts"}:
        return _fail(ValueError("kind must be llm, stt or tts"))
    if body.kind == "tts" and body.provider_id == BROWSER_TTS:
        setup = dict(state.setup or {})
        setup["tts"] = BROWSER_TTS
        state.setup = setup
        await state.db.set_setting("setup", setup)
        descriptor: ModelDescriptor | dict[str, Any] = _browser_descriptor()
    elif body.kind == "stt" and body.provider_id == BROWSER_STT:
        setup = dict(state.setup or {})
        setup["stt"] = BROWSER_STT
        state.setup = setup
        await state.db.set_setting("setup", setup)
        descriptor = {
            **_browser_descriptor(),
            "kind": "stt",
            "model": "browser speech recognition",
            "local": False,
        }
    else:
        try:
            descriptor = await state.router.switch(ProviderKind(body.kind), body.provider_id)
        except Exception as exc:
            return _fail(exc)
        setup = dict(state.setup or {})
        setup[body.kind] = body.provider_id
        state.setup = setup
        await state.db.set_setting("setup", setup)
    event = await state.db.add_event(
        f"{body.kind.upper()} switched to {body.provider_id}",
        kind="model",
        data={"kind": body.kind, "provider_id": body.provider_id},
    )
    payload = descriptor if isinstance(descriptor, dict) else descriptor.model_dump()
    # The old core's model/voice name means nothing to the new one; drop it now
    # rather than sending an unknown name on the next turn.
    preferences = state.preferences
    if await state.reconcile_preferences():
        preferences = state.preferences
        await state.publish(
            {"type": "preferences.updated", "data": {"preferences": preferences}}
        )
    await state.publish(
        {
            "type": "model.switched",
            "data": {
                "kind": body.kind,
                "provider_id": body.provider_id,
                "model": payload.get("model"),
                "selection": state.effective_selection(),
                "setup": state.setup_payload(),
                "preferences": preferences,
                "event": event,
            },
        }
    )
    return JSONResponse(
        content={
            "ok": True,
            "data": {
                "descriptor": payload,
                "selection": state.effective_selection(),
                "preferences": preferences,
            },
        }
    )


# ---------------------------------------------------------------------------
# Custom cores
# ---------------------------------------------------------------------------
@router.post("/models/discover")
async def models_discover(
    body: DiscoverRequest,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    """List the models an endpoint actually serves, before saving anything."""
    try:
        result = await discover_models(body.base_url, api_key=body.api_key or None)
    except Exception as exc:
        return _fail(exc)
    return JSONResponse(content={"ok": True, "data": result})


@router.get("/models/custom")
async def list_custom_cores(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    return {"ok": True, "data": state.setup_payload()["custom"]}


@router.get("/models/{kind}/{provider_id}/catalog")
async def provider_catalog(
    kind: ProviderKindName,
    provider_id: str,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    """Everything one provider can be asked for: models and voices.

    The dashboard's pickers are driven by this, not by the YAML lists: a local
    Ollama server serves models the config never mentions, and a TTS server
    owns its voice catalogue. Both are read from the provider itself; a provider
    that cannot enumerate says so instead of the UI inventing entries.
    """
    config = state.router.provider_config(kind, provider_id)
    if config is None:
        return _fail(NotFoundError(f"Unknown {kind.upper()} core {provider_id!r}"), status=404)
    if not config.enabled:
        return _fail(
            ConfigError(f"{provider_id} is disabled; enable it before choosing a model"),
            status=409,
        )

    models: list[str] = []
    model_error: str | None = None
    voices: list[str] = []
    if kind == "llm":
        # A model list is whatever the endpoint serves, which is exactly what
        # the add-core probe already knows how to read.
        try:
            found = await discover_models(
                config.base_url or "", api_key=config.secret_key
            )
            models = found["models"]
        except Exception as exc:
            model_error = str(exc)
        if config.model and config.model not in models:
            models.insert(0, config.model)
    elif kind == "tts":
        voices = await state.router.list_voices(kind, provider_id)
        if config.voice and config.voice not in voices:
            voices.insert(0, config.voice)
    elif config.model:
        models = [config.model]

    return JSONResponse(
        content={
            "ok": True,
            "data": {
                "kind": kind,
                "provider_id": provider_id,
                "type": config.type,
                "models": models,
                "voices": voices,
                "current_model": config.model,
                "current_voice": getattr(config, "voice", None),
                "speed": getattr(config, "speed", None),
                "error": model_error,
            },
        }
    )


@router.get("/preferences")
async def get_preferences(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    return {"ok": True, "data": state.preferences}


@router.patch("/preferences")
async def patch_preferences(
    body: PreferencesPatch,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    try:
        updated = await state.apply_preferences(body.model_dump(exclude_unset=True))
    except Exception as exc:
        return _fail(exc)
    event = await state.db.add_event("Preferences updated", kind="model", data=updated)
    await state.publish(
        {"type": "preferences.updated", "data": {"preferences": updated, "event": event}}
    )
    return JSONResponse(content={"ok": True, "data": updated})


@router.post("/tts/preview")
async def preview_voice(
    body: VoicePreviewRequest,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> Response:
    """Synthesize one sentence with a specific voice, for the picker's test button.

    This deliberately ignores the browser-TTS short circuit: the point of a
    preview is to hear a *server* voice before committing to it.
    """
    text = (body.text or PREVIEW_TEXT).strip() or PREVIEW_TEXT
    provider_id = body.provider_id or state.router.selection.get("tts")
    if not provider_id or provider_id == BROWSER_TTS:
        return _fail(
            ConfigError(
                "Pick a server voice core first; the browser voice is tested in the browser"
            ),
            status=409,
        )
    try:
        result = await state.router.synthesize(
            SpeechRequest(
                text=text,
                voice=body.voice or None,
                speed=body.speed,
                provider_id=provider_id,
                timeout_s=120,
            )
        )
    except Exception as exc:
        return _fail(exc, status=502)
    return Response(
        content=result.audio,
        media_type=result.media_type,
        headers={
            "X-Jarvis-Provider": result.provider_id,
            "X-Jarvis-Voice": result.voice or "",
        },
    )


@router.post("/models/custom")
async def create_custom_core(
    body: CustomCoreRequest,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    payload = body.model_dump(exclude_none=True)
    kind = payload.pop("kind")
    provider_id = payload.pop("id")
    try:
        entry = await state.add_custom_core(kind, provider_id, payload)
    except Exception as exc:
        return _fail(exc)
    event = await state.db.add_event(
        f"{kind.upper()} core added · {provider_id}",
        kind="model",
        data={"kind": kind, "provider_id": provider_id, "type": entry.get("type")},
    )
    await refresh_health(state)
    await state.publish(
        {
            "type": "models.changed",
            "data": {
                "action": "added",
                "kind": kind,
                "provider_id": provider_id,
                "setup": state.setup_payload(),
                "health": state.health,
                "event": event,
            },
        }
    )
    return JSONResponse(
        content={"ok": True, "data": {"core": entry, "setup": state.setup_payload()}}
    )


@router.put("/models/custom/{kind}/{provider_id}")
async def update_custom_core(
    kind: ProviderKindName,
    provider_id: str,
    body: CustomCoreRequest,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    if kind != body.kind or provider_id != body.id:
        return _fail(ValueError("Path and body must name the same core"))
    if state.custom_store is None or state.custom_store.get(kind, provider_id) is None:
        return _fail(NotFoundError(f"Unknown custom {kind} core {provider_id!r}"), status=404)
    payload = body.model_dump(exclude_none=True)
    payload.pop("kind", None)
    payload.pop("id", None)
    try:
        entry = await state.add_custom_core(kind, provider_id, payload)
    except Exception as exc:
        return _fail(exc)
    await refresh_health(state)
    await state.publish(
        {
            "type": "models.changed",
            "data": {
                "action": "updated",
                "kind": kind,
                "provider_id": provider_id,
                "setup": state.setup_payload(),
                "health": state.health,
            },
        }
    )
    return JSONResponse(
        content={"ok": True, "data": {"core": entry, "setup": state.setup_payload()}}
    )


@router.delete("/models/custom/{kind}/{provider_id}")
async def delete_custom_core(
    kind: ProviderKindName,
    provider_id: str,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    try:
        removed = await state.remove_custom_core(kind, provider_id)
    except Exception as exc:
        return _fail(exc)
    if not removed:
        return _fail(NotFoundError(f"Unknown custom {kind} core {provider_id!r}"), status=404)
    event = await state.db.add_event(
        f"{kind.upper()} core removed · {provider_id}",
        kind="model",
        data={"kind": kind, "provider_id": provider_id},
    )
    # The reload dropped the removed core's health entry; re-probe what is left
    # so the matrix does not fall back to UNKNOWN.
    await refresh_health(state)
    await state.publish(
        {
            "type": "models.changed",
            "data": {
                "action": "removed",
                "kind": kind,
                "provider_id": provider_id,
                "setup": state.setup_payload(),
                "health": state.health,
                "event": event,
            },
        }
    )
    return JSONResponse(
        content={"ok": True, "data": {"removed": provider_id, "setup": state.setup_payload()}}
    )


@router.post("/routing/mode")
async def routing_mode(
    body: RoutingModeRequest,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    state.routing_mode = body.mode
    state.router.config.llm.routing.mode = body.mode
    await state.db.set_setting("routing_mode", body.mode)
    event = await state.db.add_event(f"Routing mode set to {body.mode}", kind="model")
    await state.publish(
        {"type": "routing.mode", "data": {"mode": body.mode, "event": event}}
    )
    return {"ok": True, "data": {"mode": body.mode}}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
@router.get("/conversations")
async def list_conversations(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    return {"ok": True, "data": await state.db.list_conversations(60)}


@router.post("/conversations")
async def create_conversation(
    body: ConversationCreate,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    conversation = await state.db.create_conversation(body.title or "New conversation")
    await state.publish({"type": "conversation.created", "data": conversation})
    return {"ok": True, "data": conversation}


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    conversation = await state.db.get_conversation(conversation_id)
    if conversation is None:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "Conversation not found"},
        )
    messages = await state.db.get_messages(conversation_id, 300)
    return JSONResponse(
        content={
            "ok": True,
            "data": {"conversation": conversation, "messages": messages},
        }
    )


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    body: ConversationRename,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    await state.db.rename_conversation(conversation_id, body.title)
    await state.publish(
        {
            "type": "conversation.renamed",
            "data": {"conversation_id": conversation_id, "title": body.title},
        }
    )
    return {"ok": True}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    await state.db.delete_conversation(conversation_id)
    await state.publish(
        {"type": "conversation.deleted", "data": {"conversation_id": conversation_id}}
    )
    return {"ok": True}


@router.post("/conversations/{conversation_id}/clear")
async def clear_conversation(
    conversation_id: str,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    await state.db.clear_conversation(conversation_id)
    await state.publish(
        {"type": "conversation.cleared", "data": {"conversation_id": conversation_id}}
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat / commands
# ---------------------------------------------------------------------------
@router.post("/chat")
async def chat(
    body: ChatBody,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    if not state.setup_complete:
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "error": "Select a language core and a voice core first.",
                "code": "setup_required",
                "setup": state.setup_payload(),
            },
        )
    service = ChatService(state)
    conversation = await service.conversation(body.conversation_id)
    try:
        await state.db.add_message(conversation["id"], "user", body.message)
        response = await service.complete(
            conversation["id"],
            body.message,
            provider_id=body.provider_id,
            profile=body.profile,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )
    except Exception as exc:
        info = diagnostic(exc, state)
        await state.db.add_event(
            f"Language core failure: {info['message'][:240]}",
            level="error",
            kind="chat",
            data=info,
        )
        return JSONResponse(status_code=502, content={"ok": False, **info})
    message = await state.db.add_message(
        conversation["id"],
        "assistant",
        response.text,
        provider_id=response.provider_id,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        meta={"kind": "chat", "routing": response.routing.model_dump()},
    )
    return JSONResponse(
        content={
            "ok": True,
            "data": {
                "conversation_id": conversation["id"],
                "message": message,
                "response": response.model_dump(),
            },
        }
    )


@router.post("/commands")
async def command(
    body: CommandBody,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    conversation = await ChatService(state).conversation(body.conversation_id)
    processor = CommandProcessor(state)
    result = await processor.handle(body.command, conversation["id"])
    message = None
    if result.persist:
        message = await state.db.add_message(
            conversation["id"],
            "assistant",
            result.text,
            meta={"kind": "command", "command": body.command},
        )
    return JSONResponse(
        content={
            "ok": not result.error,
            "data": {
                "text": result.text,
                "action": result.action,
                "action_data": result.data,
                "message": message,
                "conversation_id": conversation["id"],
            },
        }
    )


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
@router.get("/memory")
async def list_memory(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    return {"ok": True, "data": await state.db.list_memory(100)}


@router.post("/memory")
async def create_memory(
    body: MemoryCreate,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    memory = await state.db.add_memory(body.text, tags=body.tags)
    await state.publish({"type": "memory.created", "data": memory})
    return {"ok": True, "data": memory}


@router.delete("/memory/{memory_id}")
async def delete_memory(
    memory_id: str,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    deleted = await state.db.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    await state.publish({"type": "memory.deleted", "data": {"id": memory_id}})
    return {"ok": True}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@router.get("/events")
async def list_events(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    return {"ok": True, "data": await state.db.list_events(160)}


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------
@router.post("/tts")
async def tts(
    body: TTSBody,
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> Response:
    if (state.setup or {}).get("tts") == BROWSER_TTS:
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "error": "Browser speech synthesis is selected; no server TTS is required.",
                "code": "browser_tts",
            },
        )
    try:
        result = await state.router.synthesize(
            SpeechRequest(
                text=body.text,
                voice=body.voice or (state.preferences.get("tts") or {}).get("voice"),
                format=body.format,
                speed=body.speed if body.speed is not None
                else (state.preferences.get("tts") or {}).get("speed"),
                provider_id=body.provider_id,
                timeout_s=120,
            )
        )
    except Exception as exc:
        return _fail(exc, status=502)
    return Response(
        content=result.audio,
        media_type=result.media_type,
        headers={
            "X-Jarvis-Provider": result.provider_id,
            "X-Jarvis-Voice": result.voice or "",
            "X-Jarvis-Latency-Ms": f"{result.latency_ms:.1f}",
        },
    )


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    model: str | None = Form(None),
    state: AppState = Depends(get_state),
    _: None = Depends(require_auth),
) -> JSONResponse:
    audio = await file.read()
    # The dashboard's tuning is the default; an explicit form field wins, so a
    # caller can transcribe one clip in another language without changing it.
    tuned = preferences.stt_request_options(state.preferences)
    try:
        result = await state.router.transcribe(
            TranscriptionRequest(
                audio=audio,
                filename=file.filename or "audio.wav",
                media_type=file.content_type or "audio/wav",
                model=model or tuned["model"],
                language=language or tuned["language"],
            )
        )
    except Exception as exc:
        return _fail(exc, status=502)
    await state.db.add_event(
        "Audio transcribed",
        kind="speech",
        data={"provider_id": result.provider_id, "language": result.language},
    )
    return JSONResponse(content={"ok": True, "data": result.model_dump()})


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@router.get("/health")
async def health(
    state: AppState = Depends(get_state), _: None = Depends(require_auth)
) -> dict[str, Any]:
    return {
        "ok": True,
        "data": {
            "status": "ok",
            "uptime_s": round(time.time() - state.started_at, 1),
            "providers": state.health,
            "health_updated_at": state.health_updated_at or None,
        },
    }


@router.get("/commands")
async def commands() -> dict[str, Any]:
    from .commands import HELP

    return {"ok": True, "data": {"help": HELP}}


__all__ = ["get_state", "require_auth", "router"]
