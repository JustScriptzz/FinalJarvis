"""WebSocket bridge: chat streaming, telemetry and live events."""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from ..core.errors import JarvisError
from .chat import ChatService, diagnostic
from .commands import CommandProcessor
from .db import Database
from .state import AppState


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def client_count(self) -> int:
        return len(self._connections)

    async def accept(self, websocket: WebSocket) -> None:
        """Accept the handshake without joining the broadcast group yet."""
        await websocket.accept()

    async def add(self, websocket: WebSocket) -> None:
        """Join the broadcast group after the private ``hello`` frame is sent."""
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def send(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        try:
            await websocket.send_json(payload)
        except (RuntimeError, WebSocketDisconnect):
            await self.disconnect(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._connections)
        if not targets:
            return
        results = await asyncio.gather(
            *(self._safe_send(target, payload) for target in targets),
            return_exceptions=True,
        )
        dead = [target for target, result in zip(targets, results, strict=True) if result is False]
        if dead:
            async with self._lock:
                for target in dead:
                    self._connections.discard(target)

    async def _safe_send(self, websocket: WebSocket, payload: dict[str, Any]) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except (RuntimeError, WebSocketDisconnect):
            return False


def _authorized(state: AppState, token: str | None) -> bool:
    secret = state.config.server.api_token
    if secret is None or not secret.get_secret_value():
        return True
    expected = secret.get_secret_value()
    return bool(token) and hmac.compare_digest(token or "", expected)


async def build_hello(state: AppState) -> dict[str, Any]:
    primary = await state.db.ensure_primary_conversation()
    telemetry = await state.telemetry.sample()
    return {
        "version": state.version,
        "boot_id": state.boot_id,
        "started_at": state.started_at,
        "status": await build_status(state),
        "setup": state.setup_payload(),
        "preferences": state.preferences,
        "system": telemetry,
        "history": state.telemetry.history(),
        "models": state.router.describe(),
        "health": state.health,
        "conversations": await state.db.list_conversations(30),
        "active_conversation": primary,
        "messages": await state.db.get_messages(primary["id"], 120),
        "events": await state.db.list_events(60),
        "memory": await state.db.list_memory(40),
    }


async def build_status(state: AppState) -> dict[str, Any]:
    telemetry = await state.telemetry.sample()
    counts: dict[str, int] = {}
    for kind in ("llm", "stt", "tts"):
        counts[kind] = len(state.router.available(kind))
    return {
        "app": state.config.app.name,
        "environment": state.config.app.environment,
        "version": state.version,
        "boot_id": state.boot_id,
        "started_at": state.started_at,
        "uptime_s": round(time.time() - state.started_at, 1),
        "host": telemetry["host"],
        "routing_mode": state.routing_mode,
        "selection": state.effective_selection(),
        "setup": state.setup,
        "setup_complete": state.setup_complete,
        "preferences": state.preferences,
        "clients": state.hub.client_count if state.hub is not None else 0,
        "conversations": await state.db.count_conversations(),
        "messages": await state.db.count_messages(),
        "providers": counts,
        "health_updated_at": state.health_updated_at or None,
    }


async def websocket_endpoint(websocket: WebSocket, token: str | None = None) -> None:
    state: AppState = websocket.app.state.jarvis
    if not _authorized(state, token):
        await websocket.close(code=4401, reason="Unauthorized")
        return

    hub: ConnectionManager = state.hub
    await hub.accept(websocket)
    try:
        await websocket.send_json({"type": "hello", "data": await build_hello(state)})
        await hub.add(websocket)
        while True:
            raw = await websocket.receive_text()
            if len(raw) > 200_000:
                await hub.send(
                    websocket,
                    {"type": "error", "data": {"message": "Payload too large"}},
                )
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await hub.send(
                    websocket,
                    {"type": "error", "data": {"message": "Invalid JSON frame"}},
                )
                continue
            if not isinstance(payload, dict):
                continue
            await _handle_frame(websocket, state, payload)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive socket guard
        try:
            await websocket.send_json(
                {"type": "error", "data": {"message": f"Socket failure: {exc}"}}
            )
        except Exception:
            pass
    finally:
        await hub.disconnect(websocket)


async def _handle_frame(websocket: WebSocket, state: AppState, payload: dict[str, Any]) -> None:
    frame_type = str(payload.get("type") or "")
    raw_data = payload.get("data")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    hub: ConnectionManager = state.hub

    if frame_type == "ping":
        await hub.send(websocket, {"type": "pong", "data": {"at": time.time()}})
        return

    if frame_type == "telemetry.request":
        await hub.send(
            websocket,
            {"type": "telemetry", "data": await state.telemetry.sample()},
        )
        return

    if frame_type == "models.refresh":
        await refresh_health(state)
        await hub.send(
            websocket,
            {
                "type": "health",
                "data": {"health": state.health, "at": state.health_updated_at},
            },
        )
        return

    if frame_type == "chat":
        request_id = str(data.get("request_id") or uuid4().hex[:12])
        message = str(data.get("message") or "").strip()
        if not message:
            await hub.send(
                websocket,
                {
                    "type": "chat.error",
                    "data": {
                        "request_id": request_id,
                        "message": "Empty transmission",
                    },
                },
            )
            return
        task: asyncio.Task[Any] = asyncio.create_task(
            _run_chat(
                websocket,
                state,
                request_id=request_id,
                message=message,
                conversation_id=data.get("conversation_id"),
            )
        )
        state.chat_tasks[request_id] = task

        def _done(_: asyncio.Task[Any], rid: str = request_id) -> None:
            state.chat_tasks.pop(rid, None)

        task.add_done_callback(_done)
        return

    if frame_type == "cancel":
        request_id = str(data.get("request_id") or "")
        pending = state.chat_tasks.get(request_id)
        if pending is not None:
            pending.cancel()
            await hub.send(
                websocket,
                {"type": "chat.cancelled", "data": {"request_id": request_id}},
            )
        return

    if frame_type == "command":
        request_id = str(data.get("request_id") or uuid4().hex[:12])
        command = str(data.get("command") or "").strip()
        conversation = await state.db.ensure_primary_conversation()
        await _run_command(
            websocket,
            state,
            request_id=request_id,
            command=command,
            conversation_id=data.get("conversation_id") or conversation["id"],
        )
        return

    await hub.send(
        websocket,
        {"type": "error", "data": {"message": f"Unknown frame type {frame_type!r}"}},
    )


async def _run_command(
    websocket: WebSocket,
    state: AppState,
    *,
    request_id: str,
    command: str,
    conversation_id: str,
) -> None:
    hub: ConnectionManager = state.hub
    processor = CommandProcessor(state)
    result = await processor.handle(command, conversation_id)
    message: dict[str, Any] | None = None
    if result.persist:
        message = await state.db.add_message(
            conversation_id,
            "assistant",
            result.text,
            meta={"kind": "command", "command": command},
        )
    if result.error:
        await state.db.add_event(
            f"Command failed: {command}",
            level="warning",
            kind="command",
        )
    await hub.send(
        websocket,
        {
            "type": "command.result",
            "data": {
                "request_id": request_id,
                "conversation_id": conversation_id,
                "text": result.text,
                "action": result.action,
                "action_data": result.data,
                "error": result.error,
                "message": message,
            },
        },
    )


def fault_text(info: dict[str, Any]) -> str:
    """Markdown body for a provider fault.

    The live frame and the persisted transcript row share this text, so a fault
    reads the same after a reload as it did the moment it happened.
    """
    hints = [str(hint) for hint in info.get("hints") or []]
    core = info.get("provider_id") or info.get("selected_llm") or "unknown"
    lines = [
        "**Language core unavailable.**",
        "",
        f"`{info.get('code') or 'provider_error'}` — "
        f"{info.get('message') or 'The provider did not respond.'}",
        "",
        f"Core: `{core}`",
    ]
    if hints:
        lines.extend(["", *(f"- {hint}" for hint in hints)])
    return "\n".join(lines)


async def _persist_fault(
    db: Database, conversation_id: str, info: dict[str, Any]
) -> dict[str, Any]:
    """Store a provider fault in the transcript so it survives a reload."""
    return await db.add_message(
        conversation_id,
        "error",
        fault_text(info),
        provider_id=info.get("provider_id"),
        model=info.get("provider_id") or info.get("selected_llm"),
        meta={"kind": "fault", **info},
    )


async def _run_chat(
    websocket: WebSocket,
    state: AppState,
    *,
    request_id: str,
    message: str,
    conversation_id: str | None,
) -> None:
    hub: ConnectionManager = state.hub
    db: Database = state.db
    service = ChatService(state)

    if not state.setup_complete:
        await hub.send(
            websocket,
            {"type": "setup.required", "data": state.setup_payload()},
        )
        return

    conversation = await service.conversation(conversation_id)
    cid = conversation["id"]

    if message.startswith("/"):
        await _run_command(
            websocket,
            state,
            request_id=request_id,
            command=message,
            conversation_id=cid,
        )
        return

    user_message = await db.add_message(cid, "user", message)
    await hub.send(
        websocket,
        {
            "type": "chat.start",
            "data": {
                "request_id": request_id,
                "conversation_id": cid,
                "user_message": user_message,
            },
        },
    )
    await db.add_event(
        "Language core engaged",
        kind="chat",
        data={"conversation_id": cid, "chars": len(message)},
    )

    chunks: list[str] = []
    response = None
    last_provider: str | None = None
    last_model: str | None = None
    try:
        async for event in service.stream(cid, message):
            if event.type == "start":
                last_provider = event.provider_id or last_provider
                last_model = event.model or last_model
                await hub.send(
                    websocket,
                    {
                        "type": "chat.meta",
                        "data": {
                            "request_id": request_id,
                            "provider_id": event.provider_id,
                            "model": event.model,
                            "attempt": event.attempt,
                            "routing": event.routing.model_dump()
                            if event.routing is not None
                            else None,
                        },
                    },
                )
            elif event.type == "text_delta" and event.text:
                chunks.append(event.text)
                await hub.send(
                    websocket,
                    {
                        "type": "chat.delta",
                        "data": {"request_id": request_id, "text": event.text},
                    },
                )
            elif event.type == "tool_call" and event.tool_call is not None:
                await hub.send(
                    websocket,
                    {
                        "type": "chat.tool",
                        "data": {
                            "request_id": request_id,
                            "tool_call": event.tool_call.model_dump(),
                        },
                    },
                )
            elif event.type == "usage" and event.usage is not None:
                await hub.send(
                    websocket,
                    {
                        "type": "chat.usage",
                        "data": {"request_id": request_id, "usage": event.usage.model_dump()},
                    },
                )
            elif event.type == "done":
                response = event.response
            elif event.type == "error" and event.error is not None:
                error = JarvisError(event.error.message)
                error.code = event.error.code
                error.retryable = event.error.retryable  # type: ignore[attr-defined]
                error.provider_id = event.error.provider_id  # type: ignore[attr-defined]
                raise error
    except asyncio.CancelledError:
        await db.add_event("Generation cancelled", level="warning", kind="chat")
        await hub.send(
            websocket,
            {"type": "chat.cancelled", "data": {"request_id": request_id}},
        )
        raise
    except Exception as exc:
        info = diagnostic(exc, state)
        fault = await _persist_fault(db, cid, info)
        event = await db.add_event(
            f"Language core failure: {info['message'][:240]}",
            level="error",
            kind="chat",
            data=info,
        )
        await hub.send(
            websocket,
            {
                "type": "chat.error",
                "data": {
                    "request_id": request_id,
                    "conversation_id": cid,
                    "message_id": fault["id"],
                    "error_message": fault,
                    **info,
                },
            },
        )
        await hub.broadcast({"type": "event", "data": event})
        return

    text = "".join(chunks) or (response.text if response is not None else "")
    if not text.strip():
        info = {
            "code": "empty_response",
            "message": "The active language core returned an empty response.",
            "provider_id": last_provider,
            "retryable": True,
            "selected_llm": (state.setup or {}).get("llm"),
            "hints": ["Try again, or switch cores from the model matrix."],
            "at": time.time(),
        }
        fault = await _persist_fault(db, cid, info)
        await hub.send(
            websocket,
            {
                "type": "chat.error",
                "data": {
                    "request_id": request_id,
                    "conversation_id": cid,
                    "message_id": fault["id"],
                    "error_message": fault,
                    **info,
                },
            },
        )
        return

    usage = response.usage if response is not None else None
    assistant_message = await db.add_message(
        cid,
        "assistant",
        text,
        provider_id=(response.provider_id if response else last_provider),
        model=(response.model if response else last_model),
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        cost_usd=response.cost_usd if response else 0.0,
        latency_ms=response.latency_ms if response else 0.0,
        meta={
            "kind": "chat",
            "routing": response.routing.model_dump() if response else None,
        },
    )
    await hub.send(
        websocket,
        {
            "type": "chat.done",
            "data": {
                "request_id": request_id,
                "conversation_id": cid,
                "message": assistant_message,
                "provider_id": assistant_message["provider_id"],
                "model": assistant_message["model"],
                "usage": usage.model_dump() if usage else None,
                "cost_usd": response.cost_usd if response else 0.0,
                "latency_ms": response.latency_ms if response else 0.0,
                "routing": response.routing.model_dump() if response else None,
            },
        },
    )
    event = await db.add_event(
        f"Response complete · {assistant_message['model'] or 'model'}",
        kind="chat",
        data={
            "conversation_id": cid,
            "latency_ms": response.latency_ms if response else 0.0,
            "tokens": usage.total_tokens if usage else 0,
        },
    )
    await hub.broadcast({"type": "event", "data": event})


async def refresh_health(state: AppState) -> None:
    try:
        state.health = await asyncio.wait_for(state.router.health(), timeout=25)
        state.health_updated_at = time.time()
    except Exception as exc:  # pragma: no cover - network failures are expected
        state.health_updated_at = time.time()
        state.health = {
            "error": {
                "probe": {
                    "health": {"status": "error", "detail": str(exc)},
                    "circuit": {"state": "unknown"},
                }
            }
        }


async def telemetry_loop(state: AppState, interval: float = 2.0) -> None:
    while True:
        try:
            snapshot = await state.telemetry.sample()
            await state.publish({"type": "telemetry", "data": snapshot})
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        await asyncio.sleep(interval)


async def health_loop(state: AppState, interval: float = 30.0) -> None:
    await asyncio.sleep(1.0)
    while True:
        await refresh_health(state)
        await state.publish(
            {
                "type": "health",
                "data": {"health": state.health, "at": state.health_updated_at},
            }
        )
        await asyncio.sleep(interval)


async def start_background_tasks(state: AppState) -> set[asyncio.Task[Any]]:
    tasks = {
        asyncio.create_task(telemetry_loop(state), name="jarvis-telemetry"),
        asyncio.create_task(health_loop(state), name="jarvis-health"),
    }
    return tasks


__all__ = [
    "ConnectionManager",
    "build_hello",
    "build_status",
    "health_loop",
    "refresh_health",
    "start_background_tasks",
    "telemetry_loop",
    "websocket_endpoint",
]


def asgi_websocket(
    app_state_name: str = "jarvis",
) -> Callable[[WebSocket], Awaitable[None]]:
    """Compatibility helper for tests that want a plain callable."""

    async def endpoint(websocket: WebSocket) -> None:
        await websocket_endpoint(websocket)

    return endpoint


__all__.append("asgi_websocket")
