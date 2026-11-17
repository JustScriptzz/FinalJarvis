"""Deterministic JARVIS commands.

These are the systems the assistant can answer without a language model:
telemetry, model selection, memory and conversation control. They are real
operations, not canned chat replies.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..core.errors import ConfigError
from ..models.base import ProviderKind
from .db import Database, utc_now
from .state import BROWSER_STT, BROWSER_TTS, AppState


@dataclass
class CommandResult:
    text: str
    action: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    persist: bool = True
    error: bool = False


HELP = """### Command index

**Diagnostics**
- `/status` — full subsystem report
- `/health` — cached provider health and circuit state
- `/models` — model matrix and active selections
- `/time` — local and UTC time

**Configuration**
- `/switch <llm|stt|tts> <provider>` — hot-swap a model
- `/mode <smart|manual>` — routing policy
- `/setup` — reopen core selection
- `/voice <on|off>` — spoken replies

**Memory and archive**
- `/remember <fact>` — store a long-term fact
- `/memory` — list stored facts
- `/forget <id|all>` — remove facts
- `/conversations` — list archived threads
- `/new [title]` — start a new thread
- `/clear` — wipe the current thread

Everything else is routed to the active language model."""


def _fmt_bytes(value: float | int | None) -> str:
    if not value:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class CommandProcessor:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.db: Database = state.db

    async def handle(self, raw: str, conversation_id: str) -> CommandResult:
        text = raw.strip()
        if not text.startswith("/"):
            return CommandResult("Not a command.", error=True, persist=False)
        try:
            parts = shlex.split(text[1:])
        except ValueError as exc:
            return CommandResult(f"Malformed command: {exc}", error=True, persist=False)
        if not parts:
            return CommandResult(HELP)
        name = parts[0].lower()
        args = parts[1:]
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            return CommandResult(
                f"Unknown directive `/{name}`. Type `/help` for the command index.",
                error=True,
            )
        try:
            return await handler(args, conversation_id)
        except (ConfigError, ValueError, KeyError) as exc:
            return CommandResult(str(exc), error=True)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    async def cmd_help(self, args: list[str], conversation_id: str) -> CommandResult:
        return CommandResult(HELP)

    async def cmd_status(self, args: list[str], conversation_id: str) -> CommandResult:
        telemetry = await self.state.telemetry.sample()
        active = self.state.effective_selection()
        setup = self.state.setup or {}
        health = self.state.health
        ok = sum(
            1
            for group in health.values()
            for item in group.values()
            if (item.get("health") or {}).get("status") == "ok"
        )
        total = sum(len(group) for group in health.values())
        lines = [
            "### All systems report",
            "",
            f"**Power core** — {telemetry['cpu']['percent']:.1f}% CPU across "
            f"{telemetry['cpu']['count']} cores, load {telemetry['cpu']['load'][0]:.2f}.",
            f"**Memory** — {telemetry['memory']['percent']:.1f}% of "
            f"{_fmt_bytes(telemetry['memory']['total'])} committed.",
            f"**Storage** — {telemetry['disk']['percent']:.1f}% of "
            f"{_fmt_bytes(telemetry['disk']['total'])} on `{telemetry['disk']['mount']}`.",
            f"**Uplink** — {_fmt_bytes(telemetry['network']['recv_rate'])}/s in, "
            f"{_fmt_bytes(telemetry['network']['send_rate'])}/s out.",
            f"**Uptime** — {_fmt_duration(telemetry['uptime_s'])}.",
            "",
            f"**Language core** — `{active.get('llm', 'unset')}` "
            f"({setup.get('llm') or 'not selected'}), routing **{self.state.routing_mode}**.",
            f"**Voice core** — `{setup.get('tts') or 'not selected'}`.",
            f"**Audio input** — `{setup.get('stt') or 'not selected'}`.",
            f"**Providers** — {ok}/{total} reporting nominal.",
        ]
        return CommandResult("\n".join(lines))

    async def cmd_health(self, args: list[str], conversation_id: str) -> CommandResult:
        if not self.state.health:
            self.state.publish_soon({"type": "health.refresh.request"})
            return CommandResult(
                "Health sweep in progress. Ask again in a moment, sir."
            )
        lines = ["### Provider health", ""]
        for kind, group in self.state.health.items():
            lines.append(f"**{kind.upper()}**")
            for provider_id, item in group.items():
                status = (item.get("health") or {}).get("status", "unknown")
                detail = (item.get("health") or {}).get("detail", "")
                circuit = (item.get("circuit") or {}).get("state", "closed")
                lines.append(
                    f"- `{provider_id}` — **{status}** · circuit {circuit}"
                    + (f" · {detail}" if detail else "")
                )
            lines.append("")
        return CommandResult("\n".join(lines).strip())

    async def cmd_models(self, args: list[str], conversation_id: str) -> CommandResult:
        active = self.state.effective_selection()
        lines = ["### Model matrix", ""]
        for kind in ("llm", "stt", "tts"):
            lines.append(f"**{kind.upper()}** — active `{active.get(kind, 'unset')}`")
            for descriptor in self.state.router.available(kind):
                marker = "●" if descriptor.id == active.get(kind) else "○"
                state = "online" if descriptor.enabled else "locked"
                origin = "local" if descriptor.local else "cloud"
                lines.append(
                    f"- {marker} `{descriptor.id}` · {descriptor.model} · "
                    f"{origin}/{state}"
                )
            lines.append("")
        lines.append(f"TTS may also be `{BROWSER_TTS}`; STT may be `{BROWSER_STT}` (client-side).")
        return CommandResult("\n".join(lines))

    async def cmd_time(self, args: list[str], conversation_id: str) -> CommandResult:
        now = datetime.now().astimezone()
        utc = datetime.now(UTC)
        return CommandResult(
            f"**Local** — {now.strftime('%A, %d %B %Y · %H:%M:%S %Z')}\n\n"
            f"**UTC** — {utc.strftime('%Y-%m-%d · %H:%M:%S')}"
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    async def cmd_switch(self, args: list[str], conversation_id: str) -> CommandResult:
        if len(args) != 2:
            return CommandResult(
                "Usage: `/switch <llm|stt|tts> <provider-id>`", error=True
            )
        kind, provider_id = args[0].lower(), args[1]
        if kind not in {"llm", "stt", "tts"}:
            return CommandResult("Kind must be llm, stt or tts.", error=True)
        if kind == "tts" and provider_id == BROWSER_TTS:
            setup = dict(self.state.setup or {})
            setup["tts"] = BROWSER_TTS
            self.state.setup = setup
            await self.db.set_setting("setup", setup)
        elif kind == "stt" and provider_id == BROWSER_STT:
            setup = dict(self.state.setup or {})
            setup["stt"] = BROWSER_STT
            self.state.setup = setup
            await self.db.set_setting("setup", setup)
        else:
            descriptor = await self.state.router.switch(ProviderKind(kind), provider_id)
            setup = dict(self.state.setup or {})
            setup[kind] = provider_id
            self.state.setup = setup
            await self.db.set_setting("setup", setup)
            self.state.publish_soon(
                {
                    "type": "model.switched",
                    "data": {
                        "kind": kind,
                        "provider_id": provider_id,
                        "model": descriptor.model,
                        "selection": self.state.effective_selection(),
                    },
                }
            )
        await self.db.add_event(
            f"{kind.upper()} switched to {provider_id}",
            kind="model",
            data={"kind": kind, "provider_id": provider_id},
        )
        return CommandResult(f"{kind.upper()} core now routed through `{provider_id}`.")

    async def cmd_mode(self, args: list[str], conversation_id: str) -> CommandResult:
        if len(args) != 1 or args[0].lower() not in {"smart", "manual"}:
            return CommandResult("Usage: `/mode <smart|manual>`", error=True)
        mode = args[0].lower()
        self.state.routing_mode = mode
        self.state.router.config.llm.routing.mode = mode  # type: ignore[assignment]
        await self.db.set_setting("routing_mode", mode)
        await self.db.add_event(f"Routing mode set to {mode}", kind="model")
        self.state.publish_soon(
            {"type": "routing.mode", "data": {"mode": mode}}
        )
        if mode == "smart":
            return CommandResult(
                "Smart routing engaged. Routine work stays local; complex work escalates."
            )
        return CommandResult("Manual routing engaged. The selected core handles every request.")

    async def cmd_setup(self, args: list[str], conversation_id: str) -> CommandResult:
        return CommandResult(
            "Opening core selection. Choose a language core and a voice core, sir.",
            action="open_setup",
            data={"setup": self.state.setup_payload()},
        )

    async def cmd_voice(self, args: list[str], conversation_id: str) -> CommandResult:
        if len(args) != 1 or args[0].lower() not in {"on", "off"}:
            return CommandResult("Usage: `/voice <on|off>`", error=True)
        enabled = args[0].lower() == "on"
        return CommandResult(
            "Voice output enabled." if enabled else "Voice output muted.",
            action="voice_on" if enabled else "voice_off",
            data={"enabled": enabled},
            persist=False,
        )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    async def cmd_remember(self, args: list[str], conversation_id: str) -> CommandResult:
        text = " ".join(args).strip()
        if not text:
            return CommandResult("Usage: `/remember <fact>`", error=True)
        memory = await self.db.add_memory(text)
        await self.db.add_event("Memory engram stored", kind="memory", data=memory)
        self.state.publish_soon({"type": "memory.created", "data": memory})
        return CommandResult(f"Committed to long-term memory: *{text}*")

    async def cmd_memory(self, args: list[str], conversation_id: str) -> CommandResult:
        items = await self.db.list_memory(40)
        if not items:
            return CommandResult(
                "Long-term memory is empty. Use `/remember <fact>` to store something."
            )
        lines = ["### Long-term memory", ""]
        for item in items:
            lines.append(f"- `{item['id']}` — {item['text']}  · *{item['created_at'][:10]}*")
        return CommandResult("\n".join(lines))

    async def cmd_forget(self, args: list[str], conversation_id: str) -> CommandResult:
        if len(args) != 1:
            return CommandResult("Usage: `/forget <id|all>`", error=True)
        target = args[0]
        if target == "all":
            items = await self.db.list_memory(500)
            for item in items:
                await self.db.delete_memory(item["id"])
            await self.db.add_event("All memory engrams erased", level="warning", kind="memory")
            self.state.publish_soon({"type": "memory.cleared", "data": {}})
            return CommandResult(f"Erased {len(items)} memory engrams.")
        deleted = await self.db.delete_memory(target)
        if not deleted:
            return CommandResult(f"No engram with id `{target}`.", error=True)
        await self.db.add_event("Memory engram erased", kind="memory", data={"id": target})
        self.state.publish_soon({"type": "memory.deleted", "data": {"id": target}})
        return CommandResult(f"Erased engram `{target}`.")

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------
    async def cmd_new(self, args: list[str], conversation_id: str) -> CommandResult:
        title = " ".join(args).strip() or "New conversation"
        conversation = await self.db.create_conversation(title)
        self.state.publish_soon(
            {"type": "conversation.created", "data": conversation}
        )
        return CommandResult(
            f"New thread `{conversation['id']}` opened: **{conversation['title']}**.",
            action="switch_conversation",
            data={"conversation": conversation},
            persist=False,
        )

    async def cmd_clear(self, args: list[str], conversation_id: str) -> CommandResult:
        await self.db.clear_conversation(conversation_id)
        await self.db.add_event("Conversation buffer cleared", kind="conversation")
        self.state.publish_soon(
            {"type": "conversation.cleared", "data": {"conversation_id": conversation_id}}
        )
        return CommandResult(
            "Current thread cleared. The past is gone, sir.",
            action="clear",
            data={"conversation_id": conversation_id},
            persist=False,
        )

    async def cmd_conversations(self, args: list[str], conversation_id: str) -> CommandResult:
        items = await self.db.list_conversations(30)
        lines = ["### Archived threads", ""]
        for item in items:
            marker = "●" if item["id"] == conversation_id else "○"
            lines.append(
                f"- {marker} `{item['id']}` — **{item['title']}** · "
                f"{item['message_count']} messages · {item['updated_at'][:16]}"
            )
        return CommandResult("\n".join(lines))

    async def cmd_about(self, args: list[str], conversation_id: str) -> CommandResult:
        return CommandResult(
            "### J.A.R.V.I.S.\n\n"
            "**Just A Rather Very Intelligent System** — a local-first personal "
            "assistant with hot-swappable language, speech and voice cores.\n\n"
            f"- Version `{self.state.version}`\n"
            f"- Boot `{self.state.boot_id}`\n"
            f"- Host `{self.state.telemetry._host['hostname']}`\n"
            f"- Uptime `{_fmt_duration(datetime.now(UTC).timestamp() - self.state.started_at)}`\n"
            f"- Interface `{self.state.config.server.public_base_url}`"
        )

    async def cmd_remembered(self, args: list[str], conversation_id: str) -> CommandResult:
        return await self.cmd_memory(args, conversation_id)

    async def cmd_sys(self, args: list[str], conversation_id: str) -> CommandResult:
        return await self.cmd_status(args, conversation_id)


__all__ = ["HELP", "CommandProcessor", "CommandResult", "utc_now"]
