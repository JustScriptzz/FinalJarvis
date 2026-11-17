"""Conversation assembly and JARVIS persona for the web interface."""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from ..core.errors import JarvisError
from ..models.types import ChatMessage, ChatRequest, ChatResponse, SpeechRequest, SpeechResult
from . import preferences
from .db import Database
from .state import AppState

#: A word, for boundary-aware keyword matching. ``\b`` is not enough on its own
#: because the keywords include things like "cpu" that appear inside longer
#: identifiers; splitting on non-alphanumerics gives clean tokens.
_WORD = re.compile(r"[a-z0-9]+")

# How many past messages a turn includes.
#
# This is deliberately generous rather than minimal. Trimming history to the
# most recent N messages makes the window *slide*: once the conversation passes
# N, every new turn drops messages from the front, which changes the prompt at
# its very beginning. A local engine caches the longest identical prompt prefix,
# so sliding invalidates the whole cache on every turn -- measured on this host
# with qwen2.5:3b, the same prompt re-evaluated in 0.2 s when reusable and 25-39 s
# from scratch. A stable window keeps the prefix reusable as turns accumulate.
#
# 200 messages is a few thousand tokens for ordinary conversation, well inside a
# 32k context, and the per-message 8000-character cap bounds the worst case.
HISTORY_LIMIT = 200

# A hard ceiling on the characters of history in one prompt, about 12k tokens.
# ``HISTORY_LIMIT`` alone is not enough: 200 messages of 8000 characters each
# would not fit in any local context window, and an overflowing prompt is
# rejected by the engine rather than answered. The budget is applied from the
# newest turn backwards, so it trims the oldest messages first and leaves the
# recent conversation -- and therefore the reusable prefix -- intact.
HISTORY_CHAR_BUDGET = 48_000
MAX_MESSAGE_CHARS = 8_000

SYSTEM_PROMPT = """You are J.A.R.V.I.S. — Just A Rather Very Intelligent System — the personal
AI of the person you are speaking with.

Identity and voice:
- You are an unflappable British butler-engineer: precise, composed, quietly witty.
- Address the user as "sir" when it feels natural. Do not begin every sentence with it.
- You are loyal and candid. If something is impossible, dangerous or a bad idea, say so
  plainly and offer the better path.
- Never grovel, never pad, never roleplay actions you cannot perform.

Answer style:
- Lead with the answer, then the detail that matters.
- Use clean Markdown: short paragraphs, bullets, bold labels, fenced code when useful.
- Be as detailed as the question deserves, but never bury the point.
- When telemetry or model state is supplied below, treat it as ground truth and cite the
  relevant numbers.
- If a local model is offline, a provider is locked or a capability is missing, state that
  directly instead of pretending.
- End with one concrete next step only when it genuinely helps.

You are running inside the user's own JARVIS deployment. The interface around you shows
live host telemetry, the model matrix, conversation memory and a diagnostic log. You
cannot yet execute shell commands, control applications or send messages; those
subsystems arrive in later phases. Do not imply otherwise."""


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
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class ChatService:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.db: Database = state.db

    async def conversation(self, conversation_id: str | None) -> dict[str, Any]:
        if conversation_id:
            existing = await self.db.get_conversation(conversation_id)
            if existing is not None:
                return existing
        return await self.db.ensure_primary_conversation()

    async def build_messages(
        self, conversation_id: str, user_text: str
    ) -> tuple[list[ChatMessage], list[dict[str, Any]]]:
        telemetry = await self.state.telemetry.sample()
        memories = await self.db.search_memory(user_text, limit=5)
        if memories:
            await self.db.touch_memory([item["id"] for item in memories])
        history = await self.db.recent_messages(conversation_id, limit=HISTORY_LIMIT)
        # The turn is capped the same way history is. If it were not, a message
        # over the cap would be sent in full now and truncated when replayed from
        # history next turn, so the two would disagree at exactly that point and
        # the prefix being cached would break once per long message.
        current = user_text.strip()[:MAX_MESSAGE_CHARS]
        messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
        budget = HISTORY_CHAR_BUDGET
        kept: list[ChatMessage] = []
        # Walk backwards from the newest message so the budget can only ever drop
        # the *oldest* turns. While the conversation fits, nothing is dropped and
        # the prompt grows by appending -- which is exactly the case where the
        # engine's cached prefix keeps paying off.
        for item in reversed(history):
            if item["role"] not in {"user", "assistant"}:
                continue
            content = (item["content"] or "").strip()
            if not content:
                continue
            text = content[:MAX_MESSAGE_CHARS]
            if len(text) > budget and kept:
                break
            budget -= len(text)
            kept.append(ChatMessage(role=item["role"], content=text))
        messages.extend(reversed(kept))
        # The turn is appended as-is (already capped above). The caller persisted
        # the same text, so next turn's replay of this turn matches exactly.
        if not (messages[-1].role == "user" and messages[-1].content == current):
            messages.append(ChatMessage(role="user", content=current))
        messages.extend(self.context_messages(user_text, telemetry, memories))
        return messages, memories

    def context_messages(
        self,
        user_text: str,
        telemetry: dict[str, Any],
        memories: list[dict[str, Any]] | None = None,
    ) -> list[ChatMessage]:
        """Live context as its own trailing message, or nothing.

        Kept out of the user's turn on purpose. If the context were appended to
        the user message, the stored history would hold the raw text while the
        prompt held text plus context, so the next turn's replay would diverge
        at that point and the engine would re-evaluate the whole prompt. Sending
        it as a separate final message keeps every earlier message identical
        between turns, which is what makes the cached prefix reusable.

        It is also omitted unless the turn is about the machine, so ordinary
        conversation does not pay for it on every later turn.
        """
        if not self.context_is_relevant(user_text):
            return []
        return [ChatMessage(role="system", content=self.live_context(telemetry, memories))]

    #: Words that mean the turn is about the machine itself, so live host state
    #: is worth attaching. Matched on word boundaries: a plain substring test
    #: fires on "systems nominal", "status update" inside "status update please"
    #: and countless ordinary words, which would attach a clock and a CPU figure
    #: to nearly every turn and defeat the point of keeping it out.
    _CONTEXT_KEYWORDS = (
        "cpu",
        "memory",
        "ram",
        "load",
        "disk",
        "storage",
        "space",
        "uptime",
        "network",
        "telemetry",
        "system",
        "status",
        "health",
        "slow",
        "performance",
        "temperature",
        "battery",
        "host",
        "cores",
        "gpu",
        "swap",
    )

    @staticmethod
    def context_is_relevant(user_text: str) -> bool:
        """Whether this turn is worth attaching live host state to.

        With a local engine the prompt is cached from the start, so anything
        injected before the conversation continues is re-evaluated on every
        later turn. Attaching a clock and a CPU figure to every message
        therefore makes each turn slower for information that is usually
        irrelevant.

        The context is attached when the user asks about the machine, which is
        exactly when it is needed, and stays out of the prompt otherwise.
        """
        words = set(_WORD.findall(user_text.lower()))
        return any(word in words for word in ChatService._CONTEXT_KEYWORDS)

    async def system_prompt(self) -> str:
        """The stable persona.

        Deliberately free of anything that changes between turns; see
        ``build_messages`` for why that matters to latency.
        """
        return SYSTEM_PROMPT

    def live_context(
        self, telemetry: dict[str, Any], memories: list[dict[str, Any]] | None = None
    ) -> str:
        """Per-turn context, rendered for the end of the user's message.

        Kept separate from the system message so the system message stays
        byte-identical between turns and can be served from the engine's prompt
        cache; see ``build_messages``.
        """
        selection = self.state.router.selection
        setup = self.state.setup or {}
        now = datetime.now().astimezone()
        lines = [
            "--- LIVE CONTEXT (current, authoritative) ---",
            f"Local time: {now.strftime('%A, %d %B %Y, %H:%M:%S %Z')}",
            f"Host: {telemetry['host']['hostname']} · {telemetry['host']['platform']}",
            f"CPU: {telemetry['cpu']['percent']:.1f}% · "
            f"memory: {telemetry['memory']['percent']:.1f}% of "
            f"{_fmt_bytes(telemetry['memory']['total'])} · "
            f"disk: {telemetry['disk']['percent']:.1f}%",
            f"Uptime: {_fmt_duration(telemetry['uptime_s'])} · "
            f"network: {_fmt_bytes(telemetry['network']['recv_rate'])}/s in",
            f"Selected language core: {setup.get('llm') or selection.get('llm')} "
            f"(router: {selection.get('llm')})",
            f"Selected voice core: {setup.get('tts') or selection.get('tts')}",
            f"Routing mode: {self.state.routing_mode}",
        ]
        if memories:
            lines.append("Known facts about the user:")
            lines.extend(f"- {item['text']}" for item in memories)
        return "\n".join(lines)

    async def complete(
        self,
        conversation_id: str,
        user_text: str,
        *,
        provider_id: str | None = None,
        profile: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        messages, _ = await self.build_messages(conversation_id, user_text)
        tuned = preferences.llm_request_options(self.state.preferences)
        request = ChatRequest(
            messages=messages,
            provider_id=provider_id,
            profile=profile,
            model=model or tuned["model"],
            max_tokens=max_tokens or tuned["max_tokens"] or 2048,
            temperature=(
                temperature if temperature is not None
                else tuned["temperature"] if tuned["temperature"] is not None
                else 0.65
            ),
            timeout_s=240,
        )
        return await self.state.router.chat(request)

    async def stream(
        self,
        conversation_id: str,
        user_text: str,
        *,
        provider_id: str | None = None,
        profile: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> AsyncIterator[Any]:
        messages, _ = await self.build_messages(conversation_id, user_text)
        tuned = preferences.llm_request_options(self.state.preferences)
        request = ChatRequest(
            messages=messages,
            provider_id=provider_id,
            profile=profile,
            model=model or tuned["model"],
            max_tokens=max_tokens or tuned["max_tokens"] or 2048,
            temperature=(
                temperature if temperature is not None
                else tuned["temperature"] if tuned["temperature"] is not None
                else 0.65
            ),
            timeout_s=240,
        )
        async for event in self.state.router.stream_chat(request):
            yield event

    async def synthesize(
        self, text: str, *, voice: str | None = None, provider_id: str | None = None
    ) -> SpeechResult:
        tuned = preferences.tts_request_options(self.state.preferences)
        return await self.state.router.synthesize(
            SpeechRequest(
                text=text[:4000],
                voice=voice or tuned["voice"],
                provider_id=provider_id,
                speed=tuned["speed"],
                timeout_s=120,
            )
        )


def diagnostic(exc: Exception, state: AppState) -> dict[str, Any]:
    """Turn a provider failure into a useful, non-secret diagnostic."""
    active = state.router.selection.get("llm", "unset")
    detail = str(exc)
    code = getattr(exc, "code", type(exc).__name__)
    retryable = bool(getattr(exc, "retryable", False))
    setup = state.setup or {}
    hints: list[str] = []
    if "connect" in detail.lower() or "connection" in detail.lower():
        hints.append(
            "The endpoint is not reachable. For local models, start Ollama or your "
            "inference server."
        )
    if "not found" in detail.lower() or "404" in detail:
        hints.append("The model may not be pulled. For Ollama: `ollama pull <model>`.")
    if "api key" in detail.lower() or "401" in detail or "403" in detail:
        hints.append(
            "Add the provider key to `.env`, then enable the provider in "
            "`config/config.yaml`."
        )
    if not hints:
        hints.append("Open `/models` to inspect the active core, or `/setup` to choose another.")
    return {
        "code": code,
        "message": detail,
        "provider_id": getattr(exc, "provider_id", None) or active,
        "retryable": retryable,
        "selected_llm": setup.get("llm") or active,
        "hints": hints,
        "at": time.time(),
    }


__all__ = ["SYSTEM_PROMPT", "ChatService", "JarvisError", "diagnostic"]
