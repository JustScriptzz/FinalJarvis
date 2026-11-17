"""Prompt construction for the chat path.

Local engines cache the longest identical prompt *prefix*, so a prompt that
changes near the top is re-evaluated from that point on every turn. On the
reference host (qwen2.5:3b) a ~900-token prompt costs 13-28 s to evaluate cold
and under a second when the prefix is reusable, so the shape of the prompt is a
latency property, not tidiness.

What these tests pin:

* the system message is a constant persona, with no clock or CPU figure in it,
* the user's own text is appended verbatim, never decorated,
* live host context is a separate trailing message, so it cannot corrupt the
  history that the next turn replays,
* and as a result, one turn's prompt is a strict prefix of the next turn's.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jarvis.core.config import JarvisConfig, ModelsConfig, load_config
from jarvis.models.router import ModelRouter
from jarvis.server.chat import (
    HISTORY_CHAR_BUDGET,
    SYSTEM_PROMPT,
    ChatService,
)
from jarvis.server.db import Database
from jarvis.server.state import AppState
from jarvis.server.telemetry import TelemetryService

from .conftest import llm_provider, models_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@pytest.fixture()
def service(tmp_path: Path) -> ChatService:
    config: JarvisConfig = load_config(CONFIG_PATH)
    config.app.data_dir = tmp_path
    config.models = ModelsConfig.model_validate(
        models_config(llm={"local-qwen": llm_provider()}, active_llm="local-qwen")
    )

    router = asyncio.run(ModelRouter(config.models).start())
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'jarvis.db'}")
    asyncio.run(db.connect())
    state = AppState(config=config, router=router, db=db, telemetry=TelemetryService())
    state.routing_mode = "manual"
    yield ChatService(state)
    asyncio.run(db.close())


def _prompt_text(messages: list[object]) -> str:
    return "\x00".join(str(getattr(m, "content", "")) for m in messages)


def test_system_message_is_the_stable_persona(service: ChatService) -> None:
    """No clock, no CPU figure: nothing that changes between two turns."""
    first, _ = asyncio.run(service.build_messages("conv", "Hello"))
    second, _ = asyncio.run(service.build_messages("conv", "Hello again"))

    assert first[0].role == "system"
    assert first[0].content == SYSTEM_PROMPT
    assert first[0].content == second[0].content


def test_user_text_is_appended_verbatim(service: ChatService) -> None:
    """The stored bytes and the sent bytes must be identical.

    Otherwise the next turn replays the history with different text at that
    position, which is what invalidated the cache before.
    """
    messages, _ = asyncio.run(service.build_messages("conv", "What is my CPU load?"))

    user_messages = [m for m in messages if m.role == "user"]
    assert [m.content for m in user_messages] == ["What is my CPU load?"]


def test_live_context_is_a_separate_trailing_message(service: ChatService) -> None:
    """Context never merges into a turn, so history stays replayable."""
    messages, _ = asyncio.run(service.build_messages("conv", "How is the CPU doing?"))

    assert messages[-1].role == "system"
    assert "LIVE CONTEXT" in (messages[-1].content or "")
    assert "Local time:" in (messages[-1].content or "")
    # No user turn carries it.
    assert all("LIVE CONTEXT" not in (m.content or "") for m in messages if m.role == "user")


def test_context_is_omitted_for_ordinary_conversation(service: ChatService) -> None:
    """Chatter should not carry a clock and a load figure on every turn."""
    messages, _ = asyncio.run(service.build_messages("conv", "Tell me a short joke."))

    assert all("LIVE CONTEXT" not in (m.content or "") for m in messages)
    assert messages[-1].role == "user"
    assert messages[-1].content == "Tell me a short joke."


@pytest.mark.parametrize(
    "text",
    [
        "Reply with exactly: systems nominal",
        "What do you make of the loading screen?",
        "Tell me about the hostess in that film",
        "Her grammar was excellent",
        "I dreamt of a spaceship",
    ],
)
def test_context_keywords_match_whole_words_only(service: ChatService, text: str) -> None:
    """Regression: a substring test matched "system" inside "systems".

    That attached a clock and a CPU figure to a large share of ordinary turns,
    because common English words contain the keywords. Those turns then diverged
    from the replayed history and cost a full prompt re-evaluation.
    """
    messages, _ = asyncio.run(service.build_messages("conv", text))

    assert all("LIVE CONTEXT" not in (m.content or "") for m in messages)


def test_context_attaches_when_the_turn_is_about_the_machine(service: ChatService) -> None:
    """The gate must not be so tight that real questions lose their data."""
    messages, _ = asyncio.run(service.build_messages("conv", "How much memory is free?"))

    assert "LIVE CONTEXT" in (messages[-1].content or "")


def test_history_budget_trims_only_the_oldest_turns(service: ChatService) -> None:
    """A long conversation must not overflow the context window.

    The budget is applied newest-first, so the recent turns survive intact and
    the prompt stays a growing prefix for as long as it fits.
    """
    conversation = asyncio.run(service.db.ensure_primary_conversation())["id"]
    filler = "x" * (HISTORY_CHAR_BUDGET // 3)

    for index in range(6):
        asyncio.run(
            _record(service, conversation, f"question {index}", filler)
        )

    messages, _ = asyncio.run(service.build_messages(conversation, "latest question"))

    total = sum(len(m.content or "") for m in messages)
    assert total <= HISTORY_CHAR_BUDGET + len("latest question") + len(SYSTEM_PROMPT) + 4096
    # The newest exchange is present; the oldest is not.
    assert any((m.content or "").startswith("question 5") for m in messages)
    assert not any((m.content or "").startswith("question 0") for m in messages)
    # Order is preserved.
    assert messages[0].role == "system"
    assert messages[-1].content == "latest question"


def test_one_turn_is_a_prefix_of_the_next(service: ChatService) -> None:
    """The property the cache depends on.

    After a completed exchange, the next prompt must begin with exactly the
    previous prompt's messages. This is what lets the engine skip the prefix
    instead of re-evaluating it.
    """
    conversation = asyncio.run(service.db.ensure_primary_conversation())["id"]

    first, _ = asyncio.run(service.build_messages(conversation, "Tell me a short joke."))
    asyncio.run(_record(service, conversation, "Tell me a short joke.", "A short joke, sir."))
    second, _ = asyncio.run(service.build_messages(conversation, "And another one."))

    assert _prompt_text(second).startswith(_prompt_text(first))


def test_context_turn_does_not_poison_later_turns(service: ChatService) -> None:
    """A turn that carried telemetry must still replay cleanly afterwards.

    The context message is transient: it is not stored, so the following turn
    replays the user text alone and the prefix still matches.
    """
    conversation = asyncio.run(service.db.ensure_primary_conversation())["id"]

    with_context, _ = asyncio.run(service.build_messages(conversation, "What is the CPU load?"))
    assert "LIVE CONTEXT" in _prompt_text(with_context)

    asyncio.run(_record(service, conversation, "What is the CPU load?", "It is low, sir."))
    later, _ = asyncio.run(service.build_messages(conversation, "And another thing."))

    # The stored turn is the bare question, with no context block in it.
    stored = [m.content for m in later if m.role == "user"]
    assert stored[0] == "What is the CPU load?"
    assert all("LIVE CONTEXT" not in (c or "") for c in stored)


async def _record(
    service: ChatService, conversation: str, user: str, assistant: str
) -> None:
    await service.db.add_message(conversation, "user", user)
    await service.db.add_message(conversation, "assistant", assistant)
