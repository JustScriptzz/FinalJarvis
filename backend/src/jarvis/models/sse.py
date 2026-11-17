"""Minimal Server-Sent Events parser used by OpenAI, Anthropic and Gemini."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


async def iter_sse(response: httpx.Response) -> AsyncIterator[tuple[str | None, str]]:
    """Yield ``(event_name, data)`` pairs from an SSE response.

    Multi-line ``data:`` fields are joined with newlines. Comments and
    non-data fields are ignored. A trailing event without a blank line is
    still yielded.
    """
    event: str | None = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip() or None
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield event, "\n".join(data_lines)
