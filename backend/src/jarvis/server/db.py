"""SQLite persistence for conversations, messages, events and memory.

The server uses a single ``aiosqlite`` connection. All access is serialized
through an asyncio lock: SQLite is fast enough for a personal assistant and a
single writer removes the usual "database is locked" failure mode.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    pinned        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    provider_id     TEXT,
    model           TEXT,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0,
    latency_ms      REAL NOT NULL DEFAULT 0,
    meta            TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    level      TEXT NOT NULL DEFAULT 'info',
    kind       TEXT NOT NULL DEFAULT 'system',
    message    TEXT NOT NULL,
    data       TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at DESC);

CREATE TABLE IF NOT EXISTS memory (
    id           TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    tags         TEXT NOT NULL DEFAULT '[]',
    source       TEXT NOT NULL DEFAULT 'user',
    created_at   TEXT NOT NULL,
    last_used_at TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _ensure_parent(path: str) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _row(row: aiosqlite.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _message(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    try:
        item["meta"] = json.loads(item.get("meta") or "{}")
    except json.JSONDecodeError:
        item["meta"] = {}
    return item


class Database:
    """Small async repository over SQLite."""

    def __init__(self, url: str) -> None:
        prefix = "sqlite+aiosqlite:///"
        if not url.startswith(prefix):
            raise ValueError(f"Only sqlite+aiosqlite URLs are supported, got {url!r}")
        self.path = url[len(prefix) :]
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self.path != ":memory:":
            await asyncio.to_thread(_ensure_parent, self.path)
        conn = await aiosqlite.connect(self.path)
        conn.row_factory = aiosqlite.Row
        await conn.executescript(_SCHEMA)
        await conn.commit()
        self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected")
        return self._conn

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------
    async def create_conversation(self, title: str = "New conversation") -> dict[str, Any]:
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        stamp = utc_now()
        async with self._lock:
            await self.conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title.strip() or "New conversation", stamp, stamp),
            )
            await self.conn.commit()
        return {
            "id": conversation_id,
            "title": title.strip() or "New conversation",
            "created_at": stamp,
            "updated_at": stamp,
            "message_count": 0,
            "pinned": 0,
        }

    async def ensure_primary_conversation(self) -> dict[str, Any]:
        rows = await self.fetchall(
            "SELECT * FROM conversations ORDER BY pinned DESC, updated_at DESC LIMIT 1"
        )
        if rows:
            return rows[0]
        return await self.create_conversation("Primary Interface")

    async def list_conversations(self, limit: int = 60) -> list[dict[str, Any]]:
        return await self.fetchall(
            "SELECT * FROM conversations ORDER BY pinned DESC, updated_at DESC LIMIT ?",
            (limit,),
        )

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        return await self.fetchone(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        )

    async def rename_conversation(self, conversation_id: str, title: str) -> None:
        async with self._lock:
            await self.conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip() or "Untitled", utc_now(), conversation_id),
            )
            await self.conn.commit()

    async def delete_conversation(self, conversation_id: str) -> None:
        async with self._lock:
            await self.conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            await self.conn.commit()

    async def clear_conversation(self, conversation_id: str) -> None:
        async with self._lock:
            await self.conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            await self.conn.execute(
                "UPDATE conversations SET message_count = 0, updated_at = ? WHERE id = ?",
                (utc_now(), conversation_id),
            )
            await self.conn.commit()

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        provider_id: str | None = None,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = f"msg_{uuid.uuid4().hex[:14]}"
        stamp = utc_now()
        async with self._lock:
            conversation = await self._fetchone_unlocked(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            )
            if conversation is None:
                raise KeyError(f"Unknown conversation {conversation_id!r}")
            await self.conn.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content, provider_id, model,
                    input_tokens, output_tokens, cost_usd, latency_ms, meta, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    provider_id,
                    model,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    latency_ms,
                    json.dumps(meta or {}, ensure_ascii=False),
                    stamp,
                ),
            )
            title = conversation["title"]
            if role == "user" and title in {"New conversation", "New Chat"}:
                title = content.strip().replace("\n", " ")[:64] or "New conversation"
            await self.conn.execute(
                """
                UPDATE conversations
                   SET updated_at = ?, message_count = message_count + 1, title = ?
                 WHERE id = ?
                """,
                (stamp, title, conversation_id),
            )
            await self.conn.commit()
        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "provider_id": provider_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "meta": meta or {},
            "created_at": stamp,
        }

    async def get_messages(self, conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = await self.fetchall(
            """
            SELECT * FROM (
                SELECT * FROM messages
                 WHERE conversation_id = ?
                 ORDER BY created_at DESC, rowid DESC
                 LIMIT ?
            ) ORDER BY created_at ASC, rowid ASC
            """,
            (conversation_id, limit),
        )
        return [_message(row) for row in rows]

    async def recent_messages(self, conversation_id: str, limit: int = 24) -> list[dict[str, Any]]:
        return await self.get_messages(conversation_id, limit)

    async def count_messages(self) -> int:
        row = await self.fetchone("SELECT COUNT(*) AS n FROM messages")
        return int(row["n"]) if row else 0

    async def count_conversations(self) -> int:
        row = await self.fetchone("SELECT COUNT(*) AS n FROM conversations")
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    async def add_event(
        self,
        message: str,
        *,
        level: str = "info",
        kind: str = "system",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stamp = utc_now()
        async with self._lock:
            cursor = await self.conn.execute(
                "INSERT INTO events (level, kind, message, data, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (level, kind, message, json.dumps(data or {}, ensure_ascii=False), stamp),
            )
            await self.conn.commit()
            event_id = cursor.lastrowid
        return {
            "id": event_id,
            "level": level,
            "kind": kind,
            "message": message,
            "data": data or {},
            "created_at": stamp,
        }

    async def list_events(self, limit: int = 120) -> list[dict[str, Any]]:
        rows = await self.fetchall(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        )
        items = []
        for row in rows:
            item = dict(row)
            try:
                item["data"] = json.loads(item.get("data") or "{}")
            except json.JSONDecodeError:
                item["data"] = {}
            items.append(item)
        return list(reversed(items))

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    async def add_memory(
        self, text: str, *, tags: list[str] | None = None, source: str = "user"
    ) -> dict[str, Any]:
        memory_id = f"mem_{uuid.uuid4().hex[:10]}"
        stamp = utc_now()
        async with self._lock:
            await self.conn.execute(
                "INSERT INTO memory (id, text, tags, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    memory_id,
                    text.strip(),
                    json.dumps(tags or [], ensure_ascii=False),
                    source,
                    stamp,
                ),
            )
            await self.conn.commit()
        return {
            "id": memory_id,
            "text": text.strip(),
            "tags": tags or [],
            "source": source,
            "created_at": stamp,
            "last_used_at": None,
        }

    async def list_memory(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self.fetchall(
            "SELECT * FROM memory ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        items = []
        for row in rows:
            item = dict(row)
            try:
                item["tags"] = json.loads(item.get("tags") or "[]")
            except json.JSONDecodeError:
                item["tags"] = []
            items.append(item)
        return items

    async def delete_memory(self, memory_id: str) -> bool:
        async with self._lock:
            cursor = await self.conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
            await self.conn.commit()
            return cursor.rowcount > 0

    async def search_memory(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        words = [word for word in _words(query) if len(word) > 2][:8]
        if not words:
            return []
        clause = " OR ".join("text LIKE ?" for _ in words)
        params = [f"%{word}%" for word in words]
        rows = await self.fetchall(
            f"SELECT * FROM memory WHERE {clause} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        )
        items = []
        for row in rows:
            item = dict(row)
            try:
                item["tags"] = json.loads(item.get("tags") or "[]")
            except json.JSONDecodeError:
                item["tags"] = []
            items.append(item)
        return items

    async def touch_memory(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        stamp = utc_now()
        async with self._lock:
            await self.conn.executemany(
                "UPDATE memory SET last_used_at = ? WHERE id = ?",
                [(stamp, memory_id) for memory_id in memory_ids],
            )
            await self.conn.commit()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    async def get_setting(self, key: str, default: Any = None) -> Any:
        row = await self.fetchone("SELECT value FROM settings WHERE key = ?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    async def set_setting(self, key: str, value: Any) -> None:
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, json.dumps(value, ensure_ascii=False, default=str)),
            )
            await self.conn.commit()

    async def delete_setting(self, key: str) -> None:
        async with self._lock:
            await self.conn.execute("DELETE FROM settings WHERE key = ?", (key,))
            await self.conn.commit()

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    async def fetchall(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        async with self._lock:
            cursor = await self.conn.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()
        return [dict(row) for row in rows]

    async def fetchone(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None:
        async with self._lock:
            return await self._fetchone_unlocked(query, params)

    async def _fetchone_unlocked(
        self, query: str, params: tuple[Any, ...]
    ) -> dict[str, Any] | None:
        cursor = await self.conn.execute(query, params)
        row = await cursor.fetchone()
        await cursor.close()
        return _row(row)


def _words(text: str) -> list[str]:
    cleaned = "".join(char if char.isalnum() else " " for char in text.lower())
    return [word for word in cleaned.split() if word]
