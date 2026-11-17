"""Model discovery against a user-supplied endpoint.

The setup screen has to be able to answer "what can I run here?" before the
user commits to a model name. Two shapes cover almost everything people
actually run self-hosted:

* ``GET {base}/models``  — OpenAI-compatible (vLLM, llama.cpp server, LM Studio,
  Ollama's ``/v1`` shim, OpenRouter, Groq, ...).
* ``GET {host}/api/tags`` — native Ollama.

Both are probed concurrently; whichever answers first with models wins. A
failure to reach the endpoint is reported, never guessed at.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .errors import ConfigError
from .logging import get_logger

log = get_logger(__name__)

_PROBE_TIMEOUT_S = 8.0


def normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip()
    if not value:
        raise ConfigError("An endpoint URL is required, for example http://localhost:11434/v1")
    if "://" not in value:
        value = f"http://{value}"
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"}:
        raise ConfigError("Endpoint URL must start with http:// or https://")
    if not parts.hostname:
        raise ConfigError("Endpoint URL has no host")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _native_ollama_root(base_url: str) -> str:
    """``http://h:11434/v1`` -> ``http://h:11434``."""
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _openai_models(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("data")
    if not isinstance(items, list):
        return []
    return [
        str(item["id"])
        for item in items
        if isinstance(item, dict) and item.get("id")
    ]


def _ollama_models(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("models")
    if not isinstance(items, list):
        return []
    return [
        str(item["name"] or item["model"])
        for item in items
        if isinstance(item, dict) and (item.get("name") or item.get("model"))
    ]


async def discover_models(
    base_url: str,
    *,
    api_key: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """List the models an endpoint serves.

    Returns ``{"models": [...], "source": "...", "endpoint": "..."}``. Raises
    ``ConfigError`` with the transport failure when nothing answers, so the HUD
    can show the real reason instead of an empty list.
    """
    root = normalize_base_url(base_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    ollama = _native_ollama_root(root)

    attempts: list[tuple[str, str, Any]] = [
        ("openai", f"{root}/models", _openai_models),
        ("ollama", f"{ollama}/api/tags", _ollama_models),
    ]
    if not root.endswith("/v1"):
        # Some servers only mount the OpenAI surface under /v1 while the bare
        # root serves something else entirely.
        attempts.append(("openai", f"{root}/v1/models", _openai_models))

    owned = http is None
    client = http or httpx.AsyncClient(follow_redirects=True)
    failures: list[str] = []
    try:
        for source, url, parser in attempts:
            try:
                response = await client.get(
                    url, headers=headers, timeout=_PROBE_TIMEOUT_S
                )
            except httpx.HTTPError as exc:
                failures.append(f"{url}: {type(exc).__name__}")
                continue
            if response.status_code >= 400:
                failures.append(f"{url}: HTTP {response.status_code}")
                continue
            try:
                payload = response.json()
            except ValueError:
                failures.append(f"{url}: response was not JSON")
                continue
            models = sorted({item for item in parser(payload) if item})
            if models:
                return {
                    "models": models,
                    "source": source,
                    "endpoint": url,
                }
            failures.append(f"{url}: no models listed")
    finally:
        if owned:
            await client.aclose()

    raise ConfigError(
        "No models found at that endpoint. "
        + ("; ".join(failures) if failures else "The server returned nothing.")
    )


__all__ = ["discover_models", "normalize_base_url"]
