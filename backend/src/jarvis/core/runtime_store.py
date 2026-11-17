"""Persistent runtime model selections.

Kept separate from ``config.yaml`` so a dashboard switch never rewrites the
user's annotated configuration file and survives restarts. Writes are atomic.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from .errors import ConfigError
from .logging import get_logger

log = get_logger(__name__)

_SCHEMA_VERSION = 1
_KINDS = ("llm", "stt", "tts")


class RuntimeSelectionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> RuntimeSelectionStore:
        return cls(Path(data_dir) / "runtime" / "models.json")

    def load(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("ignoring unreadable runtime selection store", extra={"error": str(exc)})
            return {}
        if not isinstance(payload, dict) or payload.get("version") != _SCHEMA_VERSION:
            log.warning("ignoring runtime selection store with unknown schema")
            return {}
        selection = payload.get("selection")
        if not isinstance(selection, dict):
            return {}
        return {
            kind: provider_id
            for kind, provider_id in selection.items()
            if kind in _KINDS and isinstance(provider_id, str) and provider_id
        }

    def save(self, selection: Mapping[str, str]) -> None:
        payload = {
            "version": _SCHEMA_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
            "selection": {kind: selection[kind] for kind in _KINDS if kind in selection},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".models-", suffix=".json", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except OSError as exc:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise ConfigError(f"Could not persist model selection to {self.path}: {exc}") from exc
