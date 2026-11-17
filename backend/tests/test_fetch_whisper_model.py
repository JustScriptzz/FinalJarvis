"""The Whisper model fetcher: name resolution and cache verification.

The download itself needs the network and up to three gigabytes, so the script's
own logic is tested without it: which repository a short name maps to, what it
refuses, and how it decides a cached snapshot is complete. Those are the parts
that fail silently — a truncated snapshot that looks present, or a model name
that downloads something other than what ``config.yaml`` asks for.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from scripts import fetch_whisper_model as fetch

HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None


@pytest.mark.skipif(not HAS_FASTER_WHISPER, reason="faster-whisper extra not installed")
def test_short_names_map_to_the_hub_repositories_faster_whisper_uses() -> None:
    """The fetcher must agree with the provider, or it caches the wrong model."""
    from faster_whisper.utils import _MODELS

    cases = [("large-v3", "Systran/faster-whisper-large-v3"), ("small", _MODELS["small"])]
    for name, repo in cases:
        assert fetch._repo_for(name) == repo


def test_repository_ids_pass_through_untouched() -> None:
    assert fetch._repo_for("acme/custom-whisper") == "acme/custom-whisper"


@pytest.mark.skipif(not HAS_FASTER_WHISPER, reason="faster-whisper extra not installed")
def test_unknown_name_lists_the_valid_ones() -> None:
    with pytest.raises(SystemExit) as excinfo:
        fetch._repo_for("large-v9")

    message = str(excinfo.value)
    assert "large-v9" in message
    assert "large-v3" in message


def test_missing_files_are_reported(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    missing = fetch._missing(snapshot)

    assert "model.bin" in missing
    assert "config.json" in missing
    assert "tokenizer.json" in missing
    assert any(name.startswith("vocabulary.") for name in missing)


def test_a_complete_snapshot_reports_nothing_missing(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
        (snapshot / name).write_bytes(b"x")

    assert fetch._missing(snapshot) == []


def test_vocabulary_txt_is_accepted_for_english_only_models(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
        (snapshot / name).write_bytes(b"x")

    assert fetch._missing(snapshot) == []


def test_snapshot_finds_the_downloaded_copy(tmp_path: Path) -> None:
    root = tmp_path / "whisper"
    snapshot = root / "models--Systran--faster-whisper-large-v3" / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)

    assert fetch._snapshot(root, "Systran/faster-whisper-large-v3") == snapshot
    assert fetch._snapshot(root, "Systran/faster-whisper-small") is None


def test_snapshot_prefers_the_newest_copy(tmp_path: Path) -> None:
    root = tmp_path / "whisper"
    base = root / "models--Systran--faster-whisper-large-v3" / "snapshots"
    older = base / "a"
    newer = base / "b"
    for path, stamp in ((older, 1_000_000), (newer, 2_000_000)):
        path.mkdir(parents=True)
        assert path.exists()
        os.utime(path, (stamp, stamp))

    assert fetch._snapshot(root, "Systran/faster-whisper-large-v3") == newer


def test_size_follows_links_into_the_cache(tmp_path: Path) -> None:
    """Hub snapshots are symlinks; the reported size must not be a few bytes."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "weights"
    blob.write_bytes(b"x" * 4096)
    snapshot = _make_snapshot(tmp_path, "Systran/faster-whisper-large-v3")
    (snapshot / "model.bin").symlink_to(blob)

    assert fetch._size(snapshot) >= 4096


def _make_snapshot(root: Path, repo: str) -> Path:
    path = root / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc"
    path.mkdir(parents=True)
    return path
