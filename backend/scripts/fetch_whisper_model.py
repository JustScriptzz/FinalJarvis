"""Fetch the Faster-Whisper model used by the ``local-whisper`` speech core.

The ``faster_whisper`` provider loads CTranslate2 Whisper weights through
``WhisperModel``, which resolves a model name such as ``large-v3`` against the
Hugging Face Hub and caches it under the configured ``download_root``. Without
this script that download happens on the first spoken sentence: a multi-gigabyte
fetch buried in the middle of a voice turn, with no prompt and a five-minute
provider timeout around it.

Running it first makes that download a deliberate, visible, resumable step:

    python -m scripts.fetch_whisper_model                  # large-v3, the default
    python -m scripts.fetch_whisper_model --model small
    python -m scripts.fetch_whisper_model --check          # verify, download nothing

It writes to the same cache the provider reads, so nothing else has to change.
It is idempotent: a complete model is reported and left alone, and a partial one
is resumed by the Hub client. ``--force`` deletes the cached copy and fetches it
again.

Approximate download sizes, for deciding before you start:

    tiny 75 MB   base 145 MB   small 480 MB   medium 1.5 GB
    large-v3 3.1 GB   distil-large-v3 1.5 GB (English only, much faster)

The model that is fetched is the one named in ``config/config.yaml``; pass the
same name here.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Suffixes the provider needs. ``vocabulary.*`` covers both the multilingual
# ``vocabulary.json`` and the English-only ``vocabulary.txt`` layouts.
REQUIRED = ("model.bin", "config.json", "tokenizer.json")
VOCAB_PREFIX = "vocabulary."


def _repo_for(model: str) -> str:
    """Map a shorthand size to its Hub repository, or accept a repository id."""
    if "/" in model:
        return model
    try:
        from faster_whisper.utils import _MODELS
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "faster-whisper is not installed; run: "
            "uv pip install --python .venv/bin/python 'faster-whisper>=1.1.0,<2.0'"
        ) from exc
    repo = _MODELS.get(model)
    if repo is None:
        raise SystemExit(
            f"unknown model {model!r}; expected one of: {', '.join(sorted(_MODELS))}\n"
            "or a Hub repository id such as Systran/faster-whisper-large-v3"
        )
    return repo


def _cache_dir_for(out: Path, repo: str) -> Path:
    return out / f"models--{repo.replace('/', '--')}"


def _missing(path: Path) -> list[str]:
    """Files the provider opens that are absent from a downloaded snapshot."""
    missing = [name for name in REQUIRED if not (path / name).is_file()]
    if not (
        (path / "vocabulary.json").is_file() or (path / "vocabulary.txt").is_file()
    ):
        missing.append(VOCAB_PREFIX + "{json,txt}")
    return missing


def _snapshot(out: Path, repo: str) -> Path | None:
    """Newest downloaded snapshot of ``repo`` under ``out``, if any."""
    root = _cache_dir_for(out, repo) / "snapshots"
    if not root.is_dir():
        return None
    candidates = [entry for entry in root.iterdir() if entry.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.stat().st_mtime)


def _size(path: Path) -> int:
    """Bytes held by a snapshot, following the cache's symlinks to its blobs."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:  # a dangling link is a broken download, not a crash
                continue
    return total


def build(model: str, out: Path, force: bool, retries: int = 3) -> None:
    from faster_whisper.utils import download_model

    repo = _repo_for(model)
    out.mkdir(parents=True, exist_ok=True)

    if force:
        target = _cache_dir_for(out, repo)
        if target.is_dir():
            print(f"--force: removing {target}")
            shutil.rmtree(target)

    existing = _snapshot(out, repo)
    if existing is not None:
        missing = _missing(existing)
        if not missing:
            print(f"{model} already downloaded: {existing}")
            print(f"  {_size(existing) / 1e9:.2f} GB on disk")
            return
        print(f"{model} is incomplete (missing {', '.join(missing)}); fetching the rest")

    print(f"downloading {repo} into {out}", flush=True)
    # A multi-gigabyte transfer over a home connection drops. The Hub client
    # resumes from the partial file, so a retry costs only the missing bytes.
    path: Path | None = None
    for attempt in range(1, retries + 1):
        try:
            # cache_dir is what faster_whisper.WhisperModel passes when the
            # provider supplies download_root, so the two agree on the layout.
            path = Path(download_model(model, cache_dir=str(out)))
            break
        # Any transport error is worth one more try.
        except Exception as exc:
            if attempt == retries:
                raise SystemExit(
                    f"download failed after {retries} attempts: {exc}\n"
                    "Partial data is kept; running this script again resumes it."
                ) from exc
            wait_s = 2**attempt
            print(f"  attempt {attempt} failed ({exc}); retrying in {wait_s}s", flush=True)
            time.sleep(wait_s)
    assert path is not None  # the loop either breaks or exits

    missing = _missing(path)
    if missing:
        raise SystemExit(f"download finished but {path} is missing: {', '.join(missing)}")
    print(f"\nmodel ready: {path}")
    print(f"  {_size(path) / 1e9:.2f} GB on disk")
    print(
        "\nThe local-whisper core reads it through config/config.yaml:\n"
        f"  model: {model}\n"
        f"  download_root: {out}"
    )


def check(model: str, out: Path) -> None:
    """Verify a cached copy without touching the network."""
    from faster_whisper.utils import download_model

    repo = _repo_for(model)
    path = _snapshot(out, repo)
    if path is None:
        raise SystemExit(f"{model} is not downloaded under {out}; run this script without --check")
    missing = _missing(path)
    if missing:
        raise SystemExit(f"{path} is incomplete: missing {', '.join(missing)}")
    # local_files_only is the exact query the provider makes on a cold start.
    resolved = Path(download_model(model, cache_dir=str(out), local_files_only=True))
    print(f"{model} is complete and usable offline: {resolved}")
    print(f"  {_size(resolved) / 1e9:.2f} GB on disk")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="large-v3",
        help="model name as used in config.yaml (large-v3, small, distil-large-v3, ...)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "models" / "whisper",
        help="cache directory, matching the provider's download_root",
    )
    parser.add_argument("--force", action="store_true", help="delete and fetch again")
    parser.add_argument(
        "--check", action="store_true", help="only verify a cached copy; never download"
    )
    args = parser.parse_args(argv)
    if args.check:
        check(args.model, args.out)
    else:
        build(args.model, args.out, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
