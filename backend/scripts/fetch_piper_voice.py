"""Fetch the JARVIS voice model used as the default voice core.

The voice is [`jgkawell/jarvis`](https://huggingface.co/jgkawell/jarvis) on
Hugging Face: a Piper voice trained on the J.A.R.V.I.S. character voice, with a
British RP accent. It is MIT licensed, so it can ship as a default without any
licensing question, and it is a plain Piper ONNX model — the ``piper`` provider
already knows how to run it.

Two files are needed and they must match::

    data/models/piper/jarvis-medium.onnx
    data/models/piper/jarvis-medium.onnx.json

Run it once; it is idempotent and skips files that are already complete::

    python -m scripts.fetch_piper_voice

Pass ``--force`` to re-download. ``--high`` fetches the heavier ``high`` quality
model (~110 MB) instead of ``medium`` (~63 MB); ``medium`` is the default because
it synthesises a sentence in a few seconds on a small CPU.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "jgkawell/jarvis"
BASE = f"https://huggingface.co/{REPO}/resolve/main/en/en_GB/jarvis"
USER_AGENT = "jarvis-voice-setup/1.0"
EXPECTED_MIN_BYTES = {
    "medium": 40 * 1024 * 1024,
    "high": 70 * 1024 * 1024,
}


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("content-length") or 0)
            written = 0
            with partial.open("wb") as handle:
                while chunk := response.read(1024 * 256):
                    handle.write(chunk)
                    written += len(chunk)
        if total and written != total:
            raise SystemExit(f"{target.name}: got {written} of {total} bytes")
        partial.replace(target)
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"could not download {url}: {exc}") from exc


def build(target_dir: Path, quality: str, force: bool) -> None:
    stem = f"jarvis-{quality}"
    files = [f"{stem}.onnx", f"{stem}.onnx.json"]
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        destination = target_dir / name
        if destination.is_file() and not force:
            if name.endswith(".onnx") and destination.stat().st_size < EXPECTED_MIN_BYTES[quality]:
                # A truncated earlier download is worse than no file: Piper
                # would fail late, at synthesis time, with an opaque error.
                print(f"{name} looks incomplete; re-downloading")
            else:
                print(f"{name} already present; skipping")
                continue
        print(f"downloading {BASE}/{quality}/{name}")
        _download(f"{BASE}/{quality}/{name}", destination)
        print(f"  {destination} ({destination.stat().st_size // 1024} KB)")

    model = target_dir / f"{stem}.onnx"
    config = target_dir / f"{stem}.onnx.json"
    if not model.is_file() or not config.is_file():
        raise SystemExit("download finished but the model files are missing")
    print(f"\nvoice ready: {model}")
    print(f"             {config}")
    print(
        "\nPoint config/config.yaml at them:\n"
        f"  model_path: {model}\n"
        f"  config_path: {config}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "models" / "piper",
        help="directory for the model files",
    )
    parser.add_argument(
        "--quality", choices=("medium", "high"), default="medium",
        help="model size: medium is fast on a small CPU, high sounds better",
    )
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)
    build(args.out, args.quality, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
