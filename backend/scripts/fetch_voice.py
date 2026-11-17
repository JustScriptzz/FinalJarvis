"""Build J.A.R.V.I.S.'s voice reference from the public sample dataset.

F5-TTS clones a voice from two things: a short clip and the exact words spoken
in it. ``derekurban2001/jarvis-voice-samples`` on Hugging Face is 720 clips of
the same speaker reading fixed sentences, each labelled with its transcript,
which is exactly the input cloning needs. This script picks three consecutive
sentences, joins them with a little silence, and writes the pair the provider
config points at::

    python -m scripts.fetch_voice
    # -> data/voices/jarvis-reference.wav
    # -> data/voices/jarvis-reference.txt

The clips are fetched from the Datasets Server, so no Hugging Face token or
``datasets`` install is required. Nothing is written when the files already
exist unless ``--force`` is passed, and every step reports what it did.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

DATASET = "derekurban2001/jarvis-voice-samples"
SPLIT = "train"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={DATASET.replace('/', '%2F')}&config=default&split={SPLIT}"
    "&offset=0&length=100"
)
USER_AGENT = "jarvis-voice-setup/1.0"

# Row indices into the dataset. F5 re-generates the reference audio on every
# call, so a longer clip costs time twice over: measured here, ~8.3 s of
# reference took 657 s per sentence and ~5 s took roughly half that, with no
# audible gain in likeness. Two sentences is the balance.
PICKS = (2, 3)
GAP_SECONDS = 0.18


def _get(url: str, timeout: float = 90.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _rows() -> dict[int, dict[str, object]]:
    try:
        payload = json.loads(_get(ROWS_URL, timeout=45.0))
    except urllib.error.URLError as exc:
        raise SystemExit(f"could not reach the Datasets Server: {exc}") from exc
    rows: dict[int, dict[str, object]] = {}
    for row in payload.get("rows", []):
        index = int(row["row_idx"])
        rows[index] = row["row"]
    missing = [index for index in PICKS if index not in rows]
    if missing:
        raise SystemExit(
            f"the dataset no longer lists rows {missing}; update PICKS in this script"
        )
    return rows


def _clip(row: dict[str, object]) -> tuple[bytes, str]:
    audio = row.get("audio")
    entry = audio[0] if isinstance(audio, list) else audio
    if not isinstance(entry, dict) or not entry.get("src"):
        raise SystemExit("a dataset row has no audio URL")
    return _get(str(entry["src"])), str(row.get("text") or "").strip()


def build(target: Path) -> None:
    rows = _rows()
    frames_by_index: dict[int, bytes] = {}
    params: wave._wave_params | None = None
    texts: list[str] = []
    for index in PICKS:
        raw, text = _clip(rows[index])
        with wave.open(io.BytesIO(raw)) as clip:
            if params is None:
                params = clip.getparams()
            elif (clip.getframerate(), clip.getnchannels(), clip.getsampwidth()) != (
                params.framerate,
                params.nchannels,
                params.sampwidth,
            ):
                raise SystemExit(
                    f"row {index} has a different audio layout; pick rows that match"
                )
            frames_by_index[index] = clip.readframes(clip.getnframes())
        texts.append(text)
        print(f"  row {index}: {text}")

    assert params is not None
    gap = b"\x00" * int(params.framerate * GAP_SECONDS) * params.nchannels * params.sampwidth
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        for position, index in enumerate(PICKS):
            if position:
                out.writeframes(gap)
            out.writeframes(frames_by_index[index])

    seconds = sum(
        len(frames_by_index[index]) / (params.framerate * params.nchannels * params.sampwidth)
        for index in PICKS
    ) + GAP_SECONDS * (len(PICKS) - 1)
    target.with_suffix(".txt").write_text(" ".join(texts), encoding="utf-8")
    print(f"wrote {target} ({seconds:.1f}s, {params.framerate} Hz, {params.nchannels} ch)")
    print(f"wrote {target.with_suffix('.txt')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "voices" / "jarvis-reference.wav",
        help="where to write the reference clip",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing clip")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        print(f"{args.out} already exists; pass --force to rebuild it")
        return 0
    print(f"building a voice reference from {DATASET}")
    build(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
