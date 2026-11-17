# J.A.R.V.I.S.'s voice

The default voice core is `local-jarvis`: a **Piper** voice model trained on the
J.A.R.V.I.S. character voice, running on this machine. No external TTS service,
no API key, no network at synthesis time. A sentence takes a few seconds on CPU,
so speech keeps up with a conversation.

The model is [`jgkawell/jarvis`](https://huggingface.co/jgkawell/jarvis) on
Hugging Face — the community J.A.R.V.I.S. voice widely used for Home Assistant.
It is **MIT licensed**, British RP, male, and shipped as plain Piper Onnx, so
there is no licence question and no custom runtime to maintain.

`local-f5` (F5-TTS cloning a reference clip) is kept as an alternative for
machines with a GPU. It sounds less like the character on some lines, and it is
far slower on a small CPU. See [The F5 clone](#the-f5-clone-gpu-alternative).

## The default: `local-jarvis`

### Fetching the model

```bash
cd backend
PYTHONPATH=src .venv/bin/python scripts/fetch_piper_voice.py
```

This writes two files that must belong together:

```
data/models/piper/jarvis-medium.onnx         (~63 MB)
data/models/piper/jarvis-medium.onnx.json    (voice shape + phoneme settings)
```

The script is idempotent: complete files are skipped, truncated ones are
re-downloaded. Pass `--quality high` for the ~110 MB `high` voice, which sounds
slightly better and costs proportionally more CPU; `medium` is the default
because it is fast enough to speak in real time on a small ARM host.

Neither file belongs in the repository; both live under `data/`.

### Installing the engine

```bash
cd backend
uv pip install --python .venv/bin/python piper-tts
```

That pulls `onnxruntime` and a handful of small dependencies. It installs the
`piper` executable into the environment's `bin` directory, **not** onto `PATH`.
The provider looks beside the running interpreter before checking `PATH`, so the
server works when started as `.venv/bin/python -m jarvis...` without activating
the environment first. A missing binary is reported with the install command in
the error message rather than as a generic failure.

### Measured cost

On this host (7 usable ARM cores, no GPU), through `PiperTTS.synthesize()`:

| Input | Audio | Wall time |
| --- | --- | --- |
| "Good evening, sir. All systems are nominal and the uplink is stable." | 3.62 s | 4.02 s |

That is roughly real time — several times faster than you can read the sentence
aloud, and it is not streaming, so a short sentence may finish before playback
starts and the HUD never waits.

### Level and clipping

This voice renders hot: at the CLI's default `volume: 1.0` the output peaks at
exactly 1.000 and two samples clip on a normal sentence. The shipped config sets
`volume: 0.8`, which caps the peak at 0.800 with **zero** clipped samples and
leaves the level comfortable. Measure rather than trust:

| `volume` | Peak | RMS | Clipped samples |
| --- | --- | --- | --- |
| 1.0 | 1.000 | 0.161 | 2 |
| 0.8 | 0.800 | 0.148 | 0 |
| 0.7 | 0.700 | 0.125 | 0 |

`length_scale` follows the model's own config at `1.15`; lower is faster and
flatter, higher is slower and more deliberate.

## Configuration

The shipped `config/config.yaml` already points at the fetched model:

```yaml
tts:
  active: local-jarvis
  fallback:
    - local-f5
    - local-kokoro
    # ... server cores
  providers:
    local-jarvis:
      enabled: true
      type: piper
      local: true
      binary: piper
      model_path: ./data/models/piper/jarvis-medium.onnx
      config_path: ./data/models/piper/jarvis-medium.onnx.json
      sample_rate: 22050
      format: wav
      timeout_s: 120
      max_concurrency: 1
      options:
        volume: 0.8
        length_scale: 1.15
```

`model_path` and `config_path` must name the same voice; pairing a model with
another model's config produces garbled phonemes. The provider checks that both
files exist before running the CLI, so a missing download is an error with the
path in it, not a silent fallback.

## Hearing it

A real sample generated on this machine is in
`data/voices/samples/jarvis-piper-good-evening.wav` (3.62 s, no clipping). It is
gitignored with the rest of `data/`, so regenerate one any time:

```bash
curl -s -X POST http://127.0.0.1:8080/api/tts/preview \
  -H 'Content-Type: application/json' \
  -d '{"provider_id": "local-jarvis", "text": "Good evening, sir."}' \
  --output data/voices/samples/jarvis-piper-good-evening.wav
```

## Making it the voice the server uses

`config.yaml` alone does not decide this. Once setup has been completed, the
stored setup record in SQLite is authoritative — the same rule that lets the
dashboard choose a browser voice, which the router cannot represent. So the
voice has to be engaged through the API (or the dashboard), not just edited into
the file:

```bash
curl -X POST http://127.0.0.1:8080/api/models/switch \
  -H 'Content-Type: application/json' \
  -d '{"kind": "tts", "provider_id": "local-jarvis"}'
```

The dashboard equivalent is the core matrix → voice → `local-jarvis`. Switching
also clears any stored voice name that belonged to the previous core, since an
old name means nothing to a new engine.

If synthesis fails, the response carries the provider's own error and
`GET /api/models` shows the core's health; the server does not silently fall
back to a different voice.

## The F5 clone (GPU alternative)

`local-f5` clones a reference clip of J.A.R.V.I.S. with F5-TTS. It is worth it
only where a GPU makes synthesis cheap; on a CPU-only machine it takes minutes
per sentence and is not usable for conversation.

### Where the voice comes from

The reference clip is built from
[`derekurban2001/jarvis-voice-samples`](https://huggingface.co/datasets/derekurban2001/jarvis-voice-samples):
720 short clips of one speaker, each labelled with the sentence spoken. Cloning
needs exactly that pair — audio for timbre, transcript for alignment.

```bash
cd backend
PYTHONPATH=src .venv/bin/python scripts/fetch_voice.py
```

The script picks two sentences, joins them with 180 ms of silence, and writes
`data/voices/jarvis-reference.wav` and `data/voices/jarvis-reference.txt`. Two is
deliberate: F5 re-encodes the reference on every call, so every extra second of
clip is paid for on every sentence. It is idempotent; pass `--force` to rebuild.

### Installing the engine

`f5-tts` pulls a CUDA build of PyTorch by default. On a machine without an
NVIDIA GPU, install the CPU build first so the download stays reasonable:

```bash
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.10.0+cpu" "torchaudio==2.10.0+cpu"
uv pip install --python .venv/bin/python --no-deps f5-tts
uv pip install --python .venv/bin/python --no-deps \
  soundfile cffi vocos x-transformers torchdiffeq torch-einops-utils \
  transformers-stream-generator unidecode "transformers<5" \
  "tokenizers==0.22.1" "huggingface-hub==0.35.3"
uv pip install --python .venv/bin/python \
  tqdm cached-path hydra-core omegaconf einops einx ema-pytorch pypinyin \
  rjieba matplotlib pydub audioop-lts regex encodec loguru wandb accelerate \
  multiprocess pyarrow pandas dill librosa soxr
uv pip install --python .venv/bin/python --no-deps "datasets==3.6.0"
```

Install with `--link-mode=copy` if the uv cache and the virtual environment sit
on different filesystems: the default hardlink fails with
`Operation not permitted`, which looks like a download problem but is not.

`torchcodec` is deliberately **not** installed. Its wheels bundle links to
ffmpeg shared libraries (`libavutil.so.58/60/61`) that many hosts do not have,
and a missing one breaks audio decoding with an error that reads like a
synthesis failure. The provider reads the reference clip with `soundfile`, which
bundles its own libsndfile, so no ffmpeg is involved anywhere.

### Cost on a CPU-only machine

Measured here: a 7-core ARM host, no GPU, `nfe_step: 16`, a 5.2 s reference
clip. These are real numbers from this repository, not estimates.

| Step | Time |
| --- | --- |
| First `import f5_tts` (cold page cache) | ~80–200 s |
| `import f5_tts` (warm) | ~14 s |
| Loading the model, weights already cached | 6–8 s |
| Loading the model, first run (weights download) | ~4 min |
| Synthesizing 2.1 s of audio | ~205 s |

The reference length dominates: it is re-encoded on **every** call, so the
8.3 s three-sentence reference this used to ship took 657 s per sentence, while
the 5.2 s two-sentence one takes ~205 s. That is why the clip is short.

`nfe_step` scales almost linearly, measured on the same sentence:

| `nfe_step` | Time | Audio |
| --- | --- | --- |
| 8 | 90 s | 2.94 s |
| 16 | 203 s | 2.88 s |
| 24 | 258 s | 2.93 s |

On a desktop CPU expect tens of seconds per sentence rather than minutes; with
an NVIDIA GPU (`device: cuda`) expect a few seconds. `nfe_step` 32 is the
model's own default and worth it only on a GPU.

The provider emits audio **after** the whole sentence is done — it is not
streaming. The HUD's TEST button shows elapsed time and says it is cloning
locally, so a long wait does not look like a freeze.

### Configuration

```yaml
local-f5:
  enabled: true
  type: f5_tts
  reference_audio: ./data/voices/jarvis-reference.wav
  reference_text: >-
    It's easy to tell the depth of a well. These days a chicken leg is a rare dish.
  device: cpu
  format: wav
  timeout_s: 900
  options:
    nfe_step: 16
```

`reference_text` must match the clip word for word; a mismatch makes the clone
slur or mispronounce. `tests/test_config.py::test_f5_voice_core_stays_usable_as_an_alternative`
enforces that the shipped pair agrees, because a mismatch otherwise shows up
only as bad audio.

Engage it the same way, naming `local-f5`:

```bash
curl -X POST http://127.0.0.1:8080/api/models/switch \
  -H 'Content-Type: application/json' \
  -d '{"kind": "tts", "provider_id": "local-f5"}'
```

The first request loads the model and downloads the weights, so expect several
minutes of silence before audio.
