# Speech input

JARVIS takes dictation from the microphone through whichever STT core the setup
record selects. The shipped default is `local-whisper`: Faster-Whisper, running
CTranslate2 Whisper weights **on your own machine**. Nothing you say leaves the
host.

```text
mic (HUD button or wake word)
  └─ MediaRecorder, audio/webm
       └─ POST /api/transcribe
            └─ local-whisper  (faster-whisper + CTranslate2)
                 └─ text, typed into the input and sent as a chat turn
```

Two other input paths exist and need no model at all: the browser's own speech
recognition (client side, see [below](#browser-recognition-no-download)) and
cloud STT cores, which need an API key. This page is about the local core.

## Install

The providers are optional extras because they pull large runtimes:

```bash
cd backend
uv pip install --python .venv/bin/python -e ".[local]"
```

`[local]` installs `faster-whisper` (which brings `ctranslate2` and `av`) and
`piper-tts`. If the voice is already installed, installing the package directly
is enough:

```bash
uv pip install --python .venv/bin/python "faster-whisper>=1.1.0,<2.0" "av<19"
```

The `av` bound matters: `av` 19 dropped the `metadata_errors` argument that
Faster-Whisper passes to `av.open`, and a bare install on a fresh machine
resolves to 19 today — the symptom is `TypeError: open() got an unexpected
keyword argument 'metadata_errors'` at the first transcription. The `[local]`
extra pins `av>=11,<19` so it cannot happen; a manual install must pin it too.

`uv` prints the packages it adds; it must not move `torch`, `transformers`,
`tokenizers` or `huggingface-hub`, because the F5-TTS voice depends on those
exact pins (see [voices.md](voices.md)).

## Fetch the weights before the first sentence

Without a local copy, Faster-Whisper downloads the model the first time you
speak — up to 3 GB, with no prompt, inside a 300-second provider timeout. Fetch
it deliberately instead:

```bash
cd backend
PYTHONPATH=src .venv/bin/python -m scripts.fetch_whisper_model --model large-v3
```

The script is idempotent, reports the size and the resolved path, and writes into
the same Hub cache the provider reads (`download_root` in `config.yaml`,
`data/models/whisper` by default), so no config change follows.

| Command | Effect |
| --- | --- |
| `... fetch_whisper_model` | fetch `large-v3`, skip if already complete |
| `--model small` | fetch a different model into the same cache |
| `--check` | verify a cached copy without touching the network |
| `--force` | delete the cached copy and fetch again |

A machine that has the cache restored from a backup can run `--check` before
going offline.

## Which model

The core is configured with `model:` in `config.yaml`. Sizes are the download
from the [Systran CTranslate2](https://huggingface.co/Systran) repositories.

| Model | Download | Notes |
| --- | --- | --- |
| `base` | ~145 MB | fast everywhere, mishears names and accents |
| `small` | ~480 MB | the usual choice for a laptop on battery |
| `distil-large-v3` | ~1.5 GB | English only, close to `large-v3` accuracy, several times faster |
| `large-v3` | ~3.1 GB | shipped default: multilingual, most accurate, needs a real CPU or a GPU |

On a desktop with a modern CPU, `large-v3` at `int8` transcribes a few seconds of
speech per second of audio. On a GPU it is close to instant. If you speak English
and want the same accuracy for less, `distil-large-v3` is the better trade.

## Device and precision

```yaml
local-whisper:
  type: faster_whisper
  model: large-v3
  device: auto                 # auto | cpu | cuda
  compute_type: auto           # auto = int8 on CPU, float16 on CUDA
  options:
    download_root: ./data/models/whisper
```

`auto` is resolved once, at load time:

- `device: auto` uses CUDA when CTranslate2 reports at least one GPU, otherwise
  the CPU.
- `compute_type: auto` follows that: `int8` on the CPU (half the memory traffic,
  no accuracy problem for Whisper), `float16` on a GPU (int8 there costs
  accuracy and buys no speed).
- An explicit `device` or `compute_type` always wins. Asking for `device: cuda`
  on a machine without a CUDA device fails immediately with a message telling
  you to use `auto`, instead of surfacing a driver error later.

After the first transcription the model stays resident; the health line reports
`large-v3 loaded on cpu (int8)`. Before that it reports either `cached` or
`downloads on first use`, so a cold start is never a surprise.

### Measured on the development box

The reference box is a memory-contended 8-core ARM host with heterogeneous core
speeds — worst case for local inference, and not what a desktop delivers. Still,
the numbers show the shape of the cost (large-v3, int8, 4.7 s of synthesized
speech, beam size 5):

| Phase | Time |
| --- | --- |
| Model load (resident afterwards) | 30–110 s |
| Transcribe, same clip, repeat runs | **75–390 s** |
| Accuracy | one mishearing ("dog" → "dot") on synthetic speech |

The spread within a single afternoon is the host's memory contention, not the
model: the same 4.7 s clip took 75 s at one point and 240 s an hour later. On a
desktop with a dedicated GPU this is a fraction of a second; on a desktop CPU it
is a few seconds per sentence.

Two knobs exist for constrained hosts: `options.cpu_threads` pins the worker
count (on big.LITTLE chips, the fastest core count wins — 4 beat the default
here), and a smaller `model` trades accuracy for speed. Neither belongs in the
shipped default, because both are host-specific.

`timeout_s` is deliberately **not** enforced for local transcription: CTranslate2
runs in native threads that cannot be cancelled, so a hard limit would only
abandon work that was going to succeed. Cloud STT cores enforce their timeouts
normally.

## Reading the result

`POST /api/transcribe` returns the text, the detected language, the audio
duration and the segments:

```bash
curl -s -X POST http://127.0.0.1:8080/api/transcribe \
  -F file=@clip.wav | jq '.data | {text, language, duration_s}'
```

`language` and `model` can be sent as form fields to override the configured
values for one request; the HUD sends the language from the tuning panel.

The HUD never invents a transcript: if the core is not engaged, the input panel
says **keyboard only**, and a failed transcription is reported as an error toast
with the provider's message.

## Browser recognition (no download)

The setup screen also offers a `browser` input core. It uses the Web Speech API
in Chrome or Edge, needs no model and costs no CPU — but the audio is sent to the
browser vendor's recogniser, so it is not private. It is the right choice for a
quick trial on a machine where you do not want a 3 GB download; `local-whisper`
is the right choice for a self-hosted assistant.

## Troubleshooting

**`faster-whisper not installed` in `/api/models`** — install the extra
(`uv pip install --python .venv/bin/python -e ".[local]"`) and restart the
server: the provider is chosen at import time.

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`** —
`av` 19 is installed; Faster-Whisper needs `av>=11,<19`. The `[local]` extra pins
it; downgrade with `uv pip install --python .venv/bin/python "av<19"`.

**Transcription is slow** — check the health detail for the loaded device and
precision. `cpu (int8)` with `large-v3` on a low-power CPU is the slow
combination; `small` or `distil-large-v3` is several times faster for dictation.

**`no CUDA device`** — `device: cuda` was set on a machine where CTranslate2 sees
no GPU (usually a CPU-only wheel of `ctranslate2`). Set `device: auto`.

**First dictation hangs for minutes** — the weights are downloading. Run
`scripts/fetch_whisper_model.py --check` to see whether a copy is cached, and
fetch it explicitly.

**No transcriber is engaged** — the setup record has `stt: null`, which means
keyboard only. Pick an input core in the HUD or run `/switch stt local-whisper`.
