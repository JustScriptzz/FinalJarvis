# JARVIS

Local-first, self-hosted personal AI assistant with a hot-swappable model
router, realtime voice, omnichannel messaging and full OS control.

**What works today:** the self-hosted web HUD with streaming chat, explicit
local/cloud core selection, hot-swappable LLM/STT/TTS routing with fallback,
local speech input (Faster-Whisper `large-v3`), the JARVIS voice (Piper, MIT
licensed) spoken sentence-by-sentence as answers stream in, an optional
`hey JARVIS` wake word, live host telemetry, slash commands, and a Windows
port. Still on the roadmap: MCP tools, the OS controller, the LiveKit call
pipeline and the messaging channels (see [Roadmap](#roadmap)).

---

## Quick look

One command. Installs whatever is missing (uv, Ollama, the backend, the speech
engines, ~5 GB of models on first run), then serves the HUD and opens it:

```bash
make start                # http://127.0.0.1:8080
```

Windows: `powershell -ExecutionPolicy Bypass -File scripts\start.ps1` — same
behavior (see [docs/windows.md](docs/windows.md)). Re-runs are fast: the
installer verifies what exists instead of downloading again.

On first boot the interface asks you to choose a **language core** and a
**voice core** explicitly — local or cloud, no defaults.

The core stage is a live particle sphere that swells with the real loudness of
your voice and of JARVIS's own speech, and the caption under it shows the
sentence being spoken. Answers are spoken a sentence at a time as they stream
in, the microphone (or an optional `hey JARVIS` wake word) interrupts the voice,
and everything degrades honestly when a browser has no audio graph. See
[`docs/web-interface.md`](docs/web-interface.md) for the API, WebSocket
protocol and HUD tour.


---

## Architecture

```text
                         ┌──────────────────────────────────────────┐
  Web HUD ──────────────▶│  FastAPI + WebSocket + async event bus   │
  Discord / WhatsApp  ···│  (roadmap: Phase 4 channels)             │
  WebRTC / SIP        ···│  (roadmap: Phase 3 voice pipeline)       │
                         └───────────────┬──────────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
   ┌──────▼──────┐                ┌──────▼──────┐                ┌──────▼──────┐
   │ ModelRouter │                │   Memory    │                │ Tool layer  │
   │  (Phase 1)  │                │ SQLite +    │                │ MCP + OS    │
   │             │                │ Chroma      │                │ controller  │
   │ LLM  STT TTS│                │  (Phase 2)  │                │  (Phase 2)  │
   └──────┬──────┘                └─────────────┘                └─────────────┘
          │
   local: Ollama / vLLM / llama.cpp, Faster-Whisper, Parakeet, Kokoro,
          Piper, F5-TTS
   cloud: OpenAI, Anthropic, Gemini, OpenRouter, Deepgram, Groq,
          ElevenLabs, Cartesia
```

Design decisions that shape the rest of the build:

- **The API/agent runs on the host, not in Docker.** OS control, GUI automation
  and screen capture must act on the real desktop. Docker Compose provides the
  supporting services (Redis, LiveKit, optional model servers).
- **One normalized provider contract.** Every LLM exposes the same streaming
  delta type; every STT/TTS provider returns the same result type. New
  providers do not touch the agent.
- **Secrets stay in the environment.** `config.yaml` is commit-safe and uses
  `${VAR}` expansion; dashboard model switches persist to
  `data/runtime/models.json` instead of rewriting the annotated YAML.
- **Any endpoint you run is a first-class core.** Cores added from the HUD live
  in `data/runtime/providers.json` (mode `0600`), are merged into the model
  configuration without touching the YAML, and can be probed, edited and removed
  from the setup screen or the core matrix.
- **Local-first defaults, cloud escalation opt-in.** Smart routing starts on
  Ollama and only reaches a cloud model after you enable it and add a key.
- **Failures are explicit.** Retryable errors fall back; authentication and
  bad-request errors surface immediately. A circuit breaker removes dead
  endpoints from the candidate list.

---

## Repository layout

```text
jarvis/
├── .env.example
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── docker-compose.yml              # Redis, LiveKit, optional Ollama/Speaches/Kokoro
├── config/
│   ├── config.yaml                 # single source of truth for providers + system
│   └── mcp_servers.json            # MCP servers (consumed in Phase 2)
├── infra/
│   └── livekit/
│       └── livekit.yaml            # LiveKit server config
├── docs/
│   ├── model-router.md             # router API, routing rules, provider guide
│   ├── voices.md                   # the JARVIS voice, Piper default, F5 alternative
│   ├── speech-input.md             # local Faster-Whisper input, models, GPU notes
│   ├── windows.md                  # Windows install and PowerShell startup
│   ├── latency.md                  # measured warm-turn latency and why it was slow
│   └── web-interface.md            # HTTP API, WebSocket frames, HUD, voice
├── scripts/
│   ├── install.sh                  # one-command install (Linux/macOS), idempotent
│   ├── install.ps1                 # the same for Windows
│   ├── start-ollama.sh             # Ollama with per-machine threads + engine fix
│   ├── start-ollama.ps1            # Windows twin
│   └── start.ps1                   # Windows `make serve` equivalent
└── backend/
    ├── README.md
    ├── pyproject.toml
    ├── src/jarvis/
    │   ├── __init__.py
    │   ├── py.typed
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py           # YAML + env expansion + strict validation
    │   │   ├── custom_store.py     # user-added cores, merged over the YAML
    │   │   ├── discovery.py        # endpoint -> model list probe
    │   │   ├── errors.py           # typed errors, retryability
    │   │   ├── logging.py          # JSON logs + request correlation ids
    │   │   └── runtime_store.py    # atomic, persisted model selection
    │   └── models/
    │       ├── __init__.py
    │       ├── audio.py            # PCM16/WAV, float32, resampling
    │       ├── base.py             # provider contracts + HTTP/error mapping
    │       ├── circuit.py          # per-provider circuit breaker
    │       ├── metrics.py          # usage, latency, cost accounting
    │       ├── registry.py         # type -> provider construction
    │       ├── router.py           # ModelRouter: routing, retry, fallback, hot swap
    │       ├── routing.py          # complexity scoring + capability filtering
    │       ├── sse.py              # SSE parser
    │       ├── streaming.py        # delta aggregation + tool calls
    │       ├── types.py            # normalized request/response models
    │       ├── llm/
    │       │   ├── openai_compatible.py
    │       │   ├── anthropic.py
    │       │   └── gemini.py
    │       ├── stt/
    │       │   ├── faster_whisper.py
    │       │   ├── parakeet.py
    │       │   ├── openai_compatible.py
    │       │   └── deepgram.py
    │       └── tts/
    │           ├── openai_compatible.py
    │           ├── elevenlabs.py
    │           ├── cartesia.py
    │           ├── piper.py
    │           └── f5_tts.py
    └── server/                     # Phase 2 web application
        ├── app.py                  # FastAPI factory + lifespan
        ├── chat.py                 # persona, conversation assembly, diagnostics
        ├── cli.py                  # `jarvis serve`
        ├── commands.py             # deterministic slash commands
        ├── db.py                   # SQLite repository
        ├── routes.py               # REST API
        ├── state.py                # setup validation + shared state
        ├── telemetry.py            # psutil host sampling
        ├── ws.py                   # WebSocket hub + streaming chat
        └── static/                 # HUD: HTML, CSS, JS (no build step)
    └── tests/                      # 140 tests: router, audio, config, stores,
                                    # providers, fetchers, server, chat prompt
```

---

## Quickstart

One command — installs everything (uv, Ollama, the backend environment, the
speech engines, ~5 GB of model weights on first run), then serves the HUD and
opens it:

```bash
git clone https://github.com/JustScriptzz/FinalJarvis.git
cd FinalJarvis
make start                  # Linux/macOS — Windows: run scripts\start.bat instead
```

Re-runs verify instead of downloading again. `./scripts/install.sh
--skip-models` installs the code only.

The numbered steps below are the same install spelled out by hand, for when you
want to see or change each piece. For Windows specifics see
[docs/windows.md](docs/windows.md).

### 1. Environment

```bash
cp .env.example .env
openssl rand -hex 32        # put this in LIVEKIT_API_SECRET
```

Set cloud keys only for the providers you plan to enable. The API token can
stay empty while the backend binds to `127.0.0.1`.

### 2. Ollama and models

Install Ollama from <https://ollama.com/download>, start it (or use
`scripts/start-ollama.sh`, which also picks a sensible thread count), and pull
the language core:

```bash
ollama pull qwen2.5:3b        # the shipped local LLM
```

The Docker stack (`docker compose up -d redis livekit`) is **optional** and
only needed for the Phase 3 call pipeline — the HUD, chat, speech input and
voices all run host-native with no containers. Speaches and Kokoro in the
compose `local` profile are optional alternative engines: the native
`faster_whisper` provider is the shipped STT path, and J.A.R.V.I.S.'s own voice
(`local-jarvis`, a Piper model of the character voice, MIT licensed) needs no
service at all — see [docs/voices.md](docs/voices.md).

### 3. Backend

```bash
cd backend
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
# local STT/TTS engines:
uv pip install --python .venv/bin/python -e ".[local]"
# speech input weights, one-time (3 GB for the default large-v3):
PYTHONPATH=src .venv/bin/python -m scripts.fetch_whisper_model --model large-v3
# the JARVIS voice (63 MB):
PYTHONPATH=src .venv/bin/python -m scripts.fetch_piper_voice
```

`uv` is recommended; a standard `python3.12 -m venv` works too.

### 4. Validate and test

```bash
make config-check
make test
```

Expected: all tests pass, and the config summary shows
`llm=local-qwen`, `stt=local-whisper`, `tts=local-jarvis`.

### 5. Run the web interface

```bash
make serve            # from the repository root
make serve-open       # also opens the browser
```

Open <http://127.0.0.1:8080>, choose a language core and a voice core, and the
HUD comes online. `jarvis serve --port 9000` and `jarvis serve --reload` are
available once the package is installed.

### 6. Talk to the router directly

```python
import asyncio

import httpx

from jarvis.core.config import load_config
from jarvis.core.runtime_store import RuntimeSelectionStore
from jarvis.models import ChatMessage, ChatRequest, ModelRouter


async def main() -> None:
    config = load_config("../config/config.yaml")
    async with httpx.AsyncClient() as http:
        router = ModelRouter(
            config.models,
            http_client=http,
            runtime_store=RuntimeSelectionStore.in_data_dir(config.app.data_dir),
        )
        response = await router.chat(
            ChatRequest(
                messages=[
                    ChatMessage(
                        role="user",
                        content="Summarize my day in one sentence.",
                    )
                ]
            )
        )
        print(response.text)
        print(response.provider_id, response.model, response.cost_usd)
        await router.aclose()


asyncio.run(main())
```

Streaming, tool calls, transcription, synthesis and hot switching are
documented in [`docs/model-router.md`](docs/model-router.md).

---

## Providers

| Kind | `type` | Notes |
|---|---|---|
| LLM | `openai_compatible` | OpenAI, OpenRouter, Ollama, vLLM, llama.cpp server, LM Studio, Groq |
| LLM | `anthropic` | Native Messages API, streaming, tool use |
| LLM | `gemini` | Native `generateContent` streaming, function calling |
| STT | `faster_whisper` | Local CTranslate2, lazy load, VAD, timestamps |
| STT | `parakeet` | Local NVIDIA NeMo, English |
| STT | `openai_compatible` | OpenAI Whisper API, Groq, Speaches |
| STT | `deepgram` | Nova pre-recorded, diarization, utterances |
| TTS | `openai_compatible` | OpenAI, Kokoro-FastAPI, Chatterbox, Qwen3-TTS, openedai-speech |
| TTS | `elevenlabs` | Flash v2.5, voice settings, PCM/MP3 output |
| TTS | `cartesia` | Sonic, raw/WAV/MP3, speed control |
| TTS | `piper` | Local CLI, raw PCM, no network |
| TTS | `f5_tts` | Local voice cloning from a reference clip |

Any OpenAI-compatible local server can be added with one config entry. The
router never assumes a provider family beyond the declared `type`.

---

## Smart routing

Each request is scored 0..1 from prompt length, code/reasoning signals,
multi-step structure, tool count and conversation depth:

```text
score < 0.32   -> fast profile      (small local model)
score < 0.62   -> balanced profile  (default local model)
otherwise      -> deep profile      (cloud reasoning model, if enabled)
```

Before a provider is chosen, the router filters by required capabilities
(`tools`, `vision`, `json`, `streaming`). The chain degrades
`deep -> balanced -> fast -> fallback -> active`, so a disabled cloud provider
never breaks a request. Explicit `provider_id` bypasses routing and fallback.

Per-provider retries use exponential backoff with jitter. Fallback happens only
for retryable failures (timeouts, connection errors, 429, 5xx). A provider that
fails five times in a row is circuit-broken for 30 seconds.

---

## Configuration

- `config/config.yaml` defines every provider and the system sections used by
  later phases.
- `.env` holds secrets. `${VAR}`, `${VAR:-default}` and `${VAR:?message}` are
  expanded at load time.
- `data/runtime/models.json` holds dashboard selections. It is written
  atomically with mode `0600` and applied on startup.
- Relative paths in YAML resolve against the repository root, not the current
  working directory.

Cloud providers ship disabled. Add the key, set `enabled: true`, then either
switch to them from the HUD's core matrix or pin them in `routing.profiles`.

---

## Security notes

- The HUD binds to loopback by default. Set `JARVIS_API_TOKEN` before exposing
  it through a reverse proxy; the REST and WebSocket layers enforce bearer auth.
- LiveKit's media ports are exposed on all interfaces so browser clients can
  connect. Set `rtc.use_external_ip: true` only when deploying beyond localhost.
- Do not commit `.env`, `data/`, model files or OAuth refresh tokens.
- OS-control permission tiers (`AUTO` / `NOTIFY` / `CONFIRM`) are configured in
  `config.yaml` and enforced by the Phase 2 `OSController`.
- Cost figures in `pricing` are estimates. Update them when providers change
  their rates.

---

## Roadmap

- [x] **Phase 1** — monorepo, compose, config, hot-swappable LLM/STT/TTS router
- [x] **Phase 2** — FastAPI web server, WebSocket streaming, SQLite
  conversations/memory/events, host telemetry, the HUD, explicit local/cloud
  core selection, streamed speech, wake word, one-command installer, Windows
  port. Remaining from the original spec: MCP loader, Chroma vectors and the
  OSController
- [ ] **Phase 3** — LiveKit Agents voice pipeline, SIP inbound/outbound, ProactiveEngine
- [ ] **Phase 4** — Email, Discord text + voice, WhatsApp bridge
- [ ] **Phase 5** — Mission Control extensions

---

## Troubleshooting

**`LIVEKIT_API_SECRET is required`** — only when running the optional Docker
stack for the Phase 3 call pipeline: copy `.env.example` to `.env` and set a
random secret. The host-native HUD never asks for it.

**Ollama returns 404** — pull the model named in `config.yaml`
(`qwen2.5:3b` is the shipped local model) or change `model` to one you have.

**Ollama answers `/api/version` but every completion fails with `llama-server
binary not found` or `no backends are loaded`** — the Ollama build on this host
keeps its inference engine outside `PATH` and launches it by a path relative to
its working directory. Use `scripts/start-ollama.sh` from the repository root
instead of calling `ollama serve` directly; it sets up the layout the engine
expects and explains why in its header. The same symptoms after a reboot mean
the server was started without it. On a normal install the script finds no
private engine directory and simply serves `ollama serve`, so it is safe to use
on any machine.

**Replies take 20–60 seconds although the machine is idle** — check whether the
engine is re-evaluating the whole prompt instead of reusing its cache:

```bash
# the Ollama log: stdout of scripts/start-ollama.sh, or `journalctl -u ollama` under systemd
grep -a "cached n_tokens\|prompt eval time\|  eval time" <ollama log> | tail -8
```

A healthy warm turn shows `cached n_tokens` close to the prompt size and a prompt
eval of a few hundred milliseconds. If it shows `cached n_tokens = 0`, something
volatile has been inserted before the conversation history — see
[docs/latency.md](docs/latency.md), which documents each cause with measurements.
If prompt eval is fast and `eval time` is slow, that is raw generation speed:
~2.5 tok/s on this CPU, and the only fixes are fewer tokens, a smaller model, or
better hardware. Two host-level causes are worth ruling out first, because both
are outside the application: the engine using all cores on a big.LITTLE chip, and
its weights being swapped out. Both are addressed by `scripts/start-ollama.sh`.

**Kokoro unreachable** — start the profile (`docker compose --profile local-tts
up -d`) and confirm `http://localhost:8880/v1/models`.

**Faster-Whisper import error** — install the local extra:
`uv pip install --python .venv/bin/python -e ".[local]"`.

**First dictation hangs or the transcript never appears** — `local-whisper`
downloads its model on first use. Pre-fetch it and verify the cache:

```bash
cd backend
PYTHONPATH=src .venv/bin/python -m scripts.fetch_whisper_model --model large-v3
PYTHONPATH=src .venv/bin/python -m scripts.fetch_whisper_model --check
```

Model sizes, the CPU/GPU precision rules and the browser alternative are in
[docs/speech-input.md](docs/speech-input.md).

**Remote WebRTC clients cannot connect** — set `rtc.use_external_ip: true` and
`rtc.node_ip` in `infra/livekit/livekit.yaml`, and allow UDP 50000-50100.

---

## License

MIT. See [`LICENSE`](LICENSE).
