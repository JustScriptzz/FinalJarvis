# Web interface

Phase 2 turns the router into a self-hosted web application: a FastAPI server,
a WebSocket event bus and a hand-built HUD with no frontend build step.

```bash
cd backend
uv pip install --python .venv/bin/python -e ".[dev]"
cd ..
PYTHONPATH=backend/src backend/.venv/bin/python -m jarvis.server.cli serve --config config/config.yaml
# or, after an editable install:
jarvis serve --open
```

Then open <http://127.0.0.1:8080>. On first boot JARVIS opens **core
selection** and refuses to chat until a language core and a voice core are
chosen explicitly. Nothing is auto-selected. The first completed selection
engages **manual** routing so the chosen core is actually used; switch to
`smart` from the core matrix when you want complexity-based escalation.

---

## What the HUD shows

| Region | Contents |
|---|---|
| Boot | Arc-reactor boot sequence, then a real WebSocket handshake |
| Vitals | CPU, memory, disk, load, per-core bars, network rate, top processes, host, battery (when present) |
| Core stage | Particle orb driven by real audio, state (`idle`, `processing`, `listening`, `speaking`, `error`), the sentence being spoken, active model, routing mode, last latency |
| Transcript | Streaming Markdown, provider/model metadata, tool-call notices, fault reports |
| Core matrix | LLM / STT / TTS inventory, local/cloud badges, health dots, hot switch, add/edit/remove your own cores, routing mode, usage metrics |
| Memory | Long-term engrams, search-backed, used in the system prompt |
| Log | Durable diagnostic event stream from SQLite |
| Archive | Conversation history: open, create, rename, erase |
| Voice | Browser or server TTS streamed sentence by sentence, push-to-talk via browser recognition or the selected STT provider, optional `hey JARVIS` wake word |

All telemetry is real `psutil` data. On restricted hosts (Termux, containers,
`/proc` hidden) unavailable sensors degrade to zero instead of failing the
request, and the HUD reports a locked or degraded provider honestly instead of
pretending it is online.

---

## Layout

```text
backend/src/jarvis/server/
├── __init__.py      # create_app re-export
├── app.py           # FastAPI factory, lifespan, static mount
├── chat.py          # conversation assembly, JARVIS persona, diagnostics
├── cli.py           # `jarvis serve` / `jarvis config-check`
├── commands.py      # deterministic slash commands
├── db.py            # SQLite repository (conversations, messages, events, memory)
├── routes.py        # REST API
├── schemas.py       # request models
├── state.py         # shared state, setup validation, setup payload
├── telemetry.py     # host sampling
├── ws.py            # WebSocket hub, chat streaming, background loops
└── static/
    ├── index.html
    ├── favicon.svg
    ├── css/jarvis.css
    └── js/
        ├── audio.js   # AudioMeter: shared analyser for mic and speech
        ├── orb.js     # particle sphere, state palette, level-driven swell
        └── app.js     # the interface
```

State that matters:

- `data/jarvis.db` — conversations, messages, memory, events, settings.
- `data/runtime/models.json` — router selection, written atomically.
- SQLite `settings` — completed setup and routing mode, so a restart resumes
  the same interface.

---

## REST API

All routes live under `/api`. When `server.api_token` is set, send
`Authorization: Bearer <token>`, `X-Jarvis-Token: <token>` or `?token=`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | App, host, selection, counts, uptime |
| GET | `/system` | Telemetry snapshot plus rolling history |
| GET | `/models` | Router inventory, health, setup payload |
| POST | `/models/refresh` | Probe every provider now |
| POST | `/models/switch` | Hot-swap `{kind, provider_id}` |
| POST | `/models/discover` | List the models at `{base_url}` before saving |
| GET | `/models/custom` | Cores added from the HUD |
| POST | `/models/custom` | Add a core and hot-load it |
| PUT | `/models/custom/{kind}/{id}` | Update a custom core |
| DELETE | `/models/custom/{kind}/{id}` | Remove a custom core |
| POST | `/routing/mode` | `{mode: "smart" | "manual"}` |
| GET/POST | `/setup` | Read / apply the required core selection |
| GET/POST | `/conversations` | List / create threads |
| GET/PATCH/DELETE | `/conversations/{id}` | Read, rename, erase |
| POST | `/conversations/{id}/clear` | Clear messages, keep the thread |
| POST | `/chat` | Non-streaming completion |
| POST | `/commands` | Run a slash command |
| GET/POST | `/memory`, DELETE `/memory/{id}` | Long-term facts |
| GET | `/events` | Diagnostic log |
| POST | `/tts` | Audio bytes from the selected voice core |
| POST | `/tts/preview` | One sentence in a named voice, for the picker |
| GET | `/models/{kind}/{id}/catalog` | Models and voices the core itself serves |
| GET/PATCH | `/preferences` | Model, voice and delivery tuning |
| POST | `/transcribe` | Multipart audio -> transcript |
| GET | `/health` | Provider health and uptime |
| GET | `/api/docs` | OpenAPI UI |

Errors are `{ok: false, error, code, details?}`. A model failure returns the
provider's diagnostic plus concrete hints (unreachable endpoint, missing model,
missing key), never a fabricated answer.

---

## Tuning: the model inside a core, and the voice behind it

Choosing a *core* answers which server to talk to. It does not answer the
question people actually ask — "which model is running, and which voice is
speaking?" One Ollama endpoint serves many models, one TTS server owns many
voices, and the same person may want fewer tokens in the morning than at night.
So the HUD has a second, explicit layer above the core matrix.

Two things open it, both of which read as controls rather than decoration:

- The **ACTIVE CORE** and **VOICE** readouts on the stage. They are buttons; the
  first opens the language tab, the second the voice tab.
- **CHOOSE MODEL & VOICE** in the Core Matrix panel, and the sliders icon in the
  top bar.

What it offers:

| Tab | Choice | Where the list comes from |
| --- | --- | --- |
| Language | Exact model name | `GET {base_url}/models` and native `/api/tags` from the core, not the YAML defaults |
| Language | Temperature, max tokens | Your own numbers, clamped to what a provider accepts |
| Voice | Voice, auditioned by clicking | Browser: `speechSynthesis.getVoices()`; server: the core's own voice API |
| Voice | Rate, pitch, speed | Rate and pitch for browser speech; speed for a server voice |
| Audio input | Transcriber model, language | Free text, passed straight to the STT core |

Nothing is preselected on your behalf: every picker opens on *Endpoint default*
or *Core default* until you choose. A model or voice you type by hand is accepted
too, because a server can own a name it does not list.

Tuning is stored apart from the core selection, in the settings table
(`PATCH /api/preferences`), and survives restarts. `null` clears an override and
returns the field to the core's own default; an omitted field is untouched.

Each section records the **core it was chosen on**. A model name like
`qwen2.5:14b` is meaningless on an endpoint that does not serve it, and a Kokoro
voice id means nothing to ElevenLabs, so switching a core clears that section's
model/voice and keeps the delivery numbers (temperature, tokens, rate, pitch,
speed) — those are not core-specific. This runs on `POST /api/models/switch` and
again at boot, in case a core was removed while the server was down.

That means the values are applied where they matter and nowhere else:

- the language choice rides on every `ChatRequest` as `model` and
  `temperature`/`max_tokens`;
- the voice rides on `/api/tts` as `voice` and `speed`;
- browser speech resolves the saved voice URI (or name) against the voices the
  browser currently exposes, falling back to an `en-GB` default if it vanished.

Server voices and browser voices are separate fields on purpose. A Kokoro voice
id like `af_bella` is meaningless to a browser engine, so switching between a
server core and the browser voice does not carry a stale name across.

The sliders that cannot apply to the engaged engine are disabled rather than
left live: RATE and PITCH belong to the browser speech engine, SPEED is sent to
a server voice core with each request.

### Reading a core's catalogue

```bash
curl http://127.0.0.1:8080/api/models/llm/local-qwen/catalog
```

```json
{"kind": "llm", "provider_id": "local-qwen", "type": "openai_compatible",
 "models": ["qwen2.5:7b"], "voices": [], "current_model": "qwen2.5:7b",
 "current_voice": null, "speed": null, "error": null}
```

A core that cannot enumerate reports why in `error` (the transport failure, per
URL) instead of inventing entries. A disabled core returns `409` with the reason,
and an unknown one `404`.

### Auditioning a voice

`POST /api/tts/preview` synthesizes one sentence with a specific voice and
returns the audio, so the picker can play a voice before you commit to it:

```bash
curl -X POST http://127.0.0.1:8080/api/tts/preview \
  -H 'Content-Type: application/json' \
  -d '{"provider_id": "local-kokoro", "voice": "af_bella"}' -o preview.wav
```

It deliberately bypasses the browser-TTS short circuit — the point is to hear a
*server* voice — and returns the provider's real diagnostic (`502`) when the core
is offline, rather than silence.

### What actually ran

The stage readout, the composer's right-hand label and the matrix's active card
all show the **tuned** model name when one is set, not the core's configured
default. During a turn the transcript header reports the model that served that
particular answer.

---

## Custom cores

`config/config.yaml` ships the providers that exist when you clone the repo, but
it is not the only way in. Anything you run yourself — a vLLM box on the LAN, an
Ollama daemon on another port, LM Studio, llama.cpp server, a second cloud key —
can be added from the HUD and is usable immediately, without editing YAML or
restarting.

Two entry points open the same dialog:

- **ADD CORE** on the setup screen (next to *Engage*), pre-set to the core kind
  you still have to choose.
- **ADD CORE** in the Core Matrix panel, pre-set to the tab you are looking at.

The dialog takes a kind (language / audio input / voice), an id, an endpoint, a
model, and optional credential, context window, capability flags and voice
settings. **PROBE** asks the endpoint what it actually serves before you commit:

```
GET {base_url}/models          OpenAI-compatible (vLLM, llama.cpp, LM Studio,
                               Ollama's /v1 shim, OpenRouter, Groq, ...)
GET {host}/api/tags            native Ollama
GET {base_url}/v1/models       when the URL has no /v1 suffix yet
```

Whichever answers first with a model list wins, and the model field becomes a
type-ahead over the real list. If nothing answers, the transport failure is
reported verbatim per URL — the dialog never guesses a model name.

Cores are stored in `data/runtime/providers.json` (mode `0600`), which is
git-ignored. The YAML file is never rewritten: the store is merged into the
model configuration at boot and after every change, so removals really remove
the provider because the pristine YAML view is always the base of the merge.

API keys for custom cores live in that same file and are **never returned by the
API**: descriptors report `api_key_set: true/false` and nothing more. An update
with a blank key keeps the stored one, so editing a model name never silently
loses your credential. The endpoints expose what the edit dialog needs to
prefill (`editable`) minus the secret.

Removing the core that is currently selected clears the saved setup and returns
you to the setup screen instead of quietly routing through a different model.
Custom cores are badged **YOURS** in both the setup list and the matrix, and
each row carries edit and remove controls; the server broadcasts
`models.changed` so other open tabs follow along.

```bash
curl -X POST http://127.0.0.1:8080/api/models/discover \
  -H 'Content-Type: application/json' \
  -d '{"base_url": "http://192.168.1.9:8000/v1"}'

curl -X POST http://127.0.0.1:8080/api/models/custom \
  -H 'Content-Type: application/json' \
  -d '{"kind": "llm", "id": "lab-vllm", "type": "openai_compatible",
       "base_url": "http://192.168.1.9:8000/v1", "model": "qwen2.5-coder:32b",
       "api_key": "sk-...", "context_window": 131072}'
```

Supported types per kind are the same sets `config.yaml` validates against:
`openai_compatible`/`anthropic`/`gemini` for language, `openai_compatible`/
`deepgram`/`faster_whisper`/`parakeet` for audio input, and
`openai_compatible`/`elevenlabs`/`cartesia`/`piper`/`f5_tts` for voice. Types
that run in-process (`faster_whisper`, `parakeet`, `piper`, `f5_tts`) hide the
endpoint field because they take no URL; `piper` asks for the local `.onnx`
voice file and `f5_tts` for the reference clip it clones, which are the fields
those types require. Anything the YAML supports but the dialog does not expose
(pricing, custom headers, arbitrary `options`) stays open to a hand-written YAML
provider; dashboard cores deliberately cover the common case.

A core is marked local when you say so, when it runs in-process, or when its
endpoint is loopback or a private address: a model server on your own LAN is not
someone else's cloud, and the badge should not imply otherwise.

---

## WebSocket protocol

Connect to `/ws` (append `?token=` when auth is enabled). Every frame is
`{"type": string, "data": object}`.

Server frames:

- `hello` — version, status, setup, telemetry, history, model inventory,
  conversations, messages, events, memory.
- `telemetry` — a fresh snapshot every two seconds.
- `health` — provider health after the 30-second sweep or a manual refresh.
- `setup.required`, `setup.complete` — selection lifecycle.
- `model.switched`, `routing.mode` — control-plane changes.
- `models.changed` — a core was added, updated or removed elsewhere.
- `chat.start`, `chat.meta`, `chat.delta`, `chat.tool`, `chat.usage`,
  `chat.done`, `chat.error`, `chat.cancelled` — streaming lifecycle.
- `command.result` — slash-command output plus an optional UI action.
- `conversation.*`, `memory.*`, `event` — archive, memory and log updates.

Client frames: `ping`, `telemetry.request`, `models.refresh`, `chat`, `cancel`,
`command`.

Streaming chat is request-id based, so a cancel targets exactly one generation.
The server persists the user message before calling the provider and the
assistant message only after a complete response. A provider fault is stored as
an `error` transcript message, and `chat.error` carries its `message_id` and
`error_message`, so the diagnostic survives a reload instead of vanishing with
the socket.

---

## Slash commands

```text
/status  /health  /models  /time
/switch <llm|stt|tts> <provider>
/mode <smart|manual>   /setup   /voice <on|off>
/remember <fact>  /memory  /forget <id|all>
/new [title]  /clear  /conversations  /about  /help
```

Commands are real operations: `/switch` hot-swaps the router and persists the
choice, `/remember` writes to SQLite, `/clear` wipes the active thread.

---

## The orb

The core stage is a canvas particle sphere (`js/orb.js`), not a decorative
loop. It reads two real signals:

- **state** — `idle`, `listening`, `processing`, `speaking` or `error`, each
  with its own palette and tempo, wired to the same `setCoreState()` that drives
  the caption.
- **level** — 0..1 loudness from `js/audio.js`, which runs an `AnalyserNode`
  over both the microphone and JARVIS's own speech. The sphere swells and
  brightens as you talk and as it talks back.

Three deliberate constraints:

- The boot screen uses the same orb, so the first thing you see is the thing
  that persists. It is stopped when the boot screen closes.
- `prefers-reduced-motion` draws one static frame and never schedules another.
- A hidden tab cancels the animation loop.

If no audio graph can run — Web Audio unavailable, a suspended context, a host
with no audio device — the level reads zero and the orb falls back to a quiet
synthetic pulse in the speaking and listening states, rather than sitting
still.

## Voice

- **Streamed speech**: answers are spoken a sentence at a time as they arrive,
  so the first words are heard while the rest is still being generated. The
  queue survives a fault per sentence: a server voice that fails mid-answer
  drops that sentence to the browser voice instead of going silent.
- **Interrupting**: pressing the microphone, or saying the wake word, stops the
  voice immediately. This is the only way to get a word in over a long answer.
- **Readable speech**: Markdown is cleaned before synthesis. Code becomes
  "(code omitted)", tables are dropped, and headings and bullets lose their
  glyphs and gain a full stop so they are not run into the next sentence.
- **TTS**: if the selected voice core is `browser`, speech is synthesized by
  the browser and works offline. Otherwise the HUD posts each sentence to
  `/api/tts` and plays the returned audio, falling back to the browser voice if
  the core is offline. The browser voice prefers an `en-GB` male voice to match
  the persona.
- **STT**: selecting `browser` uses `SpeechRecognition` where available;
  selecting a real STT provider records with `MediaRecorder` and posts to
  `/api/transcribe`. Without a microphone the keyboard always works.
- **Wake word**: the speaker icon arms `hey JARVIS` using the browser's own
  continuous recognition, stored in `localStorage` as `jarvis.wake`. It only
  acts on final results containing a wake phrase, pauses itself while JARVIS
  speaks or the microphone is in use (it must not hear its own voice), and
  disarms itself after two consecutive failures instead of restarting in a
  loop. It needs microphone permission and a browser with `SpeechRecognition`.

Both are opt-in. Audio is muted by default so a page load never surprises you
with sound.

---

## Security

- Default bind is `127.0.0.1`. Put a reverse proxy in front before exposing it.
- Set `JARVIS_API_TOKEN` for bearer auth; the HUD accepts `?token=` once and
  stores it locally.
- Set `server.cors_origins` to the origins you actually use; the development
  default is loopback only.
- The database and runtime selection store live under `data/`, which is
  git-ignored. `data/runtime/providers.json` holds custom-core credentials and is
  written `0600`; treat it like `.env`.
