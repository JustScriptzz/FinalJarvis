# Model Router

The router is the only place the rest of JARVIS talks to models. It owns
provider instances, selection state, retries, fallbacks, circuit breakers,
usage accounting and normalized streaming.

```python
from jarvis.core.config import load_config
from jarvis.core.runtime_store import RuntimeSelectionStore
from jarvis.models import ModelRouter

config = load_config("config/config.yaml")
router = ModelRouter(
    config.models,
    runtime_store=RuntimeSelectionStore.in_data_dir(config.app.data_dir),
)
await router.start()
...
await router.aclose()
```

If you pass an `httpx.AsyncClient`, the router does not close it; close it
yourself. If you omit it, the router creates and closes one.

---

## Chat

```python
from jarvis.models import ChatMessage, ChatRequest

response = await router.chat(
    ChatRequest(
        messages=[
            ChatMessage(role="system", content="You are JARVIS."),
            ChatMessage(role="user", content="Plan a backup strategy."),
        ],
        temperature=0.2,
        max_tokens=1024,
        tools=[
            {
                "name": "read_file",
                "description": "Read a local file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        tool_choice="auto",
    )
)

print(response.text)
print(response.tool_calls)      # normalized, parsed arguments
print(response.usage)           # input/output/total/cached tokens
print(response.provider_id, response.model, response.cost_usd)
print(response.routing)         # profile, score, reasons, candidates
```

`ChatRequest.provider_id` pins an exact provider and disables fallback.
`ChatRequest.profile` pins a named smart-routing profile. They are mutually
exclusive.

### Multimodal

```python
from jarvis.models import ImagePart, TextPart

message = ChatMessage(
    role="user",
    content=[
        TextPart(text="What is wrong with this screenshot?"),
        ImagePart(data=png_bytes, media_type="image/png"),
    ],
)
```

URL images work with OpenAI-compatible providers. Anthropic and Gemini require
inline bytes and raise `CapabilityError` for URL-only images. The router filters
providers by `supports_vision` before calling them.

---

## Streaming

```python
async for event in router.stream_chat(request):
    match event.type:
        case "start":
            print("using", event.provider_id, event.model)
        case "text_delta":
            print(event.text, end="", flush=True)
        case "tool_call_delta":
            ...  # raw fragment, for early UI feedback
        case "tool_call":
            print("tool", event.tool_call.name, event.tool_call.arguments)
        case "usage":
            print(event.usage)
        case "done":
            final = event.response
        case "error":
            print(event.error.code, event.error.message)
```

Event contract:

| Event | Meaning |
|---|---|
| `start` | A provider attempt is starting. May appear more than once if a retry happens before the first token. |
| `text_delta` | Incremental answer text. |
| `tool_call_delta` | Raw provider fragment (`index`, optional `id`/`name`, JSON arguments fragment). |
| `tool_call` | Fully aggregated and JSON-parsed tool call, emitted before `done`. |
| `usage` | Token usage update. Providers may emit it once at the end. |
| `done` | Terminal success event with the complete `ChatResponse`. |
| `error` | Terminal failure after a stream had already emitted data, or a non-retryable error. The generator then raises. |

Retries only happen **before** the first event is emitted. Once a token has
reached the caller, the router does not silently restart the answer on another
model; it emits `error` and raises. This prevents duplicated speech in the
voice pipeline.

Tool argument JSON that cannot be parsed is preserved:

```python
call.arguments      # {} when parsing failed
call.raw_arguments  # original fragment
call.parse_error    # decoder message
```

The agent can repair or reject it instead of crashing.

---

## Speech to text

```python
from jarvis.models import TranscriptionRequest

result = await router.transcribe(
    TranscriptionRequest(
        audio=wav_bytes,
        filename="note.wav",
        media_type="audio/wav",
        language="en",          # None = provider auto-detect
        timestamps=True,
    )
)

print(result.text)
for segment in result.segments:
    print(segment.start, segment.end, segment.speaker, segment.text)
print(result.duration_s, result.provider_id, result.cost_usd)
```

Providers accept WAV, MP3, M4A, OGG and WebM bytes as delivered by the channel
adapters. `filename` and `media_type` are passed through to cloud APIs.
Local providers write the bytes to a temporary file and delete it in a worker
thread.

---

## Text to speech

```python
from jarvis.models import SpeechRequest

speech = await router.synthesize(
    SpeechRequest(
        text="Good morning. Three things need your attention.",
        voice="af_heart",
        format="wav",       # provider-dependent: wav, pcm, mp3
        sample_rate=24000,
        speed=1.0,
    )
)

# speech.audio is the original provider payload, not re-encoded.
open("briefing.wav", "wb").write(speech.audio)
```

`format` and `sample_rate` are requests, not guarantees. Read
`SpeechResult.format`, `sample_rate`, `media_type` and `duration_s` before
feeding audio into the realtime pipeline. Kokoro and OpenAI return WAV; Piper
and F5-TTS return raw PCM16; ElevenLabs and Cartesia may return MP3 unless
configured otherwise.

---

## Hot switching and reload

```python
await router.switch("llm", "local-fast")   # persisted by default
await router.switch("stt", "deepgram-nova")
await router.switch("tts", "elevenlabs")
```

`switch` validates that the provider exists and is enabled, updates the active
selection, and atomically writes `data/runtime/models.json`. The next request
uses the new provider; in-flight requests finish on the old one. The YAML file
is not rewritten.

```python
new_config = load_config("config/config.yaml")
await router.reload(new_config.models)
```

`reload` builds a fresh provider graph, swaps it under a lock, and retires old
providers after a 30-second grace period so active streams can drain. Circuit
breaker state is preserved for provider ids that still exist.

Dashboard-facing introspection:

```python
router.describe()                 # active, providers, profiles, breakers, metrics
await router.health()             # per-provider health probes
router.metrics()                  # requests, tokens, audio seconds, cost, latency
await router.list_llm_models()    # GET /models where supported
```

Health probes avoid billable generation calls. They hit free metadata
endpoints (`/models`, `/v1/models`, `/v1/user`, `/v1/projects`) or check local
binary/model availability.

---

## Routing algorithm

1. **Score the request** (`routing.assess_complexity`):
   - prompt length (8k and 24k token bands)
   - code fences and coding keywords
   - reasoning/architecture keywords
   - numbered multi-step structure or many questions
   - 8+ tools
   - 8+ conversation messages
   - `request.metadata["complexity"]` overrides the heuristic
2. **Pick a profile** from `fast_threshold` / `escalate_threshold`.
3. **Build a candidate chain**:
   - `deep`: deep -> balanced -> fast -> fallback
   - `balanced`: balanced -> fast -> fallback
   - `fast`: fast -> fallback
   - active provider is appended as the last resort
4. **Filter** by enabled flag and required capabilities.
5. **Execute** with per-provider retries, then fallback.
6. **Trip breakers** on retryable failures.

Explicit provider selection skips steps 2-4's degradation and raises a
`CapabilityError` if that provider cannot handle tools/vision/JSON.

### Capability flags

Set these per LLM provider in `config.yaml`:

```yaml
supports_tools: true
supports_vision: false
supports_json: true
supports_streaming: true
```

`supports_json: false` for Anthropic is intentional: the Messages API has no
`response_format`. The router will not send JSON-schema requests there unless
you explicitly pin the provider, in which case it raises `CapabilityError`.

---

## Provider configuration

Every provider shares:

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Disabled providers are listed but never selected. |
| `type` | required | Registry key (see README matrix). |
| `model` | per type | Model id, Ollama tag, or local model name. |
| `base_url` | per type | Required for `openai_compatible`; defaults for cloud. |
| `api_key` | `null` | Use `${ENV_VAR}`. Empty local keys are fine. |
| `local` | `false` | Marks the endpoint as local for UI and policy. |
| `timeout_s` | `60` | Per-request read/write timeout. |
| `connect_timeout_s` | `10` | TCP/TLS connect timeout. |
| `max_concurrency` | `4` | Semaphore around provider calls. |
| `max_attempts` | `3` | Retries for retryable failures. |
| `headers` | `{}` | Extra HTTP headers (OpenRouter attribution, etc.). |
| `options` | `{}` | Provider-specific switches. |
| `pricing` | zeros | `input_per_mtok`, `output_per_mtok`, `per_audio_minute`, `per_1k_chars`, `per_request`. |

LLM-only: `context_window`, `max_output_tokens`, capability flags.

STT-only: `language`, `device`, `compute_type`, `model_path`, `beam_size`,
`vad_filter`, `timestamps`, `diarize`, `sample_rate`.

TTS-only: `voice`, `voices`, `format`, `sample_rate`, `speed`, `language`,
`model_path`, `config_path`, `speaker`, `reference_audio`, `reference_text`,
`device`, `binary`.

### Local examples

**Ollama / vLLM / llama.cpp**

```yaml
local-qwen:
  type: openai_compatible
  base_url: http://localhost:11434/v1
  api_key: ollama
  model: qwen2.5:7b
  supports_tools: true
  supports_vision: false
  local: true
```

**Faster-Whisper**

```yaml
local-whisper:
  type: faster_whisper
  model: large-v3
  device: auto          # cpu / cuda
  compute_type: int8    # float16 on GPU
  vad_filter: true
  options:
    download_root: ./data/models/whisper
```

**Parakeet**

```yaml
local-parakeet:
  type: parakeet
  model: nvidia/parakeet-tdt-0.6b-v2
  device: auto
```

Requires `pip install 'jarvis-core[parakeet]'`. NeMo models are large and load
on first use.

**Kokoro / Chatterbox / Qwen3-TTS**

```yaml
local-kokoro:
  type: openai_compatible
  base_url: http://localhost:8880/v1
  model: kokoro
  voice: af_heart
  format: wav
  local: true
```

Any server exposing OpenAI's `/v1/audio/speech` works here. That is the
supported integration path for Chatterbox and Qwen3-TTS; no vendor-specific
protocol is assumed.

**Piper**

```yaml
local-piper:
  type: piper
  binary: piper
  model_path: ./models/piper/en_US-lessac-medium.onnx
  config_path: ./models/piper/en_US-lessac-medium.onnx.json
  sample_rate: 22050
  format: pcm
```

**Piper voice (the shipped default: a Piper model of the character voice)**

```yaml
local-jarvis:
  type: piper
  binary: piper
  model_path: ./data/models/piper/jarvis-medium.onnx
  config_path: ./data/models/piper/jarvis-medium.onnx.json
  options:
    volume: 0.8
```

**F5-TTS voice cloning (needs a GPU to be affordable)**

```yaml
local-f5:
  type: f5_tts
  model: F5TTS_v1_Base
  reference_audio: ./data/voices/reference.wav
  reference_text: "Exact transcript of the reference clip."
  device: auto
  format: pcm
  options:
    nfe_step: 32
    cfg_strength: 2.0
    allow_dynamic_reference: false
```

`allow_dynamic_reference: true` lets a request's `voice` field point at another
local clip. Keep it false unless the caller is trusted: it turns the voice
parameter into a file path.

### Cloud examples

```yaml
openai-gpt:
  type: openai_compatible
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY:-}
  model: gpt-4.1
  enabled: true
```

```yaml
anthropic-sonnet:
  type: anthropic
  api_key: ${ANTHROPIC_API_KEY:-}
  model: claude-sonnet-4-5-20250929
  supports_json: false
```

```yaml
deepgram-nova:
  type: deepgram
  api_key: ${DEEPGRAM_API_KEY:-}
  model: nova-3
  diarize: true
```

---

## Adding a provider

1. Implement the contract in `backend/src/jarvis/models/{llm,stt,tts}/`:

```python
class MyLLM(LLMProvider):
    type_name = "my_llm"

    async def stream(self, request: ChatRequest) -> AsyncIterator[LLMStreamChunk]:
        ...
```

2. Register it in `models/registry.py`.
3. Add the type to `LLM_TYPES` / `STT_TYPES` / `TTS_TYPES` in `core/config.py`.
4. Add a config entry and tests using `httpx.MockTransport`.

Providers must raise `ProviderError` subclasses from `_request` / `_stream` so
the router can decide retry and fallback. Do not catch and swallow HTTP errors
inside a provider.

---

## Error semantics

| Error | Retry | Fallback | Typical cause |
|---|---|---|---|
| `ProviderTimeoutError` | yes | yes | read/connect timeout, HTTP 408/504 |
| `ProviderUnavailableError` | yes | yes | connection refused, HTTP 5xx |
| `ProviderRateLimitError` | yes | yes | HTTP 429 (honors `Retry-After`) |
| `ProviderAuthError` | no | no | 401/403, missing API key |
| `ProviderBadRequestError` | no | no | 400/404/422, invalid tool schema |
| `CapabilityError` | no | no | vision/JSON/tools unsupported |
| `NoProviderAvailableError` | no | — | every candidate failed or was filtered |
| `CircuitOpenError` | no | yes | provider temporarily removed |

Backoff: `min(8s, 0.4 * 2^(attempt-1))` with ±25% jitter, or the server's
`Retry-After` capped at 30 seconds.

---

## Cost accounting

Costs are computed per successful call from the provider's `pricing` block:

- LLM: input/output tokens per million.
- STT: audio minutes.
- TTS: characters per thousand plus optional audio minutes.

Usage comes from the provider response. Local providers that omit usage report
zero tokens and zero cost; this is intentional rather than estimated. Update
pricing when providers change rates; the dashboard will surface these numbers
in Phase 5.

---

## Deliberate Phase 1 limits

- No streaming STT/TTS yet. The router exposes request/response APIs; the
  Phase 3 LiveKit pipeline will add streaming STT and sentence-chunked TTS at
  the transport layer.
- Moonshine is not a native provider. Faster-Whisper and Parakeet cover the
  local STT requirement; a Moonshine server can be added through the
  OpenAI-compatible STT path if it exposes that API.
- The router is process-local. Multi-process deployments need to move circuit
  breaker and selection state to Redis; the event bus in Phase 2 provides the
  hook.
- Health checks for local engines report package/model availability, not a
  loaded inference benchmark. The first real request performs the load.
