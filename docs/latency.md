# Latency on a CPU-only host

Measured on the reference machine: a 7-core ARM big.LITTLE host (Cortex-A510 +
A710), no GPU, `qwen2.5:3b` at Q4_K_M under Ollama 0.30.10. These are real
numbers from this repository, not estimates. They are recorded because most of
the time went into problems that are invisible from the application code.

## The short version

| Stage | Before | After |
| --- | --- | --- |
| Cold turn (model loading) | 148 s | 148 s, first turn only |
| Warm turn, fresh conversation | 50–60 s | **6–7 s** for a 20-token answer |
| Re-evaluated prompt per turn | ~730 tokens | ~20 tokens (cached) |
| Time to first token | 10–40 s | **1–2 s** |
| Sustained generation | 0.5–1.6 tok/s | 2.4–2.9 tok/s |

Generation is the remaining floor. At ~2.5 tok/s a 48-token answer takes about
20 s of pure decoding; nothing above the engine can improve that on this host.

## What went wrong, in order of impact

### 1. The engine was swapping its own weights

Ollama passes `--no-mmap` to `llama-server` on CPU, which forces the whole model
into anonymous memory. Under memory pressure the kernel swapped part of it out,
and every generated token then touched a swapped page.

Same 30-token answer, same engine:

| Model memory | Time |
| --- | --- |
| `--no-mmap` (Ollama default), 402 MB swapped | 55 s |
| mmap (stripped by `scripts/start-ollama.sh`), 0 swapped | 11 s |

`scripts/start-ollama.sh` strips the flag in the launcher it generates. This is
the single largest fix and it is entirely outside the application.

### 2. All eight cores is the wrong number on a big.LITTLE chip

The engine's work is split into per-thread barriers, so every token waits for
the slowest worker. On a chip with two fast cores and six slow ones, using all
eight is markedly slower than using only the fast ones.

| Threads | Generation |
| --- | --- |
| 8 | 1.6 tok/s |
| 6 | 2.4 tok/s |
| 4 | **2.4–2.9 tok/s** |
| 2–3 | 2.5 tok/s |

The launcher injects `-t` itself because this Ollama build ignores
`OLLAMA_NUM_THREADS` and passes no `-t` of its own. The number is detected per
machine: it counts the cores in the fastest frequency tier (within 10 % of the
highest maximum frequency) and uses that, so this chip gets `-t 4` while a
uniform desktop CPU gets all of its cores. `OLLAMA_THREADS=12
./scripts/start-ollama.sh` overrides the detection.

### 3. A live timestamp in the system prompt defeated the prompt cache

This was the application-side bug, and the most expensive one because it silently
undid the cache on every turn.

Local engines reuse the longest identical prompt **prefix**. The original prompt
put a block of live telemetry — clock, CPU percentage, network rate — immediately
after the persona:

```
system: persona + "--- LIVE CONTEXT ---" + clock + CPU + memory + disk + ...
user:   <the question>
```

Two consequences, both measured:

* The clock changes every second, so the prefix was never identical. A ~450-token
  prompt cost ~9 s from scratch and ~0.2 s when cached.
* Worse, the history came *after* the volatile block, so it was re-evaluated too.

Measured with two prompts differing only in the clock:

| Layout | Reusable prefix |
| --- | --- |
| volatile block before history | 81% |
| volatile block after history | 98% |

But partial reuse is not reuse: with a changed byte in the middle, the engine
re-ran the tail. The engine log makes this unambiguous:

```
cached n_tokens = 0,   prompt eval 39452 ms / 710 tokens   # cache missed
cached n_tokens = 757, prompt eval  1458 ms /  20 tokens   # cache hit
```

### 4. The live context was merged into the user turn

The first fix moved telemetry to the end of the user's message:

```python
messages[-1] = ChatMessage(role="user", content=f"{user_text}\n\n{context}")
```

That looked right and was still wrong. The message persisted in the database is
the bare `user_text`, so on the next turn the history replays the question
*without* the context block. The prompt therefore diverges in the middle of the
history — at the previous turn — and everything after it is re-evaluated.

The fix is that the context is its own trailing message and is never persisted:

```python
messages.append(ChatMessage(role="user", content=user_text))
messages.extend(self.context_messages(user_text, telemetry, memories))
```

Because the trailing context is not part of the stored conversation, the next
turn reproduces every earlier message byte-for-byte. Verified: one turn's prompt
is a strict prefix of the next turn's, which is the property the cache needs.

The context is also attached only when the question is about the machine. It
costs ~90 tokens and, more importantly, a per-turn divergence. Matching is on
whole words, because a substring test fires on `"system"` inside `"systems
nominal"` — a phrasing this project's own tests use constantly.

### 5. Sliding history invalidated the prefix

`recent_messages(limit=24)` keeps the *most recent* 24 messages. Once a
conversation passes that, every turn drops messages from the front, which
changes the beginning of the prompt. The window is now `HISTORY_LIMIT = 200`
messages with a 48k-character budget applied newest-first, so the oldest turns
are dropped first and the recent ones stay stable.

Message ordering also had no tie-breaker: `ORDER BY created_at` with two messages
in the same millisecond could return them in either order. It now orders by
`(created_at, rowid)`.

## What "warm" means here

The first turn after a model load is always slow: the weights are read from disk
into RAM (and the page cache). On this host the first turn takes 148 s and the
next five take 6–7 s. Short bursts can exceed the sustained rate; the table above
uses 40–60 token answers, where the number is stable.

## Reproducing the measurement

```bash
# Engine-side truth: cached tokens, prompt eval and generation per request.
grep -a "cached n_tokens\|prompt eval time\|  eval time" /tmp/opencode/ollama.log | tail -8
```

```bash
# End-to-end, through the same HTTP API the browser uses.
CID=$(curl -s -X POST http://127.0.0.1:8080/api/conversations \
  -H 'Content-Type: application/json' -d '{"title":"latency"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["id"])')

for i in 1 2 3; do
  curl -s -X POST http://127.0.0.1:8080/api/chat \
    -H 'Content-Type: application/json' \
    -d "{\"message\":\"Tell me one short fact about the sea.\",\"conversation_id\":\"$CID\"}" \
    -o /dev/null -w "turn $i: %{time_total}s\n"
done
```

A fresh conversation is required for a clean measurement: reusing one that
already holds dozens of turns measures the accumulated history as well.

## If you have better hardware

Nothing here is JARVIS-specific tuning that would hurt a larger machine, but two
values are calibrated to this host:

* `OLLAMA_THREADS` (default 4) — set it to the number of *fast* cores.
* `tts.timeout_s` and the TTS core itself — `local-jarvis` (Piper) is real time
  on this CPU; `local-f5` needs a GPU to be affordable, as
  [voices.md](voices.md) explains.
