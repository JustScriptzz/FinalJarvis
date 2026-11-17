# Running JARVIS on Windows

The backend is portable Python; the web interface is a browser page. What is
*not* portable is everything that is usually typed into a shell: venv paths,
`make`, and the Ollama starter. This page replaces those with PowerShell
equivalents. It is written for Windows 10/11 64-bit.

> **Honesty note** — this port was produced on Linux. The Python code paths are
> covered by the test suite, and the `.exe` handling is unit-tested, but the
> PowerShell scripts themselves were not executed on a Windows machine. Every
> command below is also spelled out so nothing depends on the scripts.

## One command

Everything on this page is folded into a single installer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

It is idempotent, and `-SkipModels` installs the code only. The sections below
are the same steps spelled out by hand, plus the parts that remain manual.

## What you need

| Piece | Why | Get it |
| --- | --- | --- |
| Python 3.12 or 3.13, 64-bit | the backend | [python.org](https://www.python.org/downloads/) — tick **Add python.exe to PATH** |
| Ollama for Windows | the text engine | [ollama.com/download/windows](https://ollama.com/download/windows) or `winget install Ollama.Ollama` |
| uv (recommended) | fast, repeatable installs | `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` |
| ~10 GB free disk | models + dependencies | — |

Plain `pip` works everywhere `uv pip` is shown below; `uv` just resolves faster
and prints what it changes.

## Install

```powershell
cd jarvis\backend
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
uv pip install --python .venv\Scripts\python.exe -e ".[local]"
```

`.venv\Scripts\python.exe` is the Windows venv layout; there is no
`.venv\bin\python`. Every command below uses the interpreter by path, so the
venv never has to be activated.

Pull the model weights (three downloads, ~5 GB total):

```powershell
ollama pull qwen2.5:3b                                            # text, 2 GB
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m scripts.fetch_whisper_model --model large-v3   # speech input, 3.1 GB
.venv\Scripts\python.exe -m scripts.fetch_piper_voice                      # JARVIS voice, 63 MB
```

The fetchers are idempotent; re-running them verifies instead of downloading.
Whisper accepts `--model small` (~480 MB) if disk or patience is short — set the
same name under `stt.providers.local-whisper.model` in `config\config.yaml`.

## Run

Ollama's installer normally leaves it running in the tray. To start it manually,
or to pin its thread count:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-ollama.ps1
```

It detects an already-running Ollama and exits cleanly instead of failing on the
port. Then the backend:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

or, without the scripts, exactly what they do:

```powershell
cd jarvis\backend
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m jarvis.server.cli serve --config ..\config\config.yaml
```

Open <http://127.0.0.1:8080>, pick the three cores in the setup screen, and the
HUD comes online. In PowerShell, use `curl.exe` (not `curl`, which aliases
`Invoke-WebRequest`) when copying the `curl` commands from the other docs.

## What changes on Windows

| Unix | Windows |
| --- | --- |
| `make serve` | `scripts\start.ps1` |
| `scripts/start-ollama.sh` | `scripts\start-ollama.ps1` |
| `.venv/bin/python` | `.venv\Scripts\python.exe` |
| `chmod 600` on the API-key store | NTFS user ACLs; the `0600` call is a safe no-op |
| `sh -n`, `node --check` | nothing to check: the UI JS runs in the browser |

The application code itself needed exactly one change: the Piper voice lookup
now also tries `piper.exe` beside the interpreter — `pip install piper-tts` puts
it in `.venv\Scripts`, and the server finds it there. `shutil.which` already
handled `PATHEXT`.

## GPU notes

- **Ollama** uses an NVIDIA GPU automatically on Windows; nothing to set.
- **Speech input** (`local-whisper`): `device: auto` in the config picks CUDA
  when the driver is present and `float16` arithmetic on it. That makes
  `large-v3` close to instant. A CPU-only machine stays on `int8`.
- **F5-TTS voice cloning** needs the CUDA build of PyTorch. Follow
  [docs/voices.md](voices.md); on Windows the GPU wheels are the default from
  PyPI, so a plain `pip install torch torchaudio` on a machine with a current
  NVIDIA driver is usually right.

## Troubleshooting

**`...start.ps1 cannot be loaded because running scripts is disabled`** —
the commands are the `powershell -ExecutionPolicy Bypass -File ...` form above,
or unblock once for the session: `Set-ExecutionPolicy -Scope Process Bypass`.

**`ollama pull` works but the backend says the engine is unreachable** —
Ollama is not running. Look for it in the tray, or run `scripts\start-ollama.ps1`.

**Browser has no microphone** — Edge and Chrome both work; grant the mic
permission when the HUD asks. Firefox works too, but its voice list for
browser-side TTS is thinner.

**A firewall dialog appears on first run** — the server binds to `127.0.0.1`
only; you can safely deny network access and everything still works locally.

**Tests** — the same suite that runs on Linux:

```powershell
cd jarvis\backend
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q
```
