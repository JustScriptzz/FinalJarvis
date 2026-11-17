#!/bin/sh
# One-command installer for JARVIS (Linux and macOS).
#
#     ./scripts/install.sh                  # full install, models included
#     ./scripts/install.sh --skip-models    # code only, no downloads
#     ./scripts/install.sh --whisper-model small
#
# Everything here is idempotent: re-running verifies what exists instead of
# downloading again, so it is also the repair command for a half-finished
# setup. What it does, in order:
#
#   1. uv            -- installed to ~/.local/bin if missing
#   2. Ollama        -- installed via the official script if missing (Linux)
#   3. venv          -- backend/.venv with python 3.12 (uv fetches one if the
#                       system has none)
#   4. packages      -- the editable backend with the [dev] and [local] extras
#   5. .env          -- copied from .env.example, never overwritten
#   6. models        -- qwen2.5:3b into Ollama, the whisper large-v3 weights and
#                       the JARVIS Piper voice (about 5 GB; skipped with
#                       --skip-models)
#   7. config-check  -- proves the config loads before you start the server
#
# Windows has its own equivalent: scripts\install.ps1 (see docs/windows.md).

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BACKEND="$ROOT/backend"
WHISPER_MODEL="large-v3"
SKIP_MODELS=0

for arg in "$@"; do
    case "$arg" in
        --skip-models) SKIP_MODELS=1 ;;
        --whisper-model) shift_next=1 ;; # handled below
        --whisper-model=*) WHISPER_MODEL="${arg#*=}" ;;
        -h|--help)
            sed -n '2,27p' "$0"
            exit 0
            ;;
        *) echo "install: unknown option $arg (try --help)" >&2; exit 2 ;;
    esac
done
# POSIX-sh friendly two-pass for "--whisper-model NAME"
prev=""
for arg in "$@"; do
    if [ "$prev" = "--whisper-model" ]; then WHISPER_MODEL="$arg"; fi
    prev="$arg"
done

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

command -v curl >/dev/null 2>&1 || {
    echo "install: curl is required; install it with your package manager" >&2
    exit 1
}

# --- 1. uv --------------------------------------------------------------------
step "uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -fsSL https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; export PATH
fi
uv --version

# --- 2. Ollama ------------------------------------------------------------------
step "Ollama"
if ! command -v ollama >/dev/null 2>&1; then
    case "$(uname -s)" in
        Linux) curl -fsSL https://ollama.com/install.sh | sh ;;
        Darwin)
            echo "install: on macOS, install Ollama from https://ollama.com/download/mac" >&2
            echo "         (or 'brew install ollama'), then re-run this script." >&2
            exit 1
            ;;
        *) echo "install: unsupported OS $(uname -s)" >&2; exit 1 ;;
    esac
fi
ollama --version || true

# --- 3/4. venv + packages -------------------------------------------------------
step "Backend environment"
cd "$BACKEND"
# Create the venv only when missing: an existing environment is reused as-is,
# whatever its patch version, so re-runs never churn a working setup.
if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -e ".[dev,local]"

# --- 5. .env ---------------------------------------------------------------------
step "Environment file"
if [ ! -f "$ROOT/.env" ]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "created .env from .env.example (localhost defaults work as-is)"
else
    echo ".env exists, left untouched"
fi

# --- 6. models --------------------------------------------------------------------
if [ "$SKIP_MODELS" -eq 1 ]; then
    step "Models (--skip-models: skipped)"
    echo "fetch later with:"
    echo "  ollama pull qwen2.5:3b"
    echo "  (cd backend && PYTHONPATH=src .venv/bin/python -m scripts.fetch_whisper_model)"
    echo "  (cd backend && PYTHONPATH=src .venv/bin/python -m scripts.fetch_piper_voice)"
else
    step "Ollama model (qwen2.5:3b)"
    if ! curl -fsS -m 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
        echo "ollama is not running; starting it via scripts/start-ollama.sh"
        nohup sh "$ROOT/scripts/start-ollama.sh" >/tmp/jarvis-ollama.log 2>&1 &
        i=0
        while [ "$i" -lt 60 ]; do
            curl -fsS -m 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
            i=$((i + 1)); sleep 1
        done
        curl -fsS -m 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1 || {
            echo "install: Ollama did not come up; see /tmp/jarvis-ollama.log" >&2
            exit 1
        }
    fi
    ollama pull qwen2.5:3b

    step "Speech input weights (whisper $WHISPER_MODEL)"
    PYTHONPATH=src .venv/bin/python -m scripts.fetch_whisper_model --model "$WHISPER_MODEL"

    step "JARVIS voice (Piper)"
    PYTHONPATH=src .venv/bin/python -m scripts.fetch_piper_voice
fi

# --- 7. config check ----------------------------------------------------------------
step "Config check"
PYTHONPATH=src .venv/bin/python -c \
    "from jarvis.core.config import main; raise SystemExit(main())" "$ROOT/config/config.yaml"

step "Done"
echo "start JARVIS with:  make serve-open      (or scripts/start-ollama.sh first, if you used --skip-models)"
