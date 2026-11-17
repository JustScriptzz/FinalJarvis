#!/bin/sh
# Start Ollama with a thread count chosen for the machine it runs on.
#
# Most machines need nothing from this script: a standard Ollama install finds
# its own engine and picks its own thread count, and the script falls through to
# a plain `ollama serve`. It exists for two host quirks that have both been
# observed in practice, and it only applies them when they are relevant.
#
# --- Quirk 1: the engine lives in a private directory -------------------------
#
# Some Ollama builds (notably the Termux/Android packaging) keep the inference
# engine and its ggml backend plugins outside PATH:
#
#   ${OLLAMA_LIB}/{llama-server,libggml*.so,libllama*.so,...}
#
# Two separate problems follow, and a server that ignores them answers
# /api/version quite happily while failing every single completion:
#
#   1. Ollama launches the engine as the relative path "lib/ollama/llama-server",
#      resolved against its own working directory. Started anywhere else it
#      reports "llama-server binary not found" and lists every location it
#      checked -- none of them the real one.
#
#   2. The engine locates its backend plugins relative to its own executable
#      path. A symlink in the expected layout is not enough: the plugins are then
#      sought in the symlink's directory, not found, and loading dies with
#      "llama_model_load_in_file_impl: no backends are loaded". A one-line
#      wrapper that execs the real absolute binary is enough, because exec
#      replaces the process image and the engine then sees its real home.
#
# --- Quirk 2: heterogeneous core speeds --------------------------------------
#
# On a big.LITTLE or hybrid CPU the cores are not equally fast. llama.cpp splits
# work across threads at every barrier, so each token waits for the slowest
# worker: using all cores is markedly slower than using only the fast ones.
# Measured on an 8-core ARM host (four 1.8 GHz, three 2.85 GHz, one 3.05 GHz):
#
#   threads:   8      6      4      2-3
#   tok/s:    1.6    2.4   2.4-2.9  2.5
#
# So the fast cluster is used, sized automatically. On a desktop with identical
# cores every core counts as fast and the full count is used, which is what you
# want there -- nothing is left on the table.
#
# --- The mmap flag ------------------------------------------------------------
#
# On a CPU, Ollama passes "--no-mmap", forcing the whole model into anonymous
# memory. On a host under memory pressure the kernel then swaps part of it out
# and every token touches a swapped page. Measured on the same 30-token answer:
#
#   --no-mmap (Ollama default)  55 s   with 402 MB of the model swapped
#   mmap     (this launcher)    11 s   with nothing swapped
#
# A machine with memory to spare loses nothing by keeping mmap enabled, so the
# launcher drops the flag when it builds the wrapper below.
#
# Usage:
#   ./scripts/start-ollama.sh                     # detect everything
#   OLLAMA_THREADS=12 ./scripts/start-ollama.sh   # override the thread count
#   OLLAMA_LIB=/path/to/lib ./scripts/start-ollama.sh
#   OLLAMA_HOST=0.0.0.0:11434 ./scripts/start-ollama.sh
set -eu

# --- thread count -------------------------------------------------------------

# Count the cores in the fastest frequency tier: those whose maximum frequency
# is within 10% of the highest. On a uniform CPU that is every core. Where
# cpufreq is unavailable (containers, some VMs) fall back to the online count.
detect_threads() {
    max=0
    for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq; do
        [ -r "$f" ] || continue
        v=$(cat "$f" 2>/dev/null) || continue
        case "$v" in '' | *[!0-9]*) continue ;; esac
        [ "$v" -gt "$max" ] && max=$v
    done
    if [ "$max" -gt 0 ]; then
        floor=$((max - max / 10))
        n=0
        for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq; do
            [ -r "$f" ] || continue
            v=$(cat "$f" 2>/dev/null) || continue
            case "$v" in '' | *[!0-9]*) continue ;; esac
            [ "$v" -ge "$floor" ] && n=$((n + 1))
        done
        if [ "$n" -gt 0 ]; then
            printf '%s\n' "$n"
            return
        fi
    fi
    nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4\n'
}

THREADS=${OLLAMA_THREADS:-$(detect_threads)}
export OLLAMA_NUM_THREADS="$THREADS"
export OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11434}

# --- find the engine ----------------------------------------------------------

if [ -z "${OLLAMA_LIB:-}" ]; then
    for candidate in \
        /data/data/com.termux/files/usr/lib/ollama \
        /usr/local/lib/ollama \
        /usr/lib/ollama \
        /opt/ollama/lib/ollama
    do
        if [ -x "$candidate/llama-server" ]; then
            OLLAMA_LIB=$candidate
            break
        fi
    done
fi

OLLAMA_ROOT=${OLLAMA_ROOT:-${TMPDIR:-/tmp}/jarvis-ollama-root}

if [ -z "${OLLAMA_LIB:-}" ] || [ ! -x "$OLLAMA_LIB/llama-server" ]; then
    # A standard install: Ollama resolves its own engine and its own plugin
    # paths, so there is nothing to work around. Serve directly.
    echo "start-ollama: no private engine directory; using ollama as installed" >&2
    echo "start-ollama: threads=$THREADS host=$OLLAMA_HOST" >&2
    exec ollama serve
fi

# --- compatibility layout -----------------------------------------------------

mkdir -p "$OLLAMA_ROOT/lib/ollama"

# Rewrite the launcher every run: an older run may have left a symlink here,
# which is exactly the case that fails the backend lookup described above.
rm -f "$OLLAMA_ROOT/lib/ollama/llama-server"
cat > "$OLLAMA_ROOT/lib/ollama/llama-server" <<EOF
#!/bin/sh
args=""
saw_threads=0
for arg in \$@; do
    case "\$arg" in
        --no-mmap) ;;                # dropped: see the note in the header
        -t|--threads) saw_threads=1; args="\$args \$arg" ;;
        --threads=*) saw_threads=1; args="\$args \$arg" ;;
        *) args="\$args \$arg" ;;
    esac
done
[ "\$saw_threads" -eq 1 ] || args="\$args -t $THREADS"
# shellcheck disable=SC2086
exec "$OLLAMA_LIB/llama-server" \$args
EOF
chmod +x "$OLLAMA_ROOT/lib/ollama/llama-server"

# Ollama prepends a relative "lib/ollama" to the child's LD_LIBRARY_PATH, so the
# absolute library directory has to be exported as well for the engine's own
# dependencies to resolve.
export LD_LIBRARY_PATH="$OLLAMA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "start-ollama: engine=$OLLAMA_LIB threads=$THREADS host=$OLLAMA_HOST" >&2

cd "$OLLAMA_ROOT"
exec ollama serve
