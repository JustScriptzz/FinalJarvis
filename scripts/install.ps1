# One-command installer for JARVIS on Windows — the twin of scripts/install.sh.
#
#     powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -SkipModels
#     powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -WhisperModel small
#
# Idempotent: re-running verifies instead of downloading. Steps: uv, Ollama
# (winget if missing), the backend venv with the [dev] and [local] extras,
# .env, the three model downloads (~5 GB), and a config check.

param(
    [switch]$SkipModels,
    [string]$WhisperModel = "large-v3"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$python = Join-Path $backend ".venv\Scripts\python.exe"

function Step($msg) { Write-Host "`n== $msg" -ForegroundStyle White }

# --- 1. uv ----------------------------------------------------------------------
Step "uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:PATH"
}
uv --version

# --- 2. Ollama --------------------------------------------------------------------
Step "Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    } else {
        throw "Ollama not found and winget is unavailable; install it from https://ollama.com/download/windows"
    }
}

# --- 3/4. venv + packages -----------------------------------------------------------
Step "Backend environment"
Push-Location $backend
try {
    # Create the venv only when missing: an existing environment is reused as-is.
    if (-not (Test-Path $python)) {
        uv venv --python 3.12 .venv
    }
    uv pip install --python .venv\Scripts\python.exe -e ".[dev,local]"
    $env:PYTHONPATH = "src"

    # --- 5. .env ------------------------------------------------------------------
    Step "Environment file"
    $envFile = Join-Path $root ".env"
    if (-not (Test-Path $envFile)) {
        Copy-Item (Join-Path $root ".env.example") $envFile
        Write-Host "created .env from .env.example (localhost defaults work as-is)"
    } else {
        Write-Host ".env exists, left untouched"
    }

    # --- 6. models -----------------------------------------------------------------
    if ($SkipModels) {
        Step "Models (-SkipModels: skipped)"
        Write-Host "fetch later with:"
        Write-Host "  ollama pull qwen2.5:3b"
        Write-Host "  .venv\Scripts\python.exe -m scripts.fetch_whisper_model"
        Write-Host "  .venv\Scripts\python.exe -m scripts.fetch_piper_voice"
    } else {
        Step "Ollama model (qwen2.5:3b)"
        $up = $false
        try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 3 | Out-Null; $up = $true } catch {}
        if (-not $up) {
            Write-Host "ollama is not running; starting it in a hidden window"
            Start-Process powershell -WindowStyle Hidden -ArgumentList `
                "-ExecutionPolicy Bypass -File `"$root\scripts\start-ollama.ps1`""
            for ($i = 0; $i -lt 60 -and -not $up; $i++) {
                Start-Sleep -Seconds 1
                try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2 | Out-Null; $up = $true } catch {}
            }
            if (-not $up) { throw "Ollama did not come up; start it from the tray and re-run" }
        }
        ollama pull qwen2.5:3b

        Step "Speech input weights (whisper $WhisperModel)"
        & $python -m scripts.fetch_whisper_model --model $WhisperModel
        if ($LASTEXITCODE -ne 0) { throw "whisper fetch failed" }

        Step "JARVIS voice (Piper)"
        & $python -m scripts.fetch_piper_voice
        if ($LASTEXITCODE -ne 0) { throw "voice fetch failed" }
    }

    # --- 7. config check -------------------------------------------------------------
    Step "Config check"
    & $python -c "from jarvis.core.config import main; raise SystemExit(main())" (Join-Path $root "config\config.yaml")
    if ($LASTEXITCODE -ne 0) { throw "config check failed" }
} finally {
    Pop-Location
}

Step "Done"
Write-Host "start JARVIS with:  powershell -ExecutionPolicy Bypass -File scripts\start.ps1"
