# Start the JARVIS backend on Windows — the one command.
#
# First run: bootstraps whatever is missing (uv, Ollama, venv, packages, the
# ~5 GB of model downloads) by calling scripts\install.ps1, then serves the HUD
# and opens the browser. Later runs skip straight to serving. Ollama has to be
# running too; its installer usually leaves it in the tray, and
# scripts\start-ollama.ps1 covers the cases where it did not.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$python = Join-Path $backend ".venv\Scripts\python.exe"
$config = Join-Path $root "config\config.yaml"

if (-not (Test-Path $python)) {
    Write-Host "first run: bootstrapping (uv, Ollama, packages, ~5 GB of models)..."
    powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "install.ps1")
    if ($LASTEXITCODE -ne 0) { throw "bootstrap failed" }
}

$env:PYTHONPATH = "src"
Push-Location $backend
try {
    Write-Host "JARVIS starting on http://127.0.0.1:8080 (Ctrl+C to stop)"
    & $python -m jarvis.server.cli serve --config $config --open
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
