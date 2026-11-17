# Start the JARVIS backend on Windows.
#
# The equivalent of `make serve` for machines without make. Expects the venv and
# config that docs/windows.md sets up. Ollama has to be running too; the
# installer usually starts it for you, and scripts\start-ollama.ps1 covers the
# cases where it did not.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$python = Join-Path $backend ".venv\Scripts\python.exe"
$config = Join-Path $root "config\config.yaml"

if (-not (Test-Path $python)) {
    throw "venv not found at $python -- run the setup steps in docs/windows.md first"
}

$env:PYTHONPATH = "src"
Push-Location $backend
try {
    Write-Host "JARVIS starting on http://127.0.0.1:8080 (Ctrl+C to stop)"
    & $python -m jarvis.server.cli serve --config $config
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
