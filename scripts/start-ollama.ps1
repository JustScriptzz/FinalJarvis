# Start Ollama for JARVIS on Windows.
#
# The Unix version of this script (start-ollama.sh) exists because one Ollama
# build keeps its inference engine in a private directory and needs a wrapper to
# find it. The Windows build has no such quirk: the installer registers
# ollama.exe and it resolves its own engine. What remains useful here is a
# single place that:
#
#   * finds ollama.exe (PATH first, then the standard per-user install),
#   * reports when Ollama is already running (the installer starts it for you,
#     and a second `ollama serve` would fail with "address already in use"),
#   * sets OLLAMA_NUM_THREADS from the machine instead of leaving it implicit.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\start-ollama.ps1
#   $env:OLLAMA_THREADS = "12"; .\scripts\start-ollama.ps1
#   $env:OLLAMA_HOST    = "0.0.0.0:11434"; .\scripts\start-ollama.ps1

param(
    [string]$Threads = $env:OLLAMA_THREADS,
    [string]$OllamaHost = $(if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "127.0.0.1:11434" })
)

$ErrorActionPreference = "Stop"

# --- find ollama.exe ----------------------------------------------------------
$ollama = $null
$onPath = Get-Command ollama -ErrorAction SilentlyContinue
if ($onPath) {
    $ollama = $onPath.Source
} else {
    $installed = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $installed) {
        $ollama = $installed
    }
}
if (-not $ollama) {
    throw "ollama.exe not found. Install it from https://ollama.com/download/windows"
}

# --- already running? ----------------------------------------------------------
# The Ollama installer starts the service for you. Connecting to a second
# instance is not useful, so say so instead of failing on the port.
try {
    $version = Invoke-RestMethod -Uri "http://$OllamaHost/api/version" -TimeoutSec 3
    Write-Host "start-ollama: Ollama $($version.version) is already running on $OllamaHost"
    exit 0
} catch {
    # Nothing answered; we are the ones to start it.
}

# --- thread count ---------------------------------------------------------------
if (-not $Threads) {
    $Threads = [Environment]::ProcessorCount
}
$env:OLLAMA_NUM_THREADS = "$Threads"
$env:OLLAMA_HOST = $OllamaHost

Write-Host "start-ollama: engine=$ollama threads=$Threads host=$OllamaHost"
& $ollama serve
exit $LASTEXITCODE
