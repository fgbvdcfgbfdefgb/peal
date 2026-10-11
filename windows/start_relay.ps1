# start_relay.ps1 - run the Pearl relay (pearld <-> GitHub bridge).
#
# Prerequisites:
#   * pearld (release) running with a mining address configured
#   * pearl-gateway running and reachable (TCP 127.0.0.1:8337 by default)
#   * .env-relay.ps1 filled in (see setup_relay.ps1)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$envFile = Join-Path $root ".env-relay.ps1"
if (-not (Test-Path $envFile)) {
    Write-Host "ERROR: $envFile not found.  Run .\windows\setup_relay.ps1 first." -ForegroundColor Red
    exit 1
}
. $envFile

if ($env:GITHUB_TOKEN -like "PASTE*" -or -not $env:GITHUB_TOKEN) {
    Write-Host "ERROR: set GITHUB_TOKEN in $envFile first." -ForegroundColor Red
    exit 1
}

$venvPython = Join-Path $root ".venv-relay\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = "python"
    Write-Host "[relay] .venv-relay not found, using system python"
}

Write-Host "[relay] starting (repo=$env:GITHUB_REPO branch=$env:GITHUB_BRANCH)"
Write-Host "[relay] jobs -> $env:JOBS_FILE, shares <- $env:SHARES_FILE"
Write-Host "[relay] gateway: $env:MINER_RPC_TRANSPORT $env:MINER_RPC_HOST`:$env:MINER_RPC_PORT"
Write-Host "[relay] Ctrl+C to stop."
Write-Host ""

& $python -m relay
