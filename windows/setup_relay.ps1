# setup_relay.ps1 - one-time setup of the Pearl relay on Windows.
#
# Run this in PowerShell on the machine that runs pearld:
#
#     .\windows\setup_relay.ps1
#
# It will:
#   1. check that Python 3.10+ is available,
#   2. create .venv-relay and install the relay dependencies (requests),
#   3. write .env-relay.ps1 with your GitHub token (edit it afterwards).
#
# Afterwards start the relay with:
#     .\windows\start_relay.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Need($cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$cmd' not found in PATH." -ForegroundColor Red
        exit 1
    }
}

Need python

$pyVersion = (python --version) 2>&1
Write-Host "[setup] $pyVersion"

$venv = Join-Path $root ".venv-relay"
if (-not (Test-Path $venv)) {
    Write-Host "[setup] creating virtual environment $venv"
    python -m venv $venv
}

$pip = Join-Path $venv "Scripts\pip.exe"
& $pip install --upgrade pip
& $pip install -r requirements-relay.txt

$envFile = Join-Path $root ".env-relay.ps1"
if (-not (Test-Path $envFile)) {
    @"
# Pearl relay configuration - edit the values, then run start_relay.ps1.
`$env:GITHUB_TOKEN   = "PASTE_YOUR_GITHUB_TOKEN_HERE"
`$env:GITHUB_REPO    = "fgbvdcfgbfdefgb/peal"
`$env:GITHUB_BRANCH  = "main"
`$env:JOBS_FILE      = "jobs.txt"
`$env:SHARES_FILE    = "share.txt"

# pearl-gateway connection (defaults shown; only change if you configured
# the gateway differently).
`$env:MINER_RPC_TRANSPORT = "tcp"
`$env:MINER_RPC_HOST      = "127.0.0.1"
`$env:MINER_RPC_PORT      = "8337"
# `$env:MINER_RPC_TRANSPORT = "uds"
# `$env:MINER_RPC_SOCKET_PATH = "\\.\pipe\pearlgw"

# Optional tuning.
`$env:RELAY_POLL_SECONDS  = "1.0"
`$env:RELAY_SYNC_SECONDS  = "5.0"
"@ | Set-Content -Path $envFile -Encoding utf8
    Write-Host "[setup] wrote $envFile - EDIT IT and add your token."
} else {
    Write-Host "[setup] $envFile already exists, keeping it."
}

Write-Host ""
Write-Host "[setup] done.  Next steps:" -ForegroundColor Green
Write-Host "  1. edit $envFile and set GITHUB_TOKEN"
Write-Host "  2. make sure pearld and pearl-gateway are running"
Write-Host "  3. run: .\windows\start_relay.ps1"
