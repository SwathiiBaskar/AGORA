<#
.SYNOPSIS
    Runs the full quality gate: lockfile consistency, lint, format, strict typing,
    and the ENTIRE test suite (every commit's tests, not only the latest).

.DESCRIPTION
    Stops at the first failing step and exits with that step's exit code, so it is
    safe to use as a pre-commit check or from CI.

    Run from anywhere:
        powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    Write-Host ''
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Push/Pop so the caller's working directory is restored; Set-Location would leak.
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    Invoke-Step 'uv sync --locked (fails if uv.lock is out of date)' { uv sync --locked }
    Invoke-Step 'ruff check' { uv run ruff check . }
    Invoke-Step 'ruff format --check' { uv run ruff format --check . }
    Invoke-Step 'mypy (strict)' { uv run mypy }
    Invoke-Step 'pytest (entire suite)' { uv run pytest }

    Write-Host ''
    Write-Host 'All gates passed.' -ForegroundColor Green
}
finally {
    Pop-Location
}
