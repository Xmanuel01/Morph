$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "gpu_harness.py"
if (-not (Test-Path $scriptPath)) {
    Write-Host "FAIL: missing harness runner $scriptPath"
    exit 1
}

$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    & $py.Source -3 $scriptPath multi
    exit $LASTEXITCODE
}

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    & $python.Source $scriptPath multi
    exit $LASTEXITCODE
}

Write-Host "SKIPPED: python runtime not found (install Python or use py launcher)"
exit 0
