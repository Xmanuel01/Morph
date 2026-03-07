param(
    [string]$LogDir = "artifacts/gpu"
)

$ErrorActionPreference = "Stop"

function Test-Pattern {
    param(
        [string]$Path,
        [string]$Pattern
    )
    if (-not (Test-Path $Path)) { return $false }
    $text = Get-Content -Path $Path -Raw
    return $text -match $Pattern
}

function Read-Json {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    return Get-Content -Path $Path -Raw | ConvertFrom-Json
}

function Require-SingleEvidence {
    param(
        [string]$SingleLog,
        [string]$SingleJson
    )
    $logOk = (Test-Pattern -Path $SingleLog -Pattern "status:\s*PASS") `
        -and (Test-Pattern -Path $SingleLog -Pattern "nan_or_inf:\s*False") `
        -and (Test-Pattern -Path $SingleLog -Pattern "checkpoint_verified:\s*True")
    if ($logOk) { return }

    $json = Read-Json -Path $SingleJson
    if ($null -eq $json) {
        throw "Missing single GPU evidence (need $SingleLog or $SingleJson)"
    }
    if ($json.status -ne "PASS") { throw "single GPU evidence status is not PASS ($SingleJson)" }
    if ($json.nan_or_inf) { throw "single GPU evidence reports non-finite values ($SingleJson)" }
    if (-not $json.checkpoint_verified) { throw "single GPU evidence reports unverified checkpoint ($SingleJson)" }
}

function Require-MultiEvidence {
    param(
        [string]$MultiLog,
        [string]$MultiJson
    )
    if (Test-Pattern -Path $MultiLog -Pattern "PASS:\s*2-GPU DP correctness validated") { return }

    $json = Read-Json -Path $MultiJson
    if ($null -eq $json) {
        throw "Missing multi GPU evidence (need $MultiLog or $MultiJson)"
    }
    if ($json.status -ne "PASS") { throw "2-GPU evidence status is not PASS ($MultiJson)" }
    if (-not $json.checks.loss_parity) { throw "2-GPU loss parity check failed ($MultiJson)" }
    if (-not $json.checks.grad_parity) { throw "2-GPU grad parity check failed ($MultiJson)" }
}

function Require-Soak4Evidence {
    param(
        [string]$SoakLog,
        [string]$SoakJson
    )
    if (Test-Pattern -Path $SoakLog -Pattern "PASS:\s*4-GPU soak completed") { return }

    $json = Read-Json -Path $SoakJson
    if ($null -eq $json) {
        throw "Missing 4-GPU evidence (need $SoakLog or $SoakJson)"
    }
    if ($json.status -ne "PASS") { throw "4-GPU evidence status is not PASS ($SoakJson)" }
}

$singleLog = Join-Path $LogDir "single_gpu.log"
$multiLog = Join-Path $LogDir "multi_gpu.log"
$fourLog = Join-Path $LogDir "soak_4gpu.log"

$singleJson = Join-Path $LogDir "single_gpu_evidence.json"
$multiJson = Join-Path $LogDir "multi_gpu_evidence.json"
$fourJson = Join-Path $LogDir "soak_4gpu_evidence.json"

Require-SingleEvidence -SingleLog $singleLog -SingleJson $singleJson
Require-MultiEvidence -MultiLog $multiLog -MultiJson $multiJson
Require-Soak4Evidence -SoakLog $fourLog -SoakJson $fourJson

Write-Host "GPU gate evidence verified."
