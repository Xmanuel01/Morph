$ErrorActionPreference = "Stop"

$CheckpointDir = if ($env:ENKAI_CHECKPOINT_DIR) { $env:ENKAI_CHECKPOINT_DIR } else { "checkpoints/enkai_50m" }
$Config = if ($env:ENKAI_CONFIG) { $env:ENKAI_CONFIG } else { "" }
$ResumedFrom = if ($env:ENKAI_RESUMED_FROM_STEP) { [int]$env:ENKAI_RESUMED_FROM_STEP } else { 0 }
$LogPath = Join-Path $CheckpointDir "train_log.jsonl"
$stepDirs = Get-ChildItem -Path $CheckpointDir -Directory -Filter "step_*" -ErrorAction SilentlyContinue
if (-not $stepDirs -or $stepDirs.Count -eq 0) { throw "No step_* checkpoints found in $CheckpointDir" }
$Latest = ($stepDirs | Sort-Object Name | Select-Object -Last 1).FullName

$metaPath = Join-Path $Latest "meta.json"
if (-not (Test-Path $metaPath)) { throw "meta.json missing in $Latest" }

$meta = Get-Content $metaPath | ConvertFrom-Json
if (-not $meta.step) { throw "meta.step missing" }

if (-not (Test-Path $LogPath)) { throw "train_log.jsonl missing: $LogPath" }

$lines = Get-Content $LogPath | Where-Object { $_ -and $_.Trim().Length -gt 0 }
if ($lines.Count -eq 0) { throw "train_log.jsonl is empty" }
$last = $lines | Select-Object -Last 1 | ConvertFrom-Json
if ($ResumedFrom -gt 0 -and [int]$last.step -le $ResumedFrom) {
    throw "log.step did not advance after resume (last=$($last.step) resumed_from=$ResumedFrom)"
}
if ($ResumedFrom -gt 0 -and [int]$meta.step -lt $ResumedFrom) {
    Write-Host "latest checkpoint step has not rolled past resume step yet (expected when save_every > post-resume distance)"
}

$lastLoss = [double]$last.loss
if ([double]::IsNaN($lastLoss) -or [double]::IsInfinity($lastLoss)) {
    throw "Non-finite loss in log: $lastLoss"
}

$paramsPath = Join-Path $Latest "weights.bin"
if (-not (Test-Path $paramsPath)) { throw "weights.bin missing in $Latest" }

$optPath = Join-Path $Latest "optimizer.bin"
if (-not (Test-Path $optPath)) { throw "optimizer.bin missing in $Latest" }

$integrityPath = Join-Path $Latest "integrity.json"
if (-not (Test-Path $integrityPath)) {
    Write-Host "integrity.json missing (continuing; runtime checkpoint format may not emit this file)"
}

if ($Config -and (Test-Path $Config)) {
    if ($env:ENKAI_CHECK_EVAL -eq "1") {
        $out = & enkai eval $Config 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "enkai eval failed when loading checkpoint blobs: $out"
        }
    }
}

Write-Host "PASS: Checkpoint integrity OK"
Write-Host ("latest_step: {0}" -f $meta.step)
Write-Host ("last_loss: {0}" -f $lastLoss)
