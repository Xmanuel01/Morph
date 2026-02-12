$ErrorActionPreference = "Stop"

$Config = if ($env:ENKAI_CONFIG) { $env:ENKAI_CONFIG } else { "configs/ce_sanity_300.enk" }
$CheckpointDir = if ($env:ENKAI_CHECKPOINT_DIR) { $env:ENKAI_CHECKPOINT_DIR } else { "checkpoints/ce_sanity_300" }

Write-Host "Running CE loss sanity with $Config"

& enkai train $Config
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: training command failed"
    exit 1
}

$logPath = Join-Path $CheckpointDir "train_log.jsonl"
if (-not (Test-Path $logPath)) {
    Write-Host "FAIL: log file missing: $logPath"
    exit 1
}

$steps = @()
$losses = @()
foreach ($line in Get-Content $logPath) {
    if (-not $line) { continue }
    $obj = $line | ConvertFrom-Json
    if ($obj.event -eq "step") {
        $step = [int]$obj.step
        $loss = [double]$obj.loss
        if ([double]::IsNaN($loss) -or [double]::IsInfinity($loss)) {
            Write-Host "FAIL: non-finite loss at step $step"
            exit 1
        }
        $steps += $step
        $losses += $loss
    }
}

if ($losses.Count -lt 3) {
    Write-Host "FAIL: insufficient step logs to evaluate trend"
    exit 1
}

$first = $losses[0]
$last = $losses[$losses.Count - 1]
$pass = $last -le ($first * 1.05)

Write-Host ("first_loss: {0}" -f $first)
Write-Host ("last_loss: {0}" -f $last)
Write-Host ("steps_logged: {0}" -f $losses.Count)
if ($pass) {
    Write-Host "PASS: CE loss shows expected non-divergent trend"
    exit 0
}

Write-Host "FAIL: CE loss did not improve enough"
exit 1
