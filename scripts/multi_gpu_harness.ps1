$ErrorActionPreference = "Stop"

$Run = ($env:ENKAI_RUN_MULTI_GPU_TESTS -eq "1")
$SingleGpuGreen = ($env:ENKAI_SINGLE_GPU_GREEN -eq "1")
$Launcher = $env:ENKAI_DP_LAUNCH_CMD
$WorkDir = if ($env:ENKAI_DP_WORKDIR) { $env:ENKAI_DP_WORKDIR } else { "tmp/dp_harness" }
$TolLoss = if ($env:ENKAI_DP_LOSS_TOL) { [double]$env:ENKAI_DP_LOSS_TOL } else { 0.05 }
$TolGrad = if ($env:ENKAI_DP_GRAD_TOL) { [double]$env:ENKAI_DP_GRAD_TOL } else { 0.0001 }

function Skip($msg) {
    Write-Host "SKIPPED: $msg"
    exit 0
}

if (-not $Run) { Skip "ENKAI_RUN_MULTI_GPU_TESTS not set to 1" }
if (-not $SingleGpuGreen) { Skip "single-GPU gate not marked green (set ENKAI_SINGLE_GPU_GREEN=1 after soak pass)" }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Skip "nvidia-smi not available" }
$gpuCount = [int]((nvidia-smi -L | Measure-Object).Count)
if ($gpuCount -lt 2) { Skip "fewer than 2 GPUs detected" }
if ([string]::IsNullOrWhiteSpace($Launcher)) { Skip "ENKAI_DP_LAUNCH_CMD not set; provide launcher that emits rank artifacts" }

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$dataset = Join-Path $WorkDir "deterministic.txt"
$baselineLog = Join-Path $WorkDir "baseline.jsonl"
$rank0Log = Join-Path $WorkDir "rank0.jsonl"
$rank1Log = Join-Path $WorkDir "rank1.jsonl"
$rank0Grad = Join-Path $WorkDir "rank0_grads.json"
$rank1Grad = Join-Path $WorkDir "rank1_grads.json"

# Deterministic batch source.
1..200 | ForEach-Object { "sample $_ alpha beta gamma" } | Set-Content $dataset

Write-Host "Running baseline 1-GPU reference..."
if ($env:ENKAI_BASELINE_CMD) {
    iex $env:ENKAI_BASELINE_CMD
} else {
    Write-Host "SKIPPED: ENKAI_BASELINE_CMD not set"
    exit 0
}
if (-not (Test-Path $baselineLog)) { Skip "baseline log not produced: $baselineLog" }

Write-Host "Running 2-GPU launcher..."
iex $Launcher
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: launcher command failed"; exit 1 }

foreach ($p in @($rank0Log, $rank1Log, $rank0Grad, $rank1Grad)) {
    if (-not (Test-Path $p)) {
        Write-Host "FAIL: missing artifact $p"
        exit 1
    }
}

function LastLoss($path) {
    $line = Get-Content $path | Select-Object -Last 1
    return [double](($line | ConvertFrom-Json).loss)
}

$baseLoss = LastLoss $baselineLog
$r0Loss = LastLoss $rank0Log
$r1Loss = LastLoss $rank1Log
if (([Math]::Abs($r0Loss - $baseLoss) -gt $TolLoss) -or ([Math]::Abs($r1Loss - $baseLoss) -gt $TolLoss)) {
    Write-Host "FAIL: loss mismatch vs baseline (base=$baseLoss r0=$r0Loss r1=$r1Loss tol=$TolLoss)"
    exit 1
}

$g0 = Get-Content $rank0Grad | ConvertFrom-Json
$g1 = Get-Content $rank1Grad | ConvertFrom-Json
if ($g0.Count -ne $g1.Count) {
    Write-Host "FAIL: gradient length mismatch ($($g0.Count) vs $($g1.Count))"
    exit 1
}
for ($i = 0; $i -lt $g0.Count; $i++) {
    if ([Math]::Abs(([double]$g0[$i]) - ([double]$g1[$i])) -gt $TolGrad) {
        Write-Host "FAIL: allreduce mismatch at idx=$i (r0=$($g0[$i]) r1=$($g1[$i]) tol=$TolGrad)"
        exit 1
    }
}

Write-Host "PASS: 2-GPU DP correctness validated"
