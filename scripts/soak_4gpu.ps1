$ErrorActionPreference = "Stop"

$Run = ($env:ENKAI_RUN_MULTI_GPU_TESTS -eq "1")
$SingleGpuGreen = ($env:ENKAI_SINGLE_GPU_GREEN -eq "1")
$Launcher = $env:ENKAI_4GPU_LAUNCH_CMD
$MinHours = if ($env:ENKAI_4GPU_MIN_HOURS) { [int]$env:ENKAI_4GPU_MIN_HOURS } else { 3 }
$NCCL_TIMEOUT_SEC = if ($env:NCCL_TIMEOUT) { [int]$env:NCCL_TIMEOUT } else { 1800 }

function Skip($msg) {
    Write-Host "SKIPPED: $msg"
    exit 0
}

if (-not $Run) { Skip "ENKAI_RUN_MULTI_GPU_TESTS not set to 1" }
if (-not $SingleGpuGreen) { Skip "single-GPU gate not marked green (set ENKAI_SINGLE_GPU_GREEN=1 after soak pass)" }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Skip "nvidia-smi not available" }
$gpuCount = [int]((nvidia-smi -L | Measure-Object).Count)
if ($gpuCount -lt 4) { Skip "fewer than 4 GPUs detected" }
if ([string]::IsNullOrWhiteSpace($Launcher)) { Skip "ENKAI_4GPU_LAUNCH_CMD not set" }

Write-Host "Starting 4-GPU soak harness (single-node)"
Write-Host "Required minimum runtime (hours): $MinHours"
Write-Host "NCCL timeout (sec): $NCCL_TIMEOUT_SEC"
Write-Host "NCCL guidance: set NCCL_ASYNC_ERROR_HANDLING=1 and NCCL_TIMEOUT >= 1800 for long runs."

$start = Get-Date
iex $Launcher
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: launcher exited non-zero"
    exit 1
}

$hours = ((Get-Date) - $start).TotalHours
if ($hours -lt $MinHours) {
    Write-Host ("FAIL: runtime too short ({0:N2}h < {1}h)" -f $hours, $MinHours)
    exit 1
}

Write-Host ("PASS: 4-GPU soak completed ({0:N2}h)" -f $hours)
