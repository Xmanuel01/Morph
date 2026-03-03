param(
    [string]$LogDir = "artifacts/gpu"
)

$ErrorActionPreference = "Stop"

function Require-Pattern {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Message
    )
    if (-not (Test-Path $Path)) {
        throw "Missing log file: $Path"
    }
    $text = Get-Content -Path $Path -Raw
    if ($text -notmatch $Pattern) {
        throw "$Message ($Path)"
    }
}

$single = Join-Path $LogDir "single_gpu.log"
$multi = Join-Path $LogDir "multi_gpu.log"
$four = Join-Path $LogDir "soak_4gpu.log"

Require-Pattern -Path $single -Pattern "status:\s*PASS" -Message "single GPU soak did not report PASS"
Require-Pattern -Path $single -Pattern "nan_or_inf:\s*False" -Message "single GPU soak reported non-finite values"
Require-Pattern -Path $single -Pattern "checkpoint_verified:\s*True" -Message "single GPU soak did not verify checkpoints"

Require-Pattern -Path $multi -Pattern "PASS:\s*2-GPU DP correctness validated" -Message "2-GPU harness did not report PASS"
Require-Pattern -Path $four -Pattern "PASS:\s*4-GPU soak completed" -Message "4-GPU soak did not report PASS"

Write-Host "GPU gate evidence verified."
