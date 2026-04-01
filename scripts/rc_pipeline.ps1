param(
    [string]$GpuLogDir = "artifacts/gpu",
    [switch]$AllowMissingGpuEvidence,
    [switch]$SkipPackageCheck
)

$ErrorActionPreference = "Stop"

function Get-PythonInvocation {
    foreach ($candidate in @("python3", "python", "py")) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            continue
        }
        if ($candidate -eq "py") {
            return @("py", "-3")
        }
        return @($candidate)
    }
    throw "Python interpreter not found (tried python3, python, py)."
}

function Invoke-Python {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    $python = Get-PythonInvocation
    $cmd = $python[0]
    $prefix = @()
    if ($python.Count -gt 1) {
        $prefix = $python[1..($python.Count - 1)]
    }
    & $cmd @prefix @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

if ($AllowMissingGpuEvidence) {
    Write-Host "[rc] Running RC dry-run without mandatory GPU evidence."
    $args = @("-ExecutionPolicy", "Bypass", "-File", "scripts/release_pipeline.ps1")
    if ($SkipPackageCheck.IsPresent) {
        $args += "-SkipPackageCheck"
    }
    & powershell @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "[rc] RC dry-run completed."
    exit 0
}

Write-Host "[rc] Running full RC pipeline (GPU evidence required)..."
$args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts/release_pipeline.ps1",
    "-VerifyGpuEvidence",
    "-GpuLogDir", $GpuLogDir
)
if ($SkipPackageCheck.IsPresent) {
    $args += "-SkipPackageCheck"
}
& powershell @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$blockerArgs = @(
    "run", "-p", "enkai", "--",
    "readiness", "verify-blockers",
    "--profile", "full_platform",
    "--report", "artifacts/readiness/full_platform.json",
    "--json",
    "--output", "artifacts/readiness/full_platform_blockers.json",
    "--allow-skipped-required-check", "selfhost-mainline",
    "--allow-skipped-required-check", "selfhost-stage0-fallback",
    "--require-gpu-evidence"
)
if ($SkipPackageCheck.IsPresent) {
    $blockerArgs += "--skip-release-evidence"
}
Write-Host "[rc] Verifying full-platform blocker matrix with GPU evidence..."
& cargo @blockerArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$collectArgs = @("scripts/collect_release_evidence.py", "--gpu-log-dir", $GpuLogDir, "--require-gpu")
$reportArgs = @("scripts/generate_capability_report.py", "--require-gpu")
if (-not $SkipPackageCheck.IsPresent) {
    $collectArgs += "--strict"
    $reportArgs += "--strict"
}
Invoke-Python @collectArgs
Invoke-Python @reportArgs
Write-Host "[rc] RC pipeline passed with archived evidence."
