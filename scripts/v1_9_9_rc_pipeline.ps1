param(
    [string]$GpuLogDir = "artifacts/gpu",
    [switch]$AllowMissingGpuEvidence,
    [switch]$SkipPackageCheck
)

# Backward-compatible wrapper for v1.9.9 RC pipeline.
$args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts/rc_pipeline.ps1",
    "-GpuLogDir", $GpuLogDir
)
if ($AllowMissingGpuEvidence.IsPresent) {
    $args += "-AllowMissingGpuEvidence"
}
if ($SkipPackageCheck.IsPresent) {
    $args += "-SkipPackageCheck"
}
powershell @args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
