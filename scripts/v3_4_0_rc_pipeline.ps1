param(
    [switch]$AllowMissingGpuEvidence
)

# Backward-compatible wrapper for the v3.4.0 sign-off line.
$args = @("-ExecutionPolicy", "Bypass", "-File", "scripts/rc_pipeline.ps1")
if ($AllowMissingGpuEvidence.IsPresent) {
    $args += "-AllowMissingGpuEvidence"
}
& powershell @args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
