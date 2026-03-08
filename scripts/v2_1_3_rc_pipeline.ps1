param(
    [switch]$AllowMissingGpuEvidence
)

# Backward-compatible wrapper for current release line.
powershell -ExecutionPolicy Bypass -File scripts/rc_pipeline.ps1 `
    -AllowMissingGpuEvidence:$AllowMissingGpuEvidence
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
