param(
    [switch]$VerifyGpuEvidence,
    [string]$GpuLogDir = "artifacts/gpu",
    [switch]$SkipPackageCheck
)

# Backward-compatible wrapper for the version-neutral release pipeline.
powershell -ExecutionPolicy Bypass -File scripts/release_pipeline.ps1 `
    -VerifyGpuEvidence:$VerifyGpuEvidence `
    -GpuLogDir $GpuLogDir `
    -SkipPackageCheck:$SkipPackageCheck
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
