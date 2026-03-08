param(
    [switch]$VerifyGpuEvidence,
    [string]$GpuLogDir = "artifacts/gpu",
    [switch]$SkipPackageCheck
)

# Backward-compatible wrapper for the version-neutral release pipeline.
$pipelinePath = Join-Path $PSScriptRoot "release_pipeline.ps1"
$pipelineArgs = @{
    GpuLogDir = $GpuLogDir
}
if ($VerifyGpuEvidence.IsPresent) {
    $pipelineArgs["VerifyGpuEvidence"] = $true
}
if ($SkipPackageCheck.IsPresent) {
    $pipelineArgs["SkipPackageCheck"] = $true
}
& $pipelinePath @pipelineArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
