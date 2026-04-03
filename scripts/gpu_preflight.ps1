param(
    [string]$Profile = "full",
    [string]$Config = "configs/enkai_50m.enk",
    [string]$Output = "artifacts/gpu/preflight.json",
    [string]$ArtifactDir = "artifacts/gpu"
)

$ErrorActionPreference = "Stop"

$py = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $py) {
    throw "py launcher not found"
}

& $py.Source -3 scripts/gpu_preflight.py --profile $Profile --config $Config --output $Output --artifact-dir $ArtifactDir
exit $LASTEXITCODE
