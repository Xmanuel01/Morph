param(
    [switch]$VerifyGpuEvidence,
    [string]$GpuLogDir = "artifacts/gpu"
)

$ErrorActionPreference = "Stop"

function Invoke-Gate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "[v1.9] Gate failed: $Name (exit code $LASTEXITCODE)."
    }
}

Write-Host "[v1.9] Running format gate..."
Invoke-Gate -Name "format" -Command { cargo fmt --all --check }

Write-Host "[v1.9] Running clippy gate..."
Invoke-Gate -Name "clippy" -Command { cargo clippy --workspace --all-targets -- -D warnings }

Write-Host "[v1.9] Running test gate..."
Invoke-Gate -Name "tests" -Command { cargo test --workspace }

Write-Host "[v1.9] Running docs contract consistency gate..."
Invoke-Gate -Name "docs-consistency" -Command {
    powershell -ExecutionPolicy Bypass -File scripts/check_docs_consistency.ps1
}

Write-Host "[v1.9] Running serve/frontend contract snapshot gate..."
Invoke-Gate -Name "contract-snapshots" -Command {
    cargo test -p enkai --bin enkai frontend::tests::contract_snapshots_match_reference_files
}

Write-Host "[v1.9] Running self-host corpus gate..."
Invoke-Gate -Name "selfhost-ci" -Command {
    cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus
}

Write-Host "[v1.9] Running self-host replacement fixed-point gate..."
Invoke-Gate -Name "selfhost-replace-check" -Command {
    cargo run -p enkai -- litec replace-check enkai/tools/bootstrap/selfhost_corpus --no-compare-stage0
}

if ($VerifyGpuEvidence) {
    Write-Host "[v1.9] Verifying GPU gate evidence..."
    Invoke-Gate -Name "verify-gpu-evidence" -Command {
        powershell -ExecutionPolicy Bypass -File scripts/verify_gpu_gates.ps1 -LogDir $GpuLogDir
    }
}

Write-Host "[v1.9] Release pipeline gates passed."
