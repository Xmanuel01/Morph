param(
    [switch]$VerifyGpuEvidence,
    [string]$GpuLogDir = "artifacts/gpu"
)

$ErrorActionPreference = "Stop"

Write-Host "[v1.9] Running format gate..."
cargo fmt --all --check

Write-Host "[v1.9] Running clippy gate..."
cargo clippy --workspace --all-targets -- -D warnings

Write-Host "[v1.9] Running test gate..."
cargo test --workspace

Write-Host "[v1.9] Running docs contract consistency gate..."
powershell -ExecutionPolicy Bypass -File scripts/check_docs_consistency.ps1

Write-Host "[v1.9] Running self-host corpus gate..."
cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus

Write-Host "[v1.9] Running self-host replacement fixed-point gate..."
cargo run -p enkai -- litec replace-check enkai/tools/bootstrap/selfhost_corpus --no-compare-stage0

if ($VerifyGpuEvidence) {
    Write-Host "[v1.9] Verifying GPU gate evidence..."
    powershell -ExecutionPolicy Bypass -File scripts/verify_gpu_gates.ps1 -LogDir $GpuLogDir
}

Write-Host "[v1.9] Release pipeline gates passed."
