$ErrorActionPreference = "Stop"
$verifyGpu = if ($env:VERIFY_GPU_EVIDENCE) { $env:VERIFY_GPU_EVIDENCE } else { "0" }
$gpuLogDir = if ($env:GPU_LOG_DIR) { $env:GPU_LOG_DIR } else { "artifacts/gpu" }

Write-Host "[v1.8] Running format gate..."
cargo fmt --all --check

Write-Host "[v1.8] Running clippy gate..."
cargo clippy --workspace --all-targets -- -D warnings

Write-Host "[v1.8] Running test gate..."
cargo test --workspace

Write-Host "[v1.8] Running self-host corpus gate..."
cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus

Write-Host "[v1.8] Running self-host replacement fixed-point gate..."
cargo run -p enkai -- litec replace-check enkai/tools/bootstrap/selfhost_corpus --no-compare-stage0

if ($verifyGpu -eq "1") {
    Write-Host "[v1.8] Verifying GPU gate evidence..."
    powershell -ExecutionPolicy Bypass -File scripts/verify_gpu_gates.ps1 -LogDir $gpuLogDir
}

Write-Host "[v1.8] Release pipeline gates passed."
