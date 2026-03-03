$ErrorActionPreference = "Stop"

Write-Host "[v1.8] Running format gate..."
cargo fmt --all --check

Write-Host "[v1.8] Running clippy gate..."
cargo clippy --workspace --all-targets -- -D warnings

Write-Host "[v1.8] Running test gate..."
cargo test --workspace

Write-Host "[v1.8] Running self-host corpus gate..."
cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus

Write-Host "[v1.8] Release pipeline gates passed."
