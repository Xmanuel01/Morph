#!/usr/bin/env sh
set -eu

verify_gpu="${VERIFY_GPU_EVIDENCE:-0}"
gpu_log_dir="${GPU_LOG_DIR:-artifacts/gpu}"
skip_package="${SKIP_PACKAGE_CHECK:-0}"
strip_debug="${ENKAI_PIPELINE_STRIP_DEBUG:-0}"

# Default to serialized rustc jobs in release gates for deterministic memory usage.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"

echo "[release] Running format gate..."
cargo fmt --all --check

echo "[release] Running clippy gate..."
cargo clippy --workspace --all-targets -- -D warnings

echo "[release] Running test gate..."
if [ "$strip_debug" = "1" ]; then
  RUSTFLAGS="${RUSTFLAGS:-} -C debuginfo=0" cargo test --workspace -j 1
else
  cargo test --workspace -j 1
fi

echo "[release] Running docs contract consistency gate..."
python3 scripts/check_docs_consistency.py

echo "[release] Running serve/frontend contract snapshot gate..."
cargo test -p enkai --bin enkai frontend::tests::contract_snapshots_match_reference_files

echo "[release] Running self-host mainline gate..."
mkdir -p artifacts/selfhost
cargo run -p enkai -- litec mainline-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost

echo "[release] Running self-host Stage0 fallback gate..."
cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus

echo "[release] Running dependency license audit gate..."
python3 scripts/license_audit.py

echo "[release] Running benchmark target gate..."
mkdir -p dist
cargo run -p enkai --release -- bench run \
  --suite official_v2_1_9 \
  --baseline python \
  --iterations 2 \
  --warmup 1 \
  --machine-profile bench/machines/linux_ref.json \
  --output dist/benchmark_official_v2_1_9_linux.json \
  --target-speedup 5 \
  --target-memory 5 \
  --enforce-target

if [ "$skip_package" = "0" ]; then
  version="$(python3 scripts/current_version.py)"

  echo "[release] Building release binaries for package gate..."
  cargo build -p enkai --release
  cargo build -p enkai_native --release

  echo "[release] Building deterministic package and checksum..."
  python3 scripts/package_release.py \
    --version "$version" \
    --target-os linux \
    --arch x86_64 \
    --bin target/release/enkai \
    --native target/release/libenkai_native.so \
    --check-deterministic

  archive="dist/enkai-${version}-linux-x86_64.tar.gz"
  echo "[release] Verifying package checksum/layout/smoke..."
  python3 scripts/verify_release_artifact.py \
    --archive "$archive" \
    --target-os linux \
    --smoke

  echo "[release] Generating SBOM artifact..."
  python3 scripts/generate_sbom.py --output "dist/sbom-${version}-linux-x86_64.json"
fi

if [ "$verify_gpu" = "1" ]; then
  echo "[release] Verifying GPU gate evidence..."
  sh scripts/verify_gpu_gates.sh "$gpu_log_dir"
fi

echo "[release] Release pipeline gates passed."
