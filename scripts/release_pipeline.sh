#!/usr/bin/env sh
set -eu

verify_gpu="${VERIFY_GPU_EVIDENCE:-0}"
gpu_log_dir="${GPU_LOG_DIR:-artifacts/gpu}"
skip_package="${SKIP_PACKAGE_CHECK:-0}"
strip_debug="${ENKAI_PIPELINE_STRIP_DEBUG:-0}"
bench_python="${ENKAI_BENCH_PYTHON:-}"

# Default to serialized rustc jobs in release gates for deterministic memory usage.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"

if [ -z "$bench_python" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    bench_python="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    bench_python="$(command -v python3)"
  else
    bench_python="$(command -v python)"
  fi
fi

echo "[release] Running consolidated production readiness gate..."
mkdir -p artifacts/selfhost artifacts/readiness
cargo run -p enkai -- readiness check \
  --profile production \
  --json \
  --output artifacts/readiness/production.json

echo "[release] Running bootstrap release lane gate..."
cargo run -p enkai -- litec release-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost

echo "[release] Running dependency license audit gate..."
python3 scripts/license_audit.py

echo "[release] Building release binaries for benchmark/package gates..."
cargo build -p enkai --release
cargo build -p enkai_native --release

echo "[release] Running benchmark target gate..."
mkdir -p dist
cargo run -p enkai --release -- bench run \
  --suite official_v2_3_0_matrix \
  --baseline python \
  --iterations 2 \
  --warmup 1 \
  --machine-profile bench/machines/linux_ref.json \
  --output dist/benchmark_official_v2_3_0_matrix_linux.json \
  --target-speedup 15 \
  --target-memory 5 \
  --enforce-target \
  --enforce-class-targets \
  --class-targets bench/suites/official_v2_3_0_targets.json \
  --python "$bench_python"

if [ "$skip_package" = "0" ]; then
  version="$(python3 scripts/current_version.py)"

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
