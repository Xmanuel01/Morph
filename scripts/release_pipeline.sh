#!/usr/bin/env sh
set -eu

verify_gpu="${VERIFY_GPU_EVIDENCE:-0}"
gpu_log_dir="${GPU_LOG_DIR:-artifacts/gpu}"
skip_package="${SKIP_PACKAGE_CHECK:-0}"
strip_debug="${ENKAI_PIPELINE_STRIP_DEBUG:-0}"
release_min_free_gb="${ENKAI_RELEASE_MIN_FREE_GB:-4}"

# Default to serialized rustc jobs in release gates for deterministic memory usage.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"

assert_min_free_space() {
  path="$1"
  required_gb="$2"
  available_kb="$(df -Pk "$path" | awk 'NR==2 { print $4 }')"
  required_kb="$(awk "BEGIN { printf \"%d\", ${required_gb} * 1024 * 1024 }")"
  if [ "$available_kb" -lt "$required_kb" ]; then
    available_gb="$(awk "BEGIN { printf \"%.2f\", ${available_kb} / 1024 / 1024 }")"
    echo "[release] insufficient free disk space: ${available_gb} GiB available, ${required_gb} GiB required. Free space or set ENKAI_RELEASE_MIN_FREE_GB to a lower validated threshold." >&2
    exit 1
  fi
}

assert_min_free_space "." "$release_min_free_gb"

echo "[release] Running consolidated production readiness gate..."
mkdir -p artifacts/selfhost artifacts/readiness
cargo run -p enkai -- readiness check \
  --profile full_platform \
  --json \
  --output artifacts/readiness/full_platform.json \
  --skip-check selfhost-mainline \
  --skip-check selfhost-stage0-fallback

echo "[release] Running strict self-host contract freeze gate..."
cargo run -p enkai -- readiness check \
  --profile strict_selfhost \
  --json \
  --output artifacts/readiness/strict_selfhost.json

echo "[release] Running bootstrap release lane gate..."
cargo run -p enkai -- litec release-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost

echo "[release] Running dependency license audit gate..."
python3 scripts/license_audit.py

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

  echo "[release] Bootstrapping blocker verification artifact before evidence archive..."
  cargo run -p enkai -- readiness verify-blockers \
    --profile full_platform \
    --report artifacts/readiness/full_platform.json \
    --json \
    --output artifacts/readiness/full_platform_blockers.json \
    --skip-release-evidence \
    --allow-skipped-required-check selfhost-mainline \
    --allow-skipped-required-check selfhost-stage0-fallback

  echo "[release] Bootstrapping strict self-host blocker artifact before evidence archive..."
  cargo run -p enkai -- readiness verify-blockers \
    --profile strict_selfhost \
    --report artifacts/readiness/strict_selfhost.json \
    --json \
    --output artifacts/readiness/strict_selfhost_blockers.json \
    --skip-release-evidence

  echo "[release] Collecting strict release evidence bundle..."
  python3 scripts/collect_release_evidence.py --strict

  echo "[release] Verifying release blocker matrix against archived evidence..."
  cargo run -p enkai -- readiness verify-blockers \
    --profile full_platform \
    --report artifacts/readiness/full_platform.json \
    --json \
    --output artifacts/readiness/full_platform_blockers.json \
    --allow-skipped-required-check selfhost-mainline \
    --allow-skipped-required-check selfhost-stage0-fallback

  echo "[release] Verifying strict self-host blocker matrix against archived evidence..."
  cargo run -p enkai -- readiness verify-blockers \
    --profile strict_selfhost \
    --report artifacts/readiness/strict_selfhost.json \
    --json \
    --output artifacts/readiness/strict_selfhost_blockers.json

  echo "[release] Refreshing strict release evidence bundle with final blocker report..."
  python3 scripts/collect_release_evidence.py --strict

  echo "[release] Generating strict capability report..."
  python3 scripts/generate_capability_report.py --strict

  echo "[release] Generating release dashboard..."
  python3 scripts/generate_release_dashboard.py --strict
else
  echo "[release] Generating blocker verification artifact for reduced evidence mode..."
  cargo run -p enkai -- readiness verify-blockers \
    --profile full_platform \
    --report artifacts/readiness/full_platform.json \
    --json \
    --output artifacts/readiness/full_platform_blockers.json \
    --skip-release-evidence \
    --allow-skipped-required-check selfhost-mainline \
    --allow-skipped-required-check selfhost-stage0-fallback

  echo "[release] Generating strict self-host blocker artifact for reduced evidence mode..."
  cargo run -p enkai -- readiness verify-blockers \
    --profile strict_selfhost \
    --report artifacts/readiness/strict_selfhost.json \
    --json \
    --output artifacts/readiness/strict_selfhost_blockers.json \
    --skip-release-evidence

  echo "[release] Collecting reduced release evidence bundle (package checks skipped)..."
  python3 scripts/collect_release_evidence.py

  echo "[release] Generating reduced capability report (package checks skipped)..."
  python3 scripts/generate_capability_report.py

  echo "[release] Generating reduced release dashboard (package checks skipped)..."
  python3 scripts/generate_release_dashboard.py
fi

if [ "$verify_gpu" = "1" ]; then
  echo "[release] Verifying GPU gate evidence..."
  sh scripts/verify_gpu_gates.sh "$gpu_log_dir"
fi

echo "[release] Release pipeline gates passed."
