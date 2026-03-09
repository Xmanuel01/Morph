# Release Checklist

Use this checklist for production releases.

Note:
- Release pipeline scripts default `CARGO_BUILD_JOBS=1` and `CARGO_INCREMENTAL=0` when unset to reduce memory pressure.
- Optional: set `ENKAI_PIPELINE_STRIP_DEBUG=1` to run test gate with `-C debuginfo=0` on constrained Windows hosts.

## 1) Prep

- [ ] Pull latest `main`
- [ ] Update versions (workspace crates as needed)
- [ ] Update `CHANGELOG.md`
- [ ] Ensure release notes are ready

## 2) Build + QA

- [ ] `cargo fmt`
- [ ] `cargo clippy -- -D warnings`
- [ ] `cargo test`
- [ ] `python3 scripts/check_docs_consistency.py` (Linux/macOS)
- [ ] `powershell -ExecutionPolicy Bypass -File scripts/check_docs_consistency.ps1` (Windows)
- [ ] Frontend/serve contract snapshot test:
  - `cargo test -p enkai --bin enkai frontend::tests::contract_snapshots_match_reference_files`
- [ ] Benchmark target gate (v2.2.0 bounded suite):
  - `enkai bench run --suite official_v2_2_0 --baseline python --iterations 2 --warmup 1 --machine-profile bench/machines/linux_ref.json --output bench/results/official_v2_2_0.linux.json --target-speedup 5 --target-memory 5 --enforce-target`
  - Optional strict per-case mode:
    - add `--enforce-all-cases`
- [ ] Version-neutral release pipeline:
  - `powershell -ExecutionPolicy Bypass -File scripts/release_pipeline.ps1`
  - or `sh scripts/release_pipeline.sh`
- [ ] RC pipeline (GPU evidence required by default):
  - `powershell -ExecutionPolicy Bypass -File scripts/rc_pipeline.ps1`
  - or `sh scripts/rc_pipeline.sh`
- [ ] If GPU soak runs were executed, verify artifacts:
  - `powershell -ExecutionPolicy Bypass -File scripts/verify_gpu_gates.ps1 -LogDir artifacts/gpu`
  - or `sh scripts/verify_gpu_gates.sh artifacts/gpu`
- [ ] Complete `VALIDATION.md` gates for the target release
- [ ] Self-host mainline gate (Enkai-built compiler default lane):
  - `enkai litec mainline-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost`
  - expected triage artifacts:
    - `artifacts/selfhost/litec_selfhost_ci_report.json`
    - `artifacts/selfhost/litec_replace_check_report.json`
    - `artifacts/selfhost/litec_mainline_ci_report.json`
- [ ] Self-host Stage0 fallback gate:
  - `enkai litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus`
- [ ] Self-host replacement-readiness gate:
  - `enkai litec replace-check enkai/tools/bootstrap/selfhost_corpus --no-compare-stage0`
- [ ] Packaging reproducibility + checksum verification:
  - `python3 scripts/package_release.py --target-os linux --arch x86_64 --bin target/release/enkai --native target/release/libenkai_native.so --check-deterministic`
  - `python3 scripts/verify_release_artifact.py --archive dist/enkai-<version>-linux-x86_64.tar.gz --target-os linux --smoke`
  - Windows equivalent:
    `python scripts/package_release.py --target-os windows --arch x86_64 --bin target/release/enkai.exe --native target/release/enkai_native.dll --check-deterministic`
- [ ] Provenance/security gates:
  - `python3 scripts/license_audit.py`
  - `python3 scripts/generate_sbom.py --output dist/sbom-<version>-linux-x86_64.json`
- [ ] Archive release evidence package:
  - `python3 scripts/collect_release_evidence.py --gpu-log-dir artifacts/gpu --require-gpu --strict`
  - expected output: `artifacts/release/v<version>/manifest.json`
- [ ] Generate capability-complete report from archived evidence:
  - `python3 scripts/generate_capability_report.py --require-gpu --strict`
  - expected outputs:
    - `artifacts/release/v<version>/capability_complete.json`
    - `artifacts/release/v<version>/capability_complete.md`

## 3) Tag

- [ ] `scripts/release.sh vX.Y.Z` (creates annotated tag)
- [ ] Push branch + tag:
  - `git push origin main`
  - `git push origin vX.Y.Z`

## 4) GitHub Release

- [ ] Verify CI release workflow ran for tag
- [ ] Confirm assets:
  - `enkai-<version>-windows-x86_64.zip`
  - `enkai-<version>-linux-x86_64.tar.gz`
  - `enkai-<version>-linux-aarch64.tar.gz`
  - `enkai-<version>-macos-x86_64.tar.gz`
  - `enkai-<version>-macos-aarch64.tar.gz`
  - `enkai-setup-<version>.exe`
  - `.sha256` for each asset

## 5) Post-release

- [ ] Validate install scripts against the new release
- [ ] Smoke test:
  - `enkai --version`
  - `enkai run examples/...`
