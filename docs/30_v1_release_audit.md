# Enkai v1.0.0-v2.0.0 Delivery And Production-Grade Audit

## Purpose

This document records what shipped from `v1.0.0` through `v2.0.0` and the current production-readiness status.

## Source of truth used for this record

- Git tags: `v1.0.0` through `v2.0.0`.
- `CHANGELOG.md`.
- `docs/Enkai.spec` compatibility and runtime sections.
- `VALIDATION.md`.
- `docs/RELEASE_CHECKLIST.md`.
- Release pipeline run on `2026-03-07`:
  - `powershell -ExecutionPolicy Bypass -File scripts/release_pipeline.ps1`

## Release ledger (v1.0.0-v2.0.0)

### v1.0.0 - Production core freeze

- Grammar/CLI baseline frozen to v0.9.3 compatibility contract.
- Train/Eval config schema v1 enforced (`config_version: 1`) with strict validation.
- Checkpoint metadata schema v1 enforced (`format_version: 1`) with legacy fallback support.
- Training path locked to TinyLM CE forward (`forward_tinylm`), avoiding `forward_l2` in train loop.
- Formalized release governance with `VALIDATION.md` and `docs/RELEASE_CHECKLIST.md`.

### v1.1.0 - Runtime semantics + language-level ML primitives

- Runtime semantics implemented for `type`, `enum`, `impl` (including method dispatch).
- Runtime behavior implemented for `tool`, `agent`, `prompt`, `model`, `memory` declarations.
- Added `std::nn`, `std::loss`, `std::optim` surfaces.
- Added deterministic seed wiring across training/tokenizer/dataset flows.
- Added checker/runtime coverage for `Tensor`, `Device`, `DType`, `Shape` contracts.

### v1.2.0 - Scale engine + systems foundation

- Added distributed/runtime wiring for multi-rank training controls.
- Added grad accumulation, grad clipping, AMP config path.
- Added ranked checkpoint manifests and compatibility handling for distributed flow.
- Added dataset prefetch and packing-efficiency metrics.
- Added deterministic dependency lock + build caching (`enkai.lock`, `enkai build`).
- Added stdlib expansion: `std::env`, `std::path`, `std::process`, `std::time`, `std::io`, `std::log`.

### v1.3.0 - LLM backend platform

- Added `enkai serve` lifecycle for model-serving entry.
- Added routed HTTP serving + middleware + SSE/WebSocket streaming endpoints.
- Added auth/rate-limit middleware and structured response metadata headers.
- Added model registry helpers and version pinning flow.
- Added `std::db` (SQLite + Postgres) and `std::tls` helper surfaces.

### v1.4.0 - Frontend developer stack

- Added scaffolds:
  - `enkai new backend`
  - `enkai new frontend-chat`
  - `enkai new fullstack-chat`
- Added typed SDK generator: `enkai sdk generate`.
- Added frontend/backend API version pinning contract.
- Added end-to-end fullstack contract tests for streaming + version mismatch + persistence flow.

### v1.5.0 - Bootstrap lite

- Added bootstrap-lite commands:
  - `enkai fmt-lite`
  - `enkai lint-lite`
  - `enkai tokenizer-lite`
  - `enkai dataset-lite`
- Added Enkai-scripted bootstrap-lite tool implementations running through VM.
- Added runtime/compiler `bootstrap` module and checker built-ins.
- Added deterministic parity tests against Rust baseline paths.
- Added bootstrap subset spec: `docs/bootstrap_subset.md`.

### v1.6.0 - Bootstrap core

- Added bootstrap-core `litec` commands:
  - `enkai litec check`
  - `enkai litec compile`
  - `enkai litec verify`
- Added runtime/compiler `compiler` module:
  - `parse_subset`
  - `check_subset`
  - `emit_subset`
- Added Stage0/Stage1 bytecode equivalence flow and tests.
- Added bootstrap-core spec: `docs/bootstrap_core.md`.

### v1.7.0 - Self-host beta

- Added self-host command: `enkai litec selfhost <corpus_dir>`.
- Added staged frontend command: `enkai litec stage <parse|check|codegen>`.
- Added CI-targeted self-host gate: `enkai litec selfhost-ci`.
- Expanded bootstrap subset to support `use`, `type`, `enum`, `impl`, non-capturing lambdas for corpus coverage.
- Added CI lane for self-host beta regression coverage.

### v1.8.0 - Compatibility and operational self-host readiness

- Added compatibility/deprecation policy: `docs/29_compatibility_policy.md`.
- Added self-host operational workflow: `docs/28_selfhost_workflow.md`.
- Added consolidated release scripts:
  - `scripts/v1_8_release_pipeline.ps1`
  - `scripts/v1_8_release_pipeline.sh`
- Added explicit legacy compatibility gates:
  - configs without `config_version`
  - checkpoints without `format_version`

### v1.9.0 - Unified release pipeline + replacement-readiness gates

- Added stage1 run command: `enkai litec run <input.enk>`.
- Added replacement-readiness command:
  - `enkai litec replace-check <corpus_dir> [--no-compare-stage0]`
- Added master release smoke test:
  - `master_pipeline_cpu_smoke`
- Added consolidated release scripts:
  - `scripts/v1_9_release_pipeline.ps1`
  - `scripts/v1_9_release_pipeline.sh`
- Added GPU evidence verification scripts:
  - `scripts/verify_gpu_gates.ps1`
  - `scripts/verify_gpu_gates.sh`
- Added explicit distributed opt-in guard:
  - `ENKAI_ENABLE_DIST=1`

### v1.9.2 - Contract integrity and release hygiene

- Added build-time language version source wiring for CLI/version consistency.
- Fixed CI workflow correctness for package validation and added workflow linting.
- Added docs consistency checks to release/CI gates.
- Aligned spec/README/validation docs for v1.9 contract consistency.

### v1.9.3 - Runtime semantics and safety hardening

- Added stable machine-parseable runtime error codes for policy/tool failures.
- Hardened tool command/process error mapping for config, spawn, timeout, wait, IO, payload, and output-format failures.
- Expanded integration coverage for coded failure behavior:
  - tool timeout and spawn failure codes
  - deterministic policy denial code coverage for tool/http/process/db/fs paths
- Declared bytecode VM as normative production runtime contract and legacy interpreter as compatibility/reference only.

### v1.9.4 - Contract migration and readiness tooling

- Added migration CLI:
  - `enkai migrate config-v1 <in_config.enk> <out_config.enk|out.json>`
  - `enkai migrate checkpoint-meta-v1 <checkpoint_dir> [--dry-run] [--verify]`
- Added readiness scanner:
  - `enkai doctor [path]`
- Added checkpoint metadata contract validation and cross-tree consistency checks for:
  - `config_hash`
  - `model_sig`
  - `dtype`
  - `device`
- Added fixture-backed migration/doctor regression tests.
- Updated spec/policy/readme/docs to v1.9.4 and documented migration workflow.

### v1.9.5 - Distributed reliability productization

- Added first-party multi-rank harness runner:
  - `scripts/gpu_harness.py` with `multi` and `soak4` modes.
- Updated Windows/Linux wrappers to use first-party harness flow:
  - `scripts/multi_gpu_harness.ps1/.sh`
  - `scripts/soak_4gpu.ps1/.sh`
- Added structured single-GPU evidence output:
  - `artifacts/gpu/single_gpu_evidence.json`
  - `artifacts/gpu/single_gpu.log`
- Added structured 2-GPU/4-GPU evidence outputs:
  - `artifacts/gpu/multi_gpu_evidence.json`
  - `artifacts/gpu/soak_4gpu_evidence.json`
- Hardened distributed error contracts with stable machine-parseable codes:
  - `E_DIST_ENV_GATE`, `E_DIST_FEATURE_MISSING`, `E_DIST_ENV_MISMATCH`,
    `E_DIST_DEVICE_MAPPING`, `E_DIST_CUDA_COUNT`, `E_DIST_CUDA_UNAVAILABLE`,
    `E_DIST_NOT_INITIALIZED`, and related runtime guards.
- Updated GPU evidence verification scripts for JSON evidence compatibility while
  preserving legacy log parsing:
  - `scripts/verify_gpu_gates.ps1`
  - `scripts/verify_gpu_gates.sh`
- Expanded distributed guard/harness regression coverage in:
  - `enkai_tensor/tests/dist_guards.rs`
  - `enkai_tensor/tests/multi_gpu_harness.rs`

### v1.9.6 - Serve/frontend contract freeze

- Added explicit scaffolded contract snapshots:
  - `backend/contracts/backend_api.snapshot.json`
  - `backend/contracts/conversation_state.schema.json`
  - `frontend/contracts/sdk_api.snapshot.json`
- Added CI/release gate for snapshot freeze:
  - `frontend::tests::contract_snapshots_match_reference_files`
- Expanded scaffolded backend API contract with WebSocket route:
  - `GET /api/<version>/chat/ws`
- Hardened generated SDK contract:
  - deterministic error-detail parsing
  - explicit WebSocket streaming helper (`streamChatWs`)
- Versioned scaffold persistence contract:
  - `conversation_state.json` includes `schema_version: 1`
  - startup migration hook upgrades legacy v0-style persisted state.

### v1.9.7 - Packaging reproducibility + provenance hardening

- Added deterministic release packaging and checksum tooling:
  - `scripts/package_release.py`
  - `scripts/verify_release_artifact.py`
- Added version-neutral release pipeline scripts:
  - `scripts/release_pipeline.ps1`
  - `scripts/release_pipeline.sh`
- Kept backward-compatible v1.9 pipeline wrappers:
  - `scripts/v1_9_release_pipeline.ps1`
  - `scripts/v1_9_release_pipeline.sh`
- Added provenance/security gates:
  - `scripts/license_audit.py`
  - `scripts/generate_sbom.py`
- Updated CI and release workflows for:
  - cross-platform package checksum validation (Linux + Windows)
  - deterministic archive checks (`--check-deterministic`)
  - SBOM artifact generation.

### v2.0.0 - Stability cut (strict contracts enforced)

- Enforced strict contracts on default train/eval acceptance paths:
  - configs missing `config_version` are rejected.
  - checkpoints missing required v1 metadata are rejected.
- Added explicit temporary recovery path for operators:
  - `--lenient-contracts` on train/eval requires `ENKAI_ALLOW_LEGACY_CONTRACTS=1`.
- Hardened migration/readiness tooling:
  - `enkai migrate checkpoint-meta-v1 --verify --strict-contracts`
  - `enkai doctor --json` strict-by-default machine-readable report
- Added release-line wrappers:
  - `scripts/v2_0_0_rc_pipeline.ps1`
  - `scripts/v2_0_0_rc_pipeline.sh`
- Added strict contract regression coverage in `enkai/src/train.rs`, `enkai/src/migrate.rs`, and `enkai/src/main.rs`.

## Production-grade audit status

### Automated gates (executed)

Run executed on `2026-03-07`:

- `cargo fmt --all -- --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `powershell -ExecutionPolicy Bypass -File scripts/release_pipeline.ps1` -> PASS
- `powershell -ExecutionPolicy Bypass -File scripts/rc_pipeline.ps1 -AllowMissingGpuEvidence` -> PASS (dry-run, no mandatory GPU evidence)

### Coverage conclusion

- Language/runtime/core CLI: production-grade under automated CPU gates.
- Backend/serve/frontend scaffolds: production-grade under automated integration tests.
- Bootstrap-lite/core/self-host CI lanes: production-grade under deterministic parity/fixed-point gates.
- Compatibility controls (legacy config/checkpoint): production-grade under migration + doctor tests.

### Remaining operator-required evidence (not auto-proven in repo state)

- CUDA single-GPU soak evidence.
- 2-GPU distributed correctness evidence.
- 4-GPU soak reliability evidence.
- GPU evidence verification script run against collected logs.

## Final readiness verdict

- `v1.0.0` through `v2.0.0` are documented and implementation-backed in this repository.
- Current state is production-grade for CPU/non-GPU and self-host replacement-readiness gates.
- Final all-target hardware production sign-off remains blocked only by operator GPU evidence.
