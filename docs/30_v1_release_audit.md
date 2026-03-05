# Enkai v1.0.0-v1.9.0 Delivery And Production-Grade Audit

## Purpose

This document records what shipped from `v1.0.0` through `v1.9.0` and the current production-readiness status.

## Source of truth used for this record

- Git tags: `v1.0.0` through `v1.9.0`.
- `CHANGELOG.md`.
- `docs/Enkai.spec` compatibility and runtime sections.
- `VALIDATION.md`.
- `docs/RELEASE_CHECKLIST.md`.
- Release pipeline run on `2026-03-04 10:01:53 +03:00`:
  - `powershell -ExecutionPolicy Bypass -File scripts/v1_9_release_pipeline.ps1`

## Release ledger (v1.0.0-v1.9.0)

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

## Production-grade audit status

### Automated gates (executed)

Run executed on `2026-03-04`:

- `cargo fmt --all --check` -> PASS
- `cargo clippy --workspace --all-targets -- -D warnings` -> PASS
- `cargo test --workspace` -> PASS
- `cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus` -> PASS
- `cargo run -p enkai -- litec replace-check enkai/tools/bootstrap/selfhost_corpus --no-compare-stage0` -> PASS
- Consolidated pipeline script `scripts/v1_9_release_pipeline.ps1` -> PASS

### Coverage conclusion

- Language/runtime/core CLI: production-grade under automated CPU gates.
- Backend/serve/frontend scaffolds: production-grade under automated integration tests.
- Bootstrap-lite/core/self-host CI lanes: production-grade under current deterministic parity/fixed-point gates.
- Compatibility controls (legacy config/checkpoint): production-grade under automated migration tests.

### Remaining operator-required evidence (not auto-proven in repo state)

- CUDA single-GPU soak evidence.
- 2-GPU distributed correctness evidence.
- 4-GPU soak reliability evidence.
- Optional GPU evidence verification script run against collected logs.

## Final readiness verdict

- `v1.0.0` through `v1.9.0` are fully documented and implementation-backed in this repository.
- Current state is production-grade for CPU/non-GPU and self-host replacement-readiness gates.
- Final "all-target hardware production-grade" sign-off remains blocked only by operator GPU soak evidence listed above.
