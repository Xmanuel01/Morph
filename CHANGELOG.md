# Changelog

## Unreleased

### Highlights
- None yet.

### Fixes
- None yet.

### Breaking changes
- None yet.

## v1.7.0

### Highlights
- Added bootstrap self-host beta command:
  - `enkai litec selfhost <corpus_dir>`
- Added staged bootstrap frontend command:
  - `enkai litec stage <parse|check|codegen> <input.enk> [--out <program.bin>]`
- Added self-host CI command:
  - `enkai litec selfhost-ci <corpus_dir> [--no-compare-stage0]`
- Expanded bootstrap-core subset validation for Stage1 corpus support:
  - allows `use`, `type`, `enum`, `impl` declarations
  - allows non-capturing lambda expressions
- Refactored `enkai_lite.enk` into explicit parse/check/codegen phase functions, so stage orchestration for selected compiler components is executed in Enkai.
- Added shared Stage0/Stage1 bytecode equivalence helper used by `litec verify`, `litec selfhost`, and `litec selfhost-ci`.

### Fixes
- Added self-host beta tests covering expanded subset acceptance and corpus verification behavior.
- Added CI `selfhost-beta` lane to run bootstrap self-host regression tests and execute `litec selfhost-ci` against repository corpus.

### Breaking changes
- None.

## v1.6.0

### Highlights
- Added bootstrap-core CLI commands:
  - `enkai litec check <input.enk>`
  - `enkai litec compile <input.enk> --out <program.bin>`
  - `enkai litec verify <input.enk>`
- Added Enkai-scripted bootstrap-core driver (`enkai/tools/bootstrap/enkai_lite.enk`) for Stage1 orchestration.
- Added runtime/compiler `compiler` module:
  - `parse_subset`
  - `check_subset`
  - `emit_subset`
- Added subset validation in runtime for bootstrap-core compilation flow.
- Added bootstrap-core docs: `docs/bootstrap_core.md`.

### Fixes
- Added stage0/stage1 bytecode equivalence verification path (`enkai litec verify`) and tests.
- Hardened policy checks for compiler emission output paths.
- Standardized CI bootstrap parity lane by building `enkai_native` before parity tests.

### Breaking changes
- None.

## v1.5.0

### Highlights
- Added bootstrap-lite CLI commands:
  - `enkai fmt-lite [--check] <file|dir>`
  - `enkai lint-lite [--deny-warn] <file|dir>`
  - `enkai tokenizer-lite train ...`
  - `enkai dataset-lite inspect ...`
- Added Enkai-scripted bootstrap tooling implementations embedded into the CLI and executed through the VM.
- Added runtime/compiler support for bootstrap tooling primitives:
  - new `bootstrap` runtime module (`format`, `check`, `lint`, `lint_count`, `lint_json`)
  - compiler + checker built-in registration for `bootstrap`
  - expanded tokenizer/dataset callable surface for script-based utilities.
- Added bootstrap subset specification: `docs/bootstrap_subset.md`.

### Fixes
- Added deterministic parity tests comparing Enkai bootstrap-lite execution paths with Rust baselines for formatter, linter, tokenizer, and dataset inspection flows.
- Added CI parity lane for bootstrap-lite (`bootstrap::tests`).
- Improved tokenizer capability handling for config-record input in policy context extraction.

### Breaking changes
- None.

## v1.4.0

### Highlights
- Added frontend stack scaffolding in CLI:
  - `enkai new backend`
  - `enkai new frontend-chat`
  - `enkai new fullstack-chat`
- Added typed SDK generator command:
  - `enkai sdk generate <output_file> [--api-version <v>]`
- Shipped React/TypeScript reference chat UI scaffold with streaming response handling, auth token input, and error UX conventions.
- Standardized frontend/backend version pinning contract:
  - route prefix `/api/<version>`
  - request header `x-enkai-api-version`.

### Fixes
- Added contract tests to validate scaffolded backend route layout and generated SDK endpoint/header behavior.
- Added end-to-end fullstack contract test coverage for generated backend/frontend scaffolds, including streaming event parsing and version mismatch behavior.
- Added persisted conversation flow in backend scaffold with conversation ID continuity across stream/chat APIs and persisted state snapshots.
- Hardened CLI argument validation for new SDK/scaffold commands with deterministic error messages.

### Breaking changes
- None.

## v1.3.0

### Highlights
- Added serving-oriented runtime APIs: `http.serve_with`, `http.route`, `http.middleware`, `http.request`, request helpers, and streaming helpers.
- Added HTTP middleware surface for auth, rate limiting, JSONL logging, and default route handling.
- Added structured HTTP error payloads and runtime response metadata headers (`x-enkai-request-id`, `x-enkai-latency-ms`, tenant/error-code where applicable).
- Added CLI `enkai serve` model-selection flow with registry/version pinning or direct checkpoint selection.
- Added filesystem-based model registry helpers via `std::model_registry`.
- Added `std::db` (SQLite) and `std::tls` helper modules backed by `enkai_native`.

### Fixes
- Normalized rate-limit IP keys to avoid per-connection bypass when client source ports change.
- Stabilized HTTP integration tests by serializing test server lifecycles to prevent port-binding races.

### Breaking changes
- None.

## v1.2.0

### Highlights
- Added multi-GPU runtime wiring (world size/rank config), grad accumulation, grad clipping, and AMP config support.
- Implemented ranked checkpoints with manifest metadata for distributed runs.
- Added dataset prefetch and packing efficiency metrics.
- Added build cache + deterministic `enkai.lock` resolver via `enkai build`.
- Expanded stdlib with `std::env`, `std::path`, `std::time`, `std::log`, `std::io`, `std::process`.
- Hardened policy/capability checks for new IO and networking surfaces.

### Fixes
- Improved checkpoint metadata compatibility in ranked loads.
- Deterministic dataset sharding across ranks.

### Breaking changes
- None.

## v0.9.3

### Highlights
- Synced release notes with the `docs/Enkai.spec` v0.1 -> v0.9.3 compatibility framing.
- Clarified v0.9.3 runtime status for tensor training hooks:
  - native loss path currently integrated as `forward_tinylm`,
  - checkpoint C ABI hooks are present and backend-loadable,
  - distributed hooks are present but environment-gated by CUDA/NCCL/runtime setup.
- Aligned release/version references to the current production line (`v0.9.3`) in top-level docs.
- Added `enkai --version` output with language version.

### Fixes
- Reduced spec/changelog drift by aligning guaranteed behavior vs known limits.

### Breaking changes
- None.

## v0.9.2

### Highlights
- Aligned release/version references to the current production line (`v0.9.2`) in top-level docs.
- Reframed `docs/Enkai.spec` as a compatibility baseline (`v0.1 -> v0.9.2`) to reduce spec/version drift.
- Added `docs/README.md` as a docs index with versioning policy and recommended reading order.
- Added roadmap guidance that historical milestones are context, while `docs/Enkai.spec` is the current language source of truth.

### Fixes
- Clarified legacy milestone labels in `docs/Enkai.spec` (features introduced in earlier versions) as baseline behavior in `v0.9.2`.

## v0.7.1

### Fixes
- Bumped workspace crate versions to 0.7.1 for release builds.

## v0.7.0

### Highlights
- Added cooperative tasks, channels, and a TCP networking layer for concurrent workloads.
- Shipped HTTP server/client helpers and JSON parse/stringify (record/list mappings).
- Added install scripts plus automated release packaging (zip/tar.gz + checksums).
- Added Windows installer builds (Inno Setup + WiX) with signing/notarization hooks.
- Expanded docs and learn lessons for concurrency, networking, HTTP, and JSON.

### Fixes
- Stabilized scheduler + IO blocking behavior with VM tests for tasks, channels, net, HTTP, and JSON.

## v0.5.0

### Highlights
- Added multi-file module imports with caching and circular import detection.
- Enforced public/private visibility across modules with compile + typecheck validation.
- Introduced structured diagnostics (source snippets) and runtime stack traces.
- Shipped deterministic `enkai fmt` and a CLI `enkai test` runner.
- Added v0.5 docs and learn lessons for modules, visibility, formatting, and testing.

### Fixes
- Added regression coverage for formatter/test CLI paths and private access span tracking.

## v0.4.0

### Highlights
- Introduced the bytecode + VM execution pipeline with chunk constants, stack safety, globals, and CLI flags (`--trace-vm`/`--disasm`).
- Added deep scope-aware symbol allocation, short-circuit logic ops, and Optional[T] coverage in the new type checker.
- Published learn/docs axes (learn lessons + docs/dev_debugging) and stabilized the CLI to always run `enkai check` before execution.

### Fixes
- Stabilized parser/compiler/VM tests, cleaned warnings, and ensured `cargo fmt`, `cargo clippy -D warnings`, and `cargo test` pass in the v0.4 path.

## v0.3.1

### Fixes
- Correct loader diagnostics for private symbol imports and reduce `use` resolution duplication.
- Use-list edge case diagnostics and expanded parser coverage for visibility rules.
- Policy filter matching now uses path components normalization and list-valued tests.
- Added tests for string native error paths and dependency manifest variants.

## v0.3.0

### Features
- Public exports + re-exports via `pub` and `pub use`.
- Labeled-span diagnostics for parse/loader errors.
- Local path dependencies in `enkai.toml`.
- Policy filters for `domain` and `path_prefix`.
- `std.string` helpers and `std.fs` read/write (policy-gated).

### Breaking changes
- Importing private symbols now errors; only `pub` exports are importable.

## v0.2.0

### Features
- Module loader + use resolution with file-based modules.
- `enkai run .` for project roots using `enkai.toml` and `src/main.enk`.
- Parser diagnostics include line/col + snippet.
- Runtime stack traces for errors.
- Minimal formatter with `enkai fmt` and `enkai fmt --check`.
- Policy enforcement MVP with default deny and allow/deny rules.

### Release Notes v0.2.0
- Smoke test:
  - `cargo run -p enkai -- run examples/project_v02`
  - `cargo run -p enkai -- fmt --check examples/project_v02/src/main.enk`

### Breaking changes
- IO and tool calls are now policy-gated; code that called `print` or tools without a policy will fail until a policy allows those capabilities.



