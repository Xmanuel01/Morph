# Changelog

## Unreleased

### Highlights
- None yet.

### Fixes
- None yet.

### Breaking changes
- None yet.

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



