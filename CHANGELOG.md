# Changelog

## Unreleased

### Highlights
- Added train/eval config schema v1 with `config_version` and explicit validation.
- Added checkpoint metadata v1 (`format_version`, model signature, dtype, device).
- Updated spec change-control policy to implementation-first.
- Added v1.0 validation checklist updates.

### Fixes
- Improved checkpoint compatibility checks for legacy metadata.

### Breaking changes
- None yet.

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



