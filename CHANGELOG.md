# Changelog

## v0.3.0

### Features
- Public exports + re-exports via `pub` and `pub use`.
- Labeled-span diagnostics for parse/loader errors.
- Local path dependencies in `morph.toml`.
- Policy filters for `domain` and `path_prefix`.
- `std.string` helpers and `std.fs` read/write (policy-gated).

### Breaking changes
- Importing private symbols now errors; only `pub` exports are importable.

## v0.2.0

### Features
- Module loader + use resolution with file-based modules.
- `morph run .` for project roots using `morph.toml` and `src/main.morph`.
- Parser diagnostics include line/col + snippet.
- Runtime stack traces for errors.
- Minimal formatter with `morph fmt` and `morph fmt --check`.
- Policy enforcement MVP with default deny and allow/deny rules.

### Release Notes v0.2.0
- Smoke test:
  - `cargo run -p morph -- run examples/project_v02`
  - `cargo run -p morph -- fmt --check examples/project_v02/src/main.morph`

### Breaking changes
- IO and tool calls are now policy-gated; code that called `print` or tools without a policy will fail until a policy allows those capabilities.
