# Changelog

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
