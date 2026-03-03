# Enkai Bootstrap Core (v1.6)

## Purpose

`v1.6.0` introduces bootstrap-core primitives for a deterministic Stage0/Stage1 path.
The objective is to validate that Enkai-scripted compiler orchestration produces the
same bytecode as direct Rust Stage0 compilation for the supported subset.

## Commands

- `enkai litec check <input.enk>`
- `enkai litec compile <input.enk> --out <program.bin>`
- `enkai litec verify <input.enk>`

`litec verify` performs:

1. Stage0 compile (Rust parser/checker/compiler).
2. Stage1 compile through Enkai-scripted `enkai_lite.enk`.
3. Bytecode equality check on serialized `Program` output.

## Runtime Surface

Bootstrap-core relies on the `compiler` runtime module:

- `compiler.parse_subset(source)` -> summary record.
- `compiler.check_subset(source)` -> validates subset + typecheck.
- `compiler.emit_subset(source, output_path)` -> emits bytecode file.

These APIs are policy-gated for filesystem writes where applicable.

## Subset Contract

The bootstrap-core subset intentionally allows:

- top-level `import`, `policy`, `fn`, and top-level statements
- statements: `let`, assignment, expression, `if`, `while`, `return`
- expressions: literals, identifiers, unary/binary, call, index, field, list, postfix `?`

The subset intentionally rejects:

- declarations: `native::import`, `use`, `type`, `enum`, `impl`, `tool`, `prompt`, `model`, `agent`
- statements: `for`, `match`, `try/catch`, `break`, `continue`
- expressions: `lambda`, `match` expression

## Quality Gates

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- bootstrap-core tests for `litec compile`, `litec verify`, and subset-rejection behavior.
