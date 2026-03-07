# Enkai Bootstrap Subset (v1.9.8)

## Purpose

This document defines the stable subset used for bootstrap-lite tooling in v1.9.8.
The goal is deterministic tool output and safe migration toward deeper self-hosting.

## Scope

The subset is intentionally constrained to language/runtime features already proven in CI:

- Function declarations and calls.
- `let` bindings, assignment (`:=`), `if`/`else`, `while`, `return`.
- Record literals via `json.parse("{}")` and record field assignment.
- Lists and indexing where explicitly supported.
- Built-in runtime modules used by bootstrap-lite:
  - `bootstrap` (`format`, `check`, `lint`, `lint_count`, `lint_json`)
  - `tokenizer` (`train`, `load`, `save`, `encode`, `decode`)
  - `dataset` (`open`, `next_batch`)
  - `json`
- Std modules used by bootstrap scripts:
  - `std::env`
  - `std::io`

## Out of Subset

The following are intentionally excluded from bootstrap-lite scripts in v1.9.8:

- `for` loops.
- `try/catch`.
- `break` / `continue`.
- New syntax additions.

These remain outside the bootstrap subset until compiler support is production-grade and parity-gated.

## Tooling Contract

Bootstrap-lite commands:

- `enkai fmt-lite [--check] <file|dir>`
- `enkai lint-lite [--deny-warn] <file|dir>`
- `enkai tokenizer-lite train <dataset_path> <tokenizer_path> [flags]`
- `enkai dataset-lite inspect <dataset_path> <tokenizer_path> --seq-len <n> --batch-size <n> [flags]`

Bootstrap-core stage commands are documented in `docs/bootstrap_core.md`.

Determinism requirement:

- For the same inputs/config, Enkai-scripted bootstrap-lite tools must produce output parity with Rust baseline paths.
- CI parity lane must stay green before release tags are cut.

## Change Control

- Any expansion of this subset must land with:
  - implementation,
  - deterministic tests,
  - updated spec/docs.
- Backward compatibility for existing bootstrap-lite command flags and output schema is required unless a migration path is shipped.



