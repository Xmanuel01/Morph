# Self-Host Workflow (v1.9.4)

## Purpose

This document defines the day-to-day self-host workflow for bootstrap work and the
required fallback path when a self-host lane fails.

## Daily Workflow

1. Validate subset inputs:
   - `enkai litec check <file.enk>`
2. Validate Stage0/Stage1 bytecode parity:
   - `enkai litec verify <file.enk>`
3. Run staged compile path from Enkai script frontend:
   - `enkai litec stage parse <file.enk>`
   - `enkai litec stage check <file.enk>`
   - `enkai litec stage codegen <file.enk> --out <program.bin>`
4. Run stage1-compiled program directly:
   - `enkai litec run <file.enk>`
5. Run corpus-level self-host checks:
   - `enkai litec selfhost <corpus_dir>`
   - `enkai litec selfhost-ci <corpus_dir>`
6. Run replacement-readiness fixed-point gate:
   - `enkai litec replace-check <corpus_dir>`

Repository baseline corpus:
- `enkai/tools/bootstrap/selfhost_corpus`

## CI Expectation

The self-host lane is expected to run:
- bootstrap self-host tests (`bootstrap::tests::litec_selfhost_...`)
- `enkai litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus`
- `master_pipeline_cpu_smoke`

## Fallback Path

If any self-host command fails:

1. Treat Rust Stage0 as the release compiler path.
2. Keep release blocking checks on:
   - `cargo fmt --all --check`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo test --workspace`
3. Capture failure artifacts:
   - failing `.enk` file path
   - stage mode (`parse/check/codegen/verify/selfhost-ci`)
   - compiler/type/runtime error text
4. Land a targeted fix with:
   - regression test in `enkai/src/bootstrap.rs` or compiler/runtime tests,
   - docs update if behavior changed.

Self-host failures do not permit replacing Stage0 for release builds.
