# Self-Host Workflow (v2.8.1)

## Purpose

This document defines the day-to-day self-host workflow for bootstrap work and the
required fallback path when a self-host lane fails.

## Daily Workflow

1. Validate subset inputs:
   - `enkai litec check <file.enk>`
2. Validate Stage0/Stage1 bytecode parity:
   - `enkai litec verify <file.enk>`
3. Run staged compile path from Enkai script frontend (mainline default with emergency fallback):
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
7. Run mainline CI-equivalent path with deterministic triage output:
   - `enkai litec mainline-ci <corpus_dir> --triage-dir artifacts/selfhost`

Repository baseline corpus:
- `enkai/tools/bootstrap/selfhost_corpus`

## CI Expectation

Self-host CI is split into two required lanes:
- Mainline lane (Enkai-built compiler default):
  - `enkai litec mainline-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost`
- Stage0 fallback lane (mandatory safety path):
  - bootstrap self-host tests (`bootstrap::tests::litec_selfhost_...`)
  - `enkai litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus`
- plus `master_pipeline_cpu_smoke` in workspace tests.

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
   - triage JSON artifacts from `--triage-dir` when available:
     - `litec_selfhost_ci_report.json`
     - `litec_replace_check_report.json`
     - `litec_mainline_ci_report.json`
     - `litec_mainline_fallback_report.json` (automatic fallback bundle)
   - automatic fallback bundle location:
     - default: `artifacts/selfhost/litec_mainline_fallback_report.json`
     - override: `ENKAI_LITEC_TRIAGE_DIR=<dir>`
4. Land a targeted fix with:
   - regression test in `enkai/src/bootstrap.rs` or compiler/runtime tests,
   - docs update if behavior changed.

Self-host failures do not permit replacing Stage0 for release builds.






