# Full Frontend Frontier (v3.1.1)

`v3.1.1` starts the transition from the `litec` bootstrap subset to a
full-language self-hosted frontend by making the current boundary measurable.

## Purpose

The current self-host compiler can prove:

- subset/frontier files that compile and execute through the Enkai-built stage2
  compiler path
- stage0/stage2 bytecode parity for those files
- stage0/stage2 runtime parity for those files
- package-aware import resolution for file-based bootstrap compilation via
  `compiler.*_subset_file(...)`
- self-host subset validation and summary synthesis over structural
  `compiler.describe_subset*_file` output for the current bootstrap frontier
- self-host package/module subset validation over
  `compiler.describe_subset_package_file(...)` before invoking raw package-aware
  typecheck/codegen
- self-host semantic checks for duplicate symbols/import bindings, impl target existence,
  and constructor/local/imported call arity
- self-host codegen acceptance checks over emitted bytecode summaries via
  `compiler.describe_program_file(...)`

It cannot yet claim that the full current language is self-hosted.

## Command

- `enkai litec frontend-audit <corpus_dir> [--triage-dir <dir>] [--require-full-support]`

## Report

When `--triage-dir` is provided, the command writes:

- `litec_frontend_audit_report.json`

The frozen v3.1.1 audit contracts live in:

- `enkai/contracts/selfhost_frontier_v3_1_1.json`
- `enkai/contracts/selfhost_examples_v3_1_1.json`
- `enkai/contracts/selfhost_bootstrap_v3_1_1.json`
- `enkai/contracts/selfhost_negative_v3_1_1.json`

Verification output lives in:

- `artifacts/readiness/selfhost_frontier_verify.json`
- `artifacts/readiness/selfhost_examples_verify.json`
- `artifacts/readiness/selfhost_bootstrap_verify.json`
- `artifacts/readiness/selfhost_negative_verify.json`

The report includes:

- stage1/stage2 bootstrap compiler fingerprints
- stage1/stage2 fixed-point status
- total files scanned
- files accepted by the Rust frontend
- files accepted by the self-host frontend
- bytecode parity count
- runtime parity count
- frontier gap count
- invalid file count
- per-file gap/error details

## Current Status

- the frozen declaration frontier (`native::import`, `tool`, `prompt`, `model`,
  `agent`) now passes through the self-host frontend audit corpus
- the shipped `examples/` corpus now also passes through the self-host frontend
  audit
- bootstrap compiler sources now also pass through the self-host frontend audit
- a curated negative semantic corpus now verifies expected stage0/self-host
  frontend rejections instead of only positive acceptance
- the bootstrap script now owns another concrete frontend slice: subset shape
  validation over structural compiler output, while Rust still owns the parser,
  typechecker, and final bytecode emitter paths
- the bootstrap script now also validates imported package modules on the
  self-host path instead of delegating package subset validation entirely to
  Rust helpers
- the bootstrap script now owns another semantic slice and a post-emission codegen
  acceptance gate before raw emitted bytecode is accepted
- broader non-example corpora can still fail for reasons outside the frozen
  declaration frontier, including unsupported language surfaces not yet
  migrated into the self-host frontend

## Intended Use

- run the audit over broader example and corpus directories
- use the resulting report as the migration board for replacing the `lite`
  subset boundary with full-language self-host support
- only treat `v3.1.1` as complete when the frontier report is no longer used as
  a gap inventory and the self-host frontend is the source of truth
