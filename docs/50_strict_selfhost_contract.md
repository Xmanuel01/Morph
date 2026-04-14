# 50. Strict Self-Host Contract

This document freezes the `v3.1.0 -> v4.0.0` zero-Rust self-hosting target.

## Goal

The strict self-host line means the shipped Enkai platform must not require:

- Rust binaries
- Rust crates in the shipped toolchain/runtime path
- Rust-owned native modules as required production dependencies

This contract is stronger than the current `v3.0.0` release state. `v3.0.0`
remains CPU-complete and GPU-pending, but it is still a Rust workspace and is
not yet a zero-Rust shipped platform.

## Source of Truth

Machine-readable contract inputs live in:

- `enkai/contracts/readiness_strict_selfhost_v3_1_0.json`
- `enkai/contracts/strict_selfhost_release_blockers_v3_1_0.json`
- `enkai/contracts/strict_selfhost_dependency_board_v3_1_0.json`
- `enkai/contracts/selfhost_frontier_v3_1_1.json`
- `enkai/contracts/selfhost_examples_v3_1_1.json`
- `enkai/contracts/selfhost_bootstrap_v3_1_1.json`
- `enkai/contracts/selfhost_negative_v3_1_1.json`
- `enkai/contracts/selfhost_audited_surface_v3_1_1.json`

Generated inventory output lives in:

- `artifacts/readiness/strict_selfhost_dependency_inventory.json`
- `artifacts/readiness/selfhost_frontier_verify.json`
- `artifacts/readiness/selfhost_examples_verify.json`
- `artifacts/readiness/selfhost_bootstrap_verify.json`
- `artifacts/readiness/selfhost_negative_verify.json`
- `artifacts/readiness/selfhost_audited_surface_verify.json`

## Current Interpretation

- `strict_selfhost` readiness does **not** claim completion.
- It freezes the dependency inventory and the blocker model.
- It requires both the frozen declaration frontier and the shipped `examples/`
  corpus to stay green under the self-host frontend audit.
- It also requires bootstrap compiler sources and a curated negative semantic
  corpus to stay green under the self-host frontend audit contract.
- It requires a curated audited executable surface to pass `frontend-audit`,
  `selfhost-ci`, `replace-check`, and `mainline-ci` proof through one bundled
  verification artifact.
- Validation examples that depend on the repository `examples/` package layout
  remain part of the shipped examples audit, but are not forced through the
  bundled audited-surface materialization.
- Any remaining Rust-owned shipped dependency must stay visible in the release
  dashboard until the `v3.4.0` zero-Rust closure milestone removes it.

## Current Blocking Subsystems

The current frozen blockers are:

- compiler frontend
- runtime core
- systems / CLI orchestration
- native std / acceleration layer
- tensor backend
- data / registry layer

These blockers remain unresolved until their shipped Rust/native dependencies are
replaced or removed from the strict self-host product path.
