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

- `strict_selfhost` readiness originally froze the dependency inventory and the blocker model.
- `v3.3.0` shipped-surface strict-selfhost line is complete.
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
- For the shipped strict-selfhost surface, there are currently no unresolved
  blocking subsystems.
- The broader `v3.1.0 -> v4.0.0` zero-Rust target remains open beyond the
  shipped-surface closure proof and can still add future replacement work.

## Historical Blocking Subsystems (closed in `v3.3.0`)

The shipped-surface blockers that were frozen and then closed in `v3.3.0` were:

- compiler frontend
- runtime core
- systems / CLI orchestration
- native std / acceleration layer
- tensor backend
- data / registry layer

These no longer block the shipped strict-selfhost line. They remain useful as
historical decomposition for broader future zero-Rust work outside the already
closed `v3.3.0` shipped surface.

## Post-Closure Next-Step Baseline (`v3.4.0`)

After the `v3.3.0` shipped-surface closure, the next zero-Rust work is broader
than the strict-selfhost shipped release gate. The current `v3.4.0` baseline
tracks these follow-on categories explicitly:

- cross-host install-flow proof beyond the current Windows-host execution evidence
- compatibility-only storage/data paths that remain tolerated outside shipped
  strict-selfhost blockers
- globally accelerated native/tensor backend replacement beyond the current
  runtime-owned fallback boundary
- broader zero-Rust closure of historical non-shipped compatibility paths on
  the way to the eventual `v4.0.0` target

These are roadmap categories for the broader zero-Rust program. They do not
reopen the already closed `v3.3.0` strict-selfhost shipped surface.

## Compatibility-Only Storage/Data Baseline (`v3.4.0`)

The first concrete storage/data post-closure tranche makes the currently
tolerated compatibility path explicit instead of leaving it implied:

- `sqlite_binding` is the compatibility-only storage/data path still tracked
  outside shipped strict-selfhost release blockers
- the shipped strict-selfhost data/registry surface remains complete
- broader future work to replace SQLite-backed compatibility paths globally
  remains roadmap work on the way to `v4.0.0`

The source-of-record proof boundary for this tranche is:

- `artifacts/readiness/strict_selfhost_dependency_inventory.json`
- `artifacts/readiness/strict_selfhost_data_registry_protocols_surface.json`
- `artifacts/readiness/v3_4_0_compatibility_storage_data_baseline.json`

## Accelerated Native/Tensor Backend Baseline (`v3.4.0`)

The next concrete post-closure tranche makes the accelerated backend boundary
explicit instead of leaving it implied:

- the shipped strict-selfhost tensor and native std/accel surfaces remain complete
- runtime-owned fallback and shipped proof boundaries are closed for the shipped
  line
- broader global replacement of accelerated Rust/native tensor backends remains
  roadmap work on the way to `v4.0.0`

The source-of-record proof boundary for this tranche is:

- `artifacts/readiness/strict_selfhost_tensor_backend_surface.json`
- `artifacts/readiness/strict_selfhost_native_std_and_accel_surface.json`
- `artifacts/readiness/strict_selfhost_dependency_inventory.json`
- `artifacts/readiness/v3_4_0_accelerated_native_tensor_baseline.json`

## Historical Non-Shipped Compatibility Closure Baseline (`v3.4.0`)

The final post-closure baseline makes the broader non-shipped compatibility
boundary explicit:

- the shipped strict-selfhost release surface remains complete
- broader historical compatibility paths outside the shipped release gate are
  explicitly tracked as future roadmap work toward `v4.0.0`
- these paths no longer exist as an implicit unresolved category inside the
  `v3.4.0` line

The source-of-record proof boundary for this tranche is:

- `artifacts/readiness/v3_4_0_zero_rust_next_step_baseline.json`
- `artifacts/readiness/v3_4_0_install_host_matrix_baseline.json`
- `artifacts/readiness/v3_4_0_compatibility_storage_data_baseline.json`
- `artifacts/readiness/v3_4_0_accelerated_native_tensor_baseline.json`
- `artifacts/readiness/v3_4_0_non_shipped_compatibility_closure_baseline.json`
