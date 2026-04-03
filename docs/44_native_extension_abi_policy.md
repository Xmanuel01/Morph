# 44. Native Extension ABI Policy

This document defines the native extension ABI policy for Enkai.

## Contract

Native extensions must use:

- stable C ABI exports only
- explicit symbol names
- deterministic load failure behavior

The runtime-supported ABI metadata hooks are:

- `enkai_abi_version`
- `enkai_symbol_table`
- `enkai_handle_free`

## Versioning Rules

- ABI additions are additive only within a release line
- ABI removals or signature changes require a major release
- extension packages must declare their compatible Enkai runtime range

## Handle and Buffer Rules

- opaque handles must be destroyed via `enkai_handle_free`
- extensions must not require the VM to infer ownership
- buffer ownership and free behavior must be explicit and documented per export
- null returns, oversized returns, and ABI mismatch conditions must map to stable runtime errors

## Operational Rules

- native acceleration is optional unless a release gate explicitly requires it
- deterministic VM fallback must remain valid where the public Enkai interface promises fallback
- production rollouts must verify:
  - library load
  - symbol-table compatibility
  - handle free behavior
  - smoke evidence for the accelerated path

## Packaging Rules

- native libraries must be archived with the release package
- strict evidence bundles must capture the exercised accelerated path where required
- operator rollouts must pin native extension versions together with Enkai release versions

