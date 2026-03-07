# Compatibility And Deprecation Policy (v1.9.5)

## Scope

This policy defines compatibility guarantees for:
- language grammar/CLI surface,
- train/eval config schema,
- checkpoint metadata format,
- generated frontend/backend API contract.

## Compatibility Contract

- Grammar remains frozen at the v0.9.3 baseline for v1.x.
- `config_version: 1` is the stable train/eval schema for v1.x.
- Checkpoints with `format_version: 1` are the stable checkpoint format for v1.x.
- Checkpoints missing `format_version` are treated as legacy v0 metadata and remain loadable in v1.9.5.
- Generated frontend SDK/backend route contract must preserve:
  - `/api/<version>` prefix
  - `x-enkai-api-version` request header.
- Multi-rank distributed runtime must remain explicitly opt-in through `ENKAI_ENABLE_DIST=1`.

## Deprecation Rules

- Deprecations are implementation-first and must ship with tests and docs in the same change.
- Deprecated behavior must remain available for at least one minor release unless a security issue requires immediate removal.
- Any upcoming removal must include:
  - warning text in CLI/runtime path,
  - migration guidance,
  - target removal version.

Current deprecation notice:
- Legacy train/eval configs without `config_version` emit a warning.
- Planned removal target for legacy config parsing: `v2.0`.

## Required Compatibility Tests

Release candidates must pass:
- `legacy_config_without_config_version_still_trains`
- `legacy_checkpoint_meta_without_format_version_loads`
- frontend API contract tests in `enkai/src/frontend.rs`
- self-host compatibility checks:
  - `enkai litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus`
- master release smoke:
  - `master_pipeline_cpu_smoke`

## Migration Guidance

Tooling support:
- `enkai migrate config-v1 <in> <out>` converts/ejects canonical v1 train/eval config.
- `enkai migrate checkpoint-meta-v1 <checkpoint_dir> [--dry-run] [--verify]` upgrades or verifies checkpoint metadata.
- `enkai doctor [path]` reports v2.0 readiness blockers for config/checkpoint contracts.

### Legacy config -> v1 config

Minimum migration:
- add `config_version: 1`
- add explicit `backend`
- keep tokenizer declaration via one of:
  - `tokenizer_path`
  - `tokenizer_train`.

### Legacy checkpoint metadata -> v1 metadata

Recommended metadata fields in `meta.json`:
- `format_version`
- `step`
- `tokens`
- `loss`
- `config_hash`
- `model_sig`
- `dtype`
- `device`.

When resaving from v1.9.5 train/eval flows, metadata is written in v1 format.

