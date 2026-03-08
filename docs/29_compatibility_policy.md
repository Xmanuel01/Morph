# Compatibility And Deprecation Policy (v2.0.0)

## Scope

This policy defines compatibility guarantees for:
- language grammar/CLI surface,
- train/eval config schema,
- checkpoint metadata format,
- generated frontend/backend API contract.

## Compatibility Contract

- Grammar remains frozen at the v0.9.3 baseline for v2.x.
- `config_version: 1` is the stable train/eval schema for v2.x.
- Checkpoints with `format_version: 1` are the stable checkpoint format for v2.x.
- v2.0.0 strict acceptance path:
  - `enkai train` / `enkai eval` reject configs missing `config_version`.
  - v2.0.0 strict checkpoint loads reject `meta.json` missing `format_version` or required v1 metadata fields.
- Temporary legacy recovery remains explicitly gated:
  - `enkai train|eval --lenient-contracts` only when `ENKAI_ALLOW_LEGACY_CONTRACTS=1`.
  - `enkai doctor --lenient` for transition audits.
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
- Legacy config/checkpoint compatibility is no longer implicit in v2.0.0 runtime acceptance paths.
- Migration tooling remains available for controlled recovery and upgrade.

## Required Compatibility Tests

Release candidates must pass:
- `strict_contracts_require_config_version`
- `strict_contracts_reject_legacy_checkpoint_meta`
- frontend API contract tests in `enkai/src/frontend.rs`
- self-host compatibility checks:
  - `enkai litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus`
- master release smoke:
  - `master_pipeline_cpu_smoke`

## Migration Guidance

Tooling support:
- `enkai migrate config-v1 <in> <out>` converts/ejects canonical v1 train/eval config.
- `enkai migrate checkpoint-meta-v1 <checkpoint_dir> [--dry-run] [--verify] [--strict-contracts]` upgrades or verifies checkpoint metadata.
- `enkai doctor [path] [--json] [--strict-contracts|--lenient]` reports v2.0 readiness blockers for config/checkpoint contracts.

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

When resaving from v2.0.0 train/eval flows, metadata is written in v1 format.



