# Enkai v1.9.8 RC Notes (v2.0.0 Stabilization Freeze)

## Scope

`v1.9.8` is the release-candidate freeze before `v2.0.0`.

- No new language syntax.
- No bootstrap feature expansion.
- Reliability, migration, packaging, and compatibility hardening only.

## RC Freeze Policy

- Grammar remains frozen to the established v0.9.3 baseline.
- VM runtime remains the production contract.
- `v2.0.0` will enforce strict compatibility for config/checkpoint contracts.
- All RC sign-off requires archived operator GPU evidence.

## RC Gates

- CPU gates:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
- Release + packaging gates:
  - `powershell -ExecutionPolicy Bypass -File scripts/release_pipeline.ps1`
  - or `sh scripts/release_pipeline.sh`
- RC gate (GPU evidence required by default):
  - `powershell -ExecutionPolicy Bypass -File scripts/rc_pipeline.ps1`
  - or `sh scripts/rc_pipeline.sh`
- Evidence archive:
  - `scripts/collect_release_evidence.py` writes `artifacts/release/v<version>/manifest.json`.

## v2.0.0 Breaking-By-Policy Preview

`v2.0.0` will reject legacy artifacts that were tolerated in v1.9.x:

- Train/eval configs missing `config_version`.
- Checkpoints missing required v1 metadata fields unless migrated.

Use migration tooling before v2.0.0 cut:

- `enkai migrate config-v1 <in> <out>`
- `enkai migrate checkpoint-meta-v1 <checkpoint_dir>`
- `enkai doctor [path]`

Rollback path during RC:

- Keep running `v1.9.8` binaries.
- Keep a backup copy of pre-migration config/checkpoint trees before migration.
