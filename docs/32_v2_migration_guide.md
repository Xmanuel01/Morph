# Enkai v2.0.0 Migration Guide (From v1.9.8 RC)

## Why migration is required

`v2.0.0` enforces strict compatibility for training config and checkpoint metadata.
Legacy implicit acceptance paths from v1.9.x are removed from the runtime acceptance path.

## Pre-migration checklist

1. Backup your project configuration and checkpoint directories.
2. Ensure you are on `v1.9.8`.
3. Run baseline validation:
   - `cargo test --workspace`
   - `powershell -ExecutionPolicy Bypass -File scripts/release_pipeline.ps1`
   - or `sh scripts/release_pipeline.sh`

## Step 1: Migrate train/eval configs

Run:

```bash
enkai migrate config-v1 <in_config.enk> <out_config.enk>
```

Expected result:

- `config_version: 1` is present.
- required fields are explicit.
- defaults are normalized.

## Step 2: Migrate checkpoint metadata

Run:

```bash
enkai migrate checkpoint-meta-v1 <checkpoint_dir>
```

Optional verification:

```bash
enkai migrate checkpoint-meta-v1 <checkpoint_dir> --verify
```

Expected result:

- `meta.json` includes required v1 keys (`format_version`, `config_hash`, `model_sig`, `dtype`, `device`, etc.).
- legacy omissions are filled or flagged.

## Step 3: Run readiness scanner

Run:

```bash
enkai doctor [path]
```

Expected result:

- no v2.0 blockers for config/checkpoint contracts.
- explicit report for anything still non-compliant.

## Step 4: Validate migrated assets

1. Run train/eval smoke on migrated config.
2. Resume from migrated checkpoints.
3. Re-run release/RC pipelines as required.

## Rollback strategy

- Keep backups from pre-migration state.
- If migration output is not acceptable:
  - restore backups,
  - re-run migration after fixing blockers reported by `enkai doctor`,
  - validate again before retrying the v2.0 path.
