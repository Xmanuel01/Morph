# 34. Model Lifecycle CLI (v2.1 foundation)

Enkai provides local registry lifecycle operations for serving workflows.

## Commands

- `enkai model register <registry_dir> <name> <version> <checkpoint_path> [--activate]`
- `enkai model list <registry_dir> [name] [--json]`
- `enkai model promote <registry_dir> <name> <version>`
- `enkai model retire <registry_dir> <name> <version>`
- `enkai model rollback <registry_dir> <name> <version>`

## Registry Layout

Under `<registry_dir>/<name>/<version>/`:
- `model.meta.json`
- `checkpoint_path.txt`

Under `<registry_dir>/<name>/`:
- `.active_version`

Under `<registry_dir>/`:
- `registry.json`

## Serve Resolution Behavior

When serving via `--registry` + `--model`:
- active pointer (`.active_version`) is used by default
- `--latest` selects highest semver available
- checkpoint pointer (`checkpoint_path.txt`) is used when present

This keeps model selection deterministic for operators.
