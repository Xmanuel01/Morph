# 34. Model Lifecycle CLI (v2.1 foundation)

Enkai provides local registry lifecycle operations for serving workflows.

## Commands

- `enkai model register <registry_dir> <name> <version> <checkpoint_path> [--activate]`
- `enkai model list <registry_dir> [name] [--json]`
- `enkai model load <registry_dir> <name> <version>`
- `enkai model unload <registry_dir> <name> <version>`
- `enkai model loaded <registry_dir> [name] [--json]`
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
- `.serve_state.json` (explicit load/unload lifecycle state for serving)

## Serve Resolution Behavior

When serving via `--registry` + `--model`:
- active pointer (`.active_version`) is used by default
- `--latest` selects highest semver available
- checkpoint pointer (`checkpoint_path.txt`) is used when present
- `--require-loaded` enforces that selected version exists in `.serve_state.json`

When serving via `--multi-model --registry <dir>`:
- request must include `x-enkai-model-name` and `x-enkai-model-version`
- runtime verifies requested tuple is loaded in `.serve_state.json`
- deterministic failures:
  - missing selector headers -> `400 missing_model_selector`
  - selector not loaded -> `409 model_not_loaded`

This keeps model selection deterministic for operators.
