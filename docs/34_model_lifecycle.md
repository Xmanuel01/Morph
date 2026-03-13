# 34. Model Lifecycle CLI (v2.5.3)

Enkai provides local registry lifecycle operations for serving workflows plus
additive remote-registry sync commands with immutable artifact metadata and
optional signature verification.

## Commands

- `enkai model register <registry_dir> <name> <version> <checkpoint_path> [--activate]`
- `enkai model list <registry_dir> [name] [--json]`
- `enkai model load <registry_dir> <name> <version>`
- `enkai model unload <registry_dir> <name> <version>`
- `enkai model loaded <registry_dir> [name] [--json]`
- `enkai model promote <registry_dir> <name> <version>`
- `enkai model retire <registry_dir> <name> <version>`
- `enkai model rollback <registry_dir> <name> <version>`
- `enkai model push <registry_dir> <name> <version> --registry <remote_registry_dir> [--sign]`
- `enkai model pull <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
- `enkai model promote-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
- `enkai model retire-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
- `enkai model rollback-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`

## Registry Layout

Under `<registry_dir>/<name>/<version>/`:
- `model.meta.json`
- `checkpoint_path.txt`
- `remote.manifest.json` (present after remote pull/push sync)
- `remote.manifest.sig` (present when signed)

Under `<registry_dir>/<name>/`:
- `.active_version`

Under `<registry_dir>/`:
- `registry.json`
- `.serve_state.json` (explicit load/unload lifecycle state for serving)
- `audit.log.jsonl` (append-only lifecycle and remote-sync audit events)

## Remote Sync Contract

- Remote registry is filesystem-backed and reuses the same registry layout.
- `push` writes immutable `remote.manifest.json` metadata with `artifact_digest`.
- Current immutable digest scope is the model identity plus `checkpoint_path.txt` hash.
- If a remote version already exists with a different digest, `push` is rejected.
- `--sign` writes `remote.manifest.sig` using `ENKAI_MODEL_SIGNING_KEY`.
- `--verify-signature` validates `remote.manifest.sig` before `pull`/remote state sync.
- `--fallback-local` keeps pinned local version behavior when remote is unavailable or signature validation fails.

Signature key:
- `ENKAI_MODEL_SIGNING_KEY` is required for signature create/verify operations.

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
