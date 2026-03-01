# Enkai Docs

## Versioning policy

- Current production release line: `v1.3.0`.
- `docs/Enkai.spec` is the normative language reference for `v1.3.0`.
- Known limits and partial features are tracked inside `docs/Enkai.spec` under `Known Limits in v1.3.0`.
- Spec updates are implementation-first (update docs after code + tests land).
- Historical implementation snapshots remain in `docs/v0.1-status.md`, `docs/v0.2-status.md`, and `docs/v0.3-plan.md`.

## Reading order

1. `docs/Enkai.spec` (language reference)
2. `docs/roadmap.md` (delivery direction)
3. `docs/07_bytecode_vm.md`, `docs/09_modules.md`, `docs/13_ffi.md` (core implementation topics)
4. `docs/18_http_server.md`, `docs/26_serve_cli.md`, `docs/tensor_api.md`, `docs/gpu_backend.md`, `docs/checkpoints_sharded.md` (serving + tensor/training stack)
