# Enkai Docs

## Versioning policy

- Current production release line: `v2.6.3`.
- `docs/Enkai.spec` is the normative language reference for `v2.6.3`.
- Known limits and partial features are tracked inside `docs/Enkai.spec` under `Known Limits in v2.6.3`.
- VM bytecode runtime behavior is the production contract; the legacy tree-walk interpreter is non-production/reference only.
- Spec updates are implementation-first (update docs after code + tests land).
- Full-platform readiness gating for the v2.5+ program is tracked by:
  - `enkai/contracts/readiness_full_platform_v2_5_0.json`
  - `enkai/contracts/full_platform_release_blockers_v2_5_0.json`
- Historical implementation snapshots remain in `docs/v0.1-status.md`, `docs/v0.2-status.md`, and `docs/v0.3-plan.md`.

## Reading order

1. `docs/Enkai.spec` (language reference)
2. `docs/roadmap.md`, `docs/30_v1_release_audit.md`, `docs/31_v2_rc_notes.md`, `docs/32_v2_migration_guide.md` (delivery direction + release ledger + RC/migration policy)
3. `docs/07_bytecode_vm.md`, `docs/09_modules.md`, `docs/13_ffi.md` (core implementation topics)
4. `docs/18_http_server.md`, `docs/26_serve_cli.md`, `docs/27_frontend_stack.md`, `docs/bootstrap_subset.md`, `docs/bootstrap_core.md`, `docs/28_selfhost_workflow.md`, `docs/29_compatibility_policy.md`, `docs/tensor_api.md`, `docs/gpu_backend.md`, `docs/checkpoints_sharded.md` (serving + frontend + bootstrap + compatibility + tensor/training stack)
5. `docs/33_benchmark_suite.md`, `docs/34_model_lifecycle.md`, `docs/35_data_algo_stack.md`, `docs/36_capability_complete_report.md`, `docs/37_readiness_matrix.md` (benchmark policy + model lifecycle + data/algo foundation + release evidence sign-off + production readiness gates)





