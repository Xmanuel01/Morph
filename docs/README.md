# Enkai Docs

## Versioning policy

- Current release line: `v3.4.0` (`strict-selfhost closure complete; post-closure baseline scopes closed`).
- `docs/Enkai.spec` is the normative language reference for `v3.4.0`.
- Known limits and partial features are tracked inside `docs/Enkai.spec` under `Known Limits in v3.4.0`.
- `docs/40_registry_convergence.md` defines the registry convergence contract for checkpoint, simulation, environment, and native-extension artifacts.
- `docs/41_scale_reliability.md` defines the `v3.0.0` scale/reliability contract for supervised simulation clusters and degraded registry fallback.
- `VALIDATION.md` defines the strict CPU/GPU validation matrix for release claims.
- `bench/machines/` and `bench/baselines/validation_cpu_v3_0_0.json` define the local/reference CPU validation envelopes and regression baselines.
- `docs/47_gpu_operator_preflight.md` defines the operator preflight path for real GPU sign-off hosts.
- `docs/42_agi_runbook.md`, `docs/43_llm_runbook.md`, `docs/44_native_extension_abi_policy.md`, `docs/45_deployment_rollback_runbook.md`, and `docs/46_benchmark_profiling_guide.md` are the v3.0.0 publication assets for the final stability cut.
- `docs/49_v3_0_0_quality_assurance.md` records the final CPU-side QA results and the remaining GPU evidence blocker.
- `docs/50_strict_selfhost_contract.md` freezes the `v3.1.0 -> v4.0.0` zero-Rust self-host dependency boundary and blocker inventory.
- `docs/51_full_frontend_frontier.md` records the current measurable gap between the bootstrap subset and a full-language self-host frontend.
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
5. `docs/33_benchmark_suite.md`, `docs/34_model_lifecycle.md`, `docs/35_data_algo_stack.md`, `docs/36_capability_complete_report.md`, `docs/37_readiness_matrix.md`, `docs/39_adam0_reference_stack.md`, `VALIDATION.md`, `docs/42_agi_runbook.md`, `docs/43_llm_runbook.md`, `docs/44_native_extension_abi_policy.md`, `docs/45_deployment_rollback_runbook.md`, `docs/46_benchmark_profiling_guide.md`, `docs/48_release_dashboard.md`, `docs/49_v3_0_0_quality_assurance.md`, `docs/50_strict_selfhost_contract.md`, `docs/51_full_frontend_frontier.md` (benchmark policy + model lifecycle + data/algo foundation + release evidence sign-off + production readiness gates + Adam-0 bounded reference suite + explicit validation matrix + v3.0.0 runbooks/policy guides + final QA findings + strict self-host boundary + full frontend frontier/negative audit coverage)







