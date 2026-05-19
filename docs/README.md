# Enkai Docs

This directory contains both learner documentation and release/proof documentation.
If you are new to Enkai, start with the learner path below before reading the
release ledgers.

## Learner Reading Order

1. `docs/53_language_handbook.md` - one-stop learner handbook for syntax, CLI, policies, stdlib, tensors, training, and proof boundaries.
2. `docs/00_getting_started.md` - install check, first program, CLI basics, and learning path.
3. `docs/01_syntax.md` - blocks, variables, imports, policies, arrays, and complete examples.
4. `docs/02_types.md` - primitives, arrays, vectors, tensors, constants, and inference rules.
5. `docs/03_functions.md` - function syntax, parameters, returns, and mutation patterns.
6. `docs/04_scopes.md` - function scope, block scope, shadowing, and mutation scope.
7. `docs/05_globals.md` - top-level bindings and global API guidance.
8. `docs/06_logic_ops.md` - arithmetic, comparison, boolean logic, and strings.
9. `docs/08_errors.md` - syntax/type/import/policy/runtime diagnostics.
10. `docs/09_modules.md` - standard-library and local module imports.
11. `docs/10_visibility.md` - public exports and private helpers.
12. `docs/11_formatter.md` - `enkai fmt` and official tagged-closer style.
13. `docs/12_testing.md` - `enkai test` and recommended local validation loop.
14. `docs/20_json.md` - `std::json`, `json.enkai`, parsing, and import rules.
15. `docs/52_cli_and_style.md` - complete official CLI/style summary.
16. `docs/tensor_api.md` - AI-native tensor API, current acceleration boundaries, and proof gates.
17. `docs/54_llm_package_registry.md` - stable LLM package manifests, lockfiles, dependency resolution, and integrity gates.
18. `docs/55_app_platform_closure.md` - bounded and live MySQL, gRPC, and mobile platform closure proof.
19. `docs/56_tensor_ffi_opaque_handles.md` - tensor FFI opaque capability-token handle closure.
20. `docs/57_native_snn_batched_kernels.md` - native SNN batched kernel closure.
21. `docs/58_multi_node_orchestration_closure.md` - real-hardware multi-node orchestration closure.
22. `docs/Enkai.spec` - normative language reference.

## Official CLI For Users

```text
enkai run <file.enkai>      Run an Enkai file or project.
enkai check <file.enkai>    Check syntax, types, imports, and policies.
enkai fmt <file.enkai>      Format source.
enkai build [dir]           Build/check a project.
enkai test [dir]            Run project tests.
enkai help                  Show the full CLI.
enkai version               Print the version.
```

`enkai safari` is reserved for a future interactive workspace. Use `enkai run`
for normal execution.

## Documentation Status Rule

Learner docs describe how to write Enkai code. Release and readiness docs
describe what has been proven. If a GPU, distributed, security, or production
claim does not have a green verifier artifact under `artifacts/readiness/`, treat
it as planned or proof-ready, not closed.

## Current Release / Proof Context

- Current closed release line: `v3.8.0`.
- Next active proof line: `v3.9.0` CUDA-first production LLM runtime foundation.
- `docs/Enkai.spec` is the normative language reference for the closed `v3.8.0` line.
- `docs/roadmap.md` tracks release direction and proof status.
- `VALIDATION.md` defines strict CPU/GPU validation matrices for release claims.
- GPU production claims require archived hardware evidence; proof-ready code is not the same as green hardware sign-off.

## Release / Engineering Reading Order

1. `docs/roadmap.md`, `docs/30_v1_release_audit.md`, `docs/31_v2_rc_notes.md`, `docs/32_v2_migration_guide.md`.
2. `docs/07_bytecode_vm.md`, `docs/13_ffi.md`, `docs/14_native_modules.md`.
3. `docs/18_http_server.md`, `docs/19_http_client.md`, `docs/26_serve_cli.md`, `docs/27_frontend_stack.md`.
4. `docs/bootstrap_subset.md`, `docs/bootstrap_core.md`, `docs/28_selfhost_workflow.md`, `docs/29_compatibility_policy.md`.
5. `docs/33_benchmark_suite.md`, `docs/34_model_lifecycle.md`, `docs/35_data_algo_stack.md`, `docs/37_readiness_matrix.md`.
6. `docs/39_adam0_reference_stack.md`, `docs/42_agi_runbook.md`, `docs/43_llm_runbook.md`.
7. `docs/44_native_extension_abi_policy.md`, `docs/45_deployment_rollback_runbook.md`, `docs/46_benchmark_profiling_guide.md`, `docs/47_gpu_operator_preflight.md`.
8. `docs/48_release_dashboard.md`, `docs/49_v3_0_0_quality_assurance.md`, `docs/50_strict_selfhost_contract.md`, `docs/51_full_frontend_frontier.md`.
9. `docs/54_llm_package_registry.md` for the `v4.0` LLM package/model ecosystem tranche.
10. `docs/55_app_platform_closure.md` for bounded MySQL, gRPC, and mobile platform closure.
11. `docs/56_tensor_ffi_opaque_handles.md` for tensor FFI handle-hardening closure.
12. `docs/57_native_snn_batched_kernels.md` for native SNN batched kernel closure.
13. `docs/58_multi_node_orchestration_closure.md` for real-hardware multi-node orchestration closure.

## Historical Docs

`docs/v0.1-status.md`, `docs/v0.2-status.md`, and `docs/v0.3-plan.md` are
historical snapshots. They preserve old milestone language and should not be read
as the current user-facing syntax guide.
