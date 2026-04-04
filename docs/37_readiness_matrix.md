# 37. Production Readiness Matrix (v2.5.0 full-platform line)

This matrix defines the objective sign-off contract for "production-ready" in the
`v2.5.0 -> v2.9.4` cycle.

Full-platform production envelope:
- single-node and multi-node training/serving validation paths
- web + mobile frontend contract targets
- multi-engine DB core gates
- bootstrap mainline default with Stage0 emergency fallback lane
- signed remote registry + local cache lifecycle contract
- VM runtime as normative execution contract

## Non-Hardware Readiness Gate Bundle

Run:

`enkai readiness check --profile full_platform --json --output artifacts/readiness/full_platform.json`

Selective pipeline reuse:
- `enkai readiness check ... --skip-check <id>` may omit checks already executed by a stronger release gate.
- Release pipelines use this to skip standalone self-host lanes when `enkai litec release-ci` is run separately.
- Generated deploy-validation smoke reports emitted by the full-platform profile:
  - `artifacts/readiness/deploy_backend.json`
  - `artifacts/readiness/deploy_fullstack.json`
  - `artifacts/readiness/deploy_backend_smoke.json`
  - `artifacts/readiness/deploy_fullstack_smoke.json`
- Generated simulation smoke reports emitted by the full-platform profile:
  - `artifacts/readiness/grpc_smoke.json`
  - `artifacts/readiness/grpc_evidence_verify.json`
  - `artifacts/readiness/sim_smoke.json`
  - `artifacts/readiness/sim_evidence_verify.json`
  - `artifacts/readiness/sim_native_smoke.json`
  - `artifacts/readiness/sim_native_evidence_verify.json`
  - `artifacts/readiness/sim_stdlib_smoke.json`
  - `artifacts/readiness/sim_stdlib_evidence_verify.json`
  - `artifacts/readiness/adam0_reference_suite.json`
  - `artifacts/readiness/adam0_reference_suite_verify.json`
  - `artifacts/readiness/model_registry_convergence.json`
  - `artifacts/readiness/model_registry_convergence_verify.json`
  - `artifacts/readiness/cluster_scale_smoke.json`
  - `artifacts/readiness/cluster_scale_evidence_verify.json`
  - `artifacts/readiness/registry_degraded_smoke.json`
  - `artifacts/readiness/registry_degraded_evidence_verify.json`
  - `artifacts/readiness/snn_agent_kernel_smoke.json`
  - `artifacts/readiness/snn_agent_kernel_evidence_verify.json`
  - `artifacts/sim/smoke_run.json`
  - `artifacts/sim/smoke_profile.json`
  - `artifacts/sim/smoke_replay.json`
  - `artifacts/sim/native_smoke_run.json`
  - `artifacts/sim/native_smoke_profile.json`
  - `artifacts/sim/stdlib_smoke_run.json`
  - `artifacts/sim/stdlib_smoke_profile.json`
  - `artifacts/sim/adam0_baseline_100_run.json`
  - `artifacts/sim/adam0_baseline_100_profile.json`
  - `artifacts/sim/adam0_baseline_100_snapshot.json`
  - `artifacts/sim/adam0_baseline_100_replay.json`
  - `artifacts/sim/adam0_stress_1000_run.json`
  - `artifacts/sim/adam0_stress_1000_profile.json`
  - `artifacts/sim/adam0_stress_1000_snapshot.json`
  - `artifacts/sim/adam0_stress_1000_replay.json`
  - `artifacts/sim/adam0_target_10000_run.json`
  - `artifacts/sim/adam0_target_10000_profile.json`
  - `artifacts/sim/adam0_target_10000_snapshot.json`
  - `artifacts/sim/adam0_target_10000_replay.json`
  - `artifacts/sim/snn_agent_kernel_run.json`
  - `artifacts/sim/snn_agent_kernel_profile.json`
  - `artifacts/grpc/probe.json`
  - `artifacts/grpc/server.jsonl`
  - `artifacts/grpc/conversation_state.json`
  - `artifacts/registry/sim_lineage.json`
  - `artifacts/registry/sim_snapshot.manifest.json`
  - `artifacts/registry/local/registry.json`
  - `artifacts/registry/remote/registry.json`
  - `artifacts/registry/cache/registry.json`
  - `artifacts/registry/remote/adam0-sim/v<version>/remote.manifest.json`
  - `artifacts/registry/remote/adam0-sim/v<version>/remote.manifest.sig`
  - `artifacts/cluster_scale/run.json`
  - `artifacts/registry_degraded/cache/audit.log.jsonl`
- `artifacts/validation/ffi_correctness.json`
- `artifacts/validation/determinism_event_queue.json`
- `artifacts/validation/determinism_sim_replay.json`
- `artifacts/validation/determinism_adam0_reference_100.json`
- `artifacts/validation/determinism_sim_coroutines.json`
- `artifacts/validation/pool_safety.json`
- `artifacts/validation/adam0_fake10.json`
- `artifacts/validation/adam0_ref100.json`
- `artifacts/validation/adam0_stress1000.json`
- `artifacts/validation/adam0_target10000.json`
- `artifacts/validation/perf_ffi_noop.json`
- `artifacts/validation/perf_sparse_dot.json`
  - `artifacts/validation/perf_adam0_reference_100.json`
  - `artifacts/validation/perf_adam0_reference_1000.json`
  - `artifacts/validation/perf_adam0_reference_10000.json`

Manifest:
- `enkai/contracts/readiness_full_platform_v2_5_0.json`
- release blocker matrix:
  - `enkai/contracts/full_platform_release_blockers_v2_5_0.json`
- machine-readable blocker verification:
  - `enkai readiness verify-blockers --profile full_platform --report artifacts/readiness/full_platform.json --json --output artifacts/readiness/full_platform_blockers.json`
  - output:
    - `artifacts/readiness/full_platform_blockers.json`
  - release pipelines may add:
    - `--allow-skipped-required-check selfhost-mainline`
    - `--allow-skipped-required-check selfhost-stage0-fallback`
    because those readiness checks are intentionally replaced by the stronger `enkai litec release-ci` gate in the consolidated pipeline.
  - strict release evidence archives both:
    - `artifacts/readiness/full_platform.json`
    - `artifacts/readiness/full_platform_blockers.json`
  - strict capability reporting requires the archived blocker report to be present and passing.

The command executes a deterministic gate bundle:
- format/lint/test
- docs/spec consistency
- frontend/backend contract snapshot tests
- backend HTTP contract smoke
- backend gRPC runtime smoke
- backend gRPC runtime evidence verification
- LLM runtime smoke
- simulation std/runtime CLI smoke
- simulation smoke evidence semantic verification
- simulation native FFI smoke
- simulation native FFI evidence verification
- simulation stdlib primitive smoke with native-backed acceleration proof
- simulation stdlib evidence verification
- Adam-0 reference suite (100 / 1000 / 10000 agents)
- Adam-0 reference suite evidence verification
- signed model/simulation registry convergence smoke
- signed model/simulation registry convergence verification
- multi-node simulation cluster scale smoke
- multi-node simulation cluster scale evidence verification
- registry degraded-mode fallback smoke
- registry degraded-mode fallback evidence verification
- SNN/agent environment kernel smoke
- SNN/agent environment kernel evidence verification
- DB core smoke
- generated backend/fullstack deploy validation smoke
- bootstrap mainline + Stage0 fallback lanes
- benchmark fairness + target smoke enforcement (`official_v2_3_0_matrix`, workload-equivalence contract)
- CPU validation proof suites:
  - FFI correctness
  - event-queue determinism
  - simulation replay determinism
  - pool safety
  - Adam-0 fake10 + ref100 CPU validation
  - baseline capture for FFI noop, sparse dot, and Adam-0 ref100

## GPU Evidence (Release Blocking)

`v2.9.4` release sign-off requires operator evidence and verifier pass:
- single-GPU stability evidence
- 2-GPU loss/grad parity evidence
- 4-GPU soak evidence

Verification:
- `scripts/verify_gpu_gates.ps1`
- `scripts/verify_gpu_gates.sh`

## Sign-Off Rule

A release is marked "production-ready" only when:
- full-platform non-hardware readiness bundle is green, and
- RC/release artifact gates are green, and
- GPU evidence package is present and verifier-clean.
