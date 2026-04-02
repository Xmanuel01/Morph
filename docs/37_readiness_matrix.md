# 37. Production Readiness Matrix (v2.5.0 full-platform line)

This matrix defines the objective sign-off contract for "production-ready" in the
`v2.5.0 -> v2.9.0` cycle.

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
  - `artifacts/readiness/sim_smoke.json`
  - `artifacts/readiness/sim_evidence_verify.json`
  - `artifacts/readiness/sim_native_smoke.json`
  - `artifacts/readiness/sim_native_evidence_verify.json`
  - `artifacts/readiness/sim_stdlib_smoke.json`
  - `artifacts/readiness/sim_stdlib_evidence_verify.json`
  - `artifacts/readiness/adam0_reference_suite.json`
  - `artifacts/readiness/adam0_reference_suite_verify.json`
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
- LLM runtime smoke
- simulation std/runtime CLI smoke
- simulation smoke evidence semantic verification
- simulation native FFI smoke
- simulation native FFI evidence verification
- simulation stdlib primitive smoke with native-backed acceleration proof
- simulation stdlib evidence verification
- Adam-0 reference suite (100 / 1000 / 10000 agents)
- Adam-0 reference suite evidence verification
- SNN/agent environment kernel smoke
- SNN/agent environment kernel evidence verification
- DB core smoke
- generated backend/fullstack deploy validation smoke
- bootstrap mainline + Stage0 fallback lanes
- benchmark fairness + target smoke enforcement (`official_v2_3_0_matrix`, workload-equivalence contract)

## GPU Evidence (Release Blocking)

`v2.9.0` release sign-off requires operator evidence and verifier pass:
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
