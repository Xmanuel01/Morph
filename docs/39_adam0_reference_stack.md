# 39. Adam-0 Reference Stack (v3.0.0)

This document defines the bounded `v3.0.0` Adam-0 reference stack and the
release evidence required to claim the milestone is complete.

## Scope

- orchestration remains in Enkai std/runtime surfaces
- hot paths use native-backed execution through existing `native::import`
- no syntax changes
- bounded production envelope:
  - single-node native acceleration enabled
  - reference host class: 8 logical cores, 16 GiB RAM, SSD-backed workspace

## Reference Workload

- reference package entrypoint: `examples/adam0_reference/main.enk`
- fake integration package entrypoint: `examples/adam0_fake/main.enk`
- official suite definition:
  - `bench/suites/adam0_reference_v2_9_4.json`

Suite cases:
- `adam0_baseline_100`
- `adam0_stress_1000`
- `adam0_target_10000`

The suite is parameterized through environment variables, not baked-in constants:
- `ENKAI_ADAM0_CASE_ID`
- `ENKAI_ADAM0_AGENT_COUNT`
- `ENKAI_ADAM0_SHARD_SIZE`
- `ENKAI_ADAM0_ROUNDS`
- `ENKAI_ADAM0_FANOUT`
- `ENKAI_ADAM0_POOL_CAPACITY`
- `ENKAI_ADAM0_MAX_EVENTS`
- `ENKAI_ADAM0_SEED`
- `ENKAI_ADAM0_NEIGHBOR_RADIUS_MILLI`
- `ENKAI_ADAM0_MOVE_STEP_MILLI`
- `ENKAI_ADAM0_SPACING_MILLI`
- `ENKAI_ADAM0_HARDWARE_CLASS`
- `ENKAI_ADAM0_REFERENCE_HOST`

## Evidence

Generate suite evidence with:

- `python3 scripts/readiness_adam0_reference_suite.py --enkai-bin <enkai> --workspace . --output artifacts/readiness/adam0_reference_suite.json`

Verify suite evidence with:

- `python3 scripts/verify_adam0_reference_suite.py --summary artifacts/readiness/adam0_reference_suite.json --output artifacts/readiness/adam0_reference_suite_verify.json`

Required archived artifacts:
- `artifacts/readiness/adam0_reference_suite.json`
- `artifacts/readiness/adam0_reference_suite_verify.json`
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

## Acceptance

`v3.0.0` is complete only when:
- all three suite cases complete successfully
- archived verification passes
- native evidence is visible in profiles:
  - `ffi_calls > 0`
  - `native_function_calls > 0`
  - `sim_coroutines_spawned > 0`
  - `timing_ms.native_calls > 0`
- replay artifacts are consistent with the corresponding run snapshots
