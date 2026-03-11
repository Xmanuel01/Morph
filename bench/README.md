# Enkai Benchmark Harness (Official v2.3.0 Class Matrix)

This harness provides reproducible Enkai-vs-Python benchmarking with strict fairness checks and
class-based gating.

## Core Commands

- Run suite:
  - `enkai bench run --suite <name> --baseline python --output <file>`
- Fairness-only precheck:
  - `enkai bench run --suite official_v2_3_0_matrix --baseline python --fairness-check-only --output bench/results/fairness.json`
- Release target gate:
  - `enkai bench run --suite official_v2_3_0_matrix --baseline python --iterations 2 --warmup 1 --machine-profile bench/machines/windows_ref.json --target-speedup 15 --target-memory 5 --enforce-target --enforce-class-targets --class-targets bench/suites/official_v2_3_0_targets.json --output bench/results/official_v2_3_0_matrix.windows.json`
- Per-case profiler output:
  - `enkai bench profile --case <id> --output bench/results/profiles/<id>.json`

## Official Assets

- Official matrix suite:
  - `bench/suites/official_v2_3_0_matrix.json`
- Class suites:
  - `bench/suites/official_v2_3_0_vm_compute.json`
  - `bench/suites/official_v2_3_0_native_bridge.json`
  - `bench/suites/official_v2_3_0_cli_workflows.json`
  - `bench/suites/official_v2_3_0_ai_data_workflows.json`
- Class target thresholds:
  - `bench/suites/official_v2_3_0_targets.json`
- Workload-equivalence contract:
  - `bench/contracts/workload_equivalence_v1.json`
- Frozen pre-recovery baseline:
  - `bench/baselines/v2_2_0/pre_recovery_baseline.json`

## Output Contract

Benchmark reports use `schema_version: 2` and include:
- suite + machine profile metadata
- fairness contract status
- per-case Enkai/Python samples and deltas
- per-case pass/fail
- class summaries + class gate failures
- summary gate status

## Pinned Environments

Use pinned reference machine profiles for bounded claims:
- `bench/machines/linux_ref.json`
- `bench/machines/windows_ref.json`

For release sign-off, benchmark claims are bounded to:
- the official v2.3.0 class matrix suite
- pinned machine profiles
- archived result artifacts in `bench/results/*.json` and release evidence under `artifacts/release/`
