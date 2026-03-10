# 33. Benchmark Suite (v2.3.0 Matrix)

This document defines the benchmark contract used for bounded Enkai performance claims.

## CLI

Run benchmark suites with:

- `enkai bench run --suite <name> --baseline <python|none> --output <file>`

Additional controls:

- `--iterations <n>`
- `--warmup <n>`
- `--machine-profile <file>`
- `--target-speedup <pct>`
- `--target-memory <pct>`
- `--enforce-target`
- `--enforce-all-cases`
- `--enforce-class-targets --class-targets <file>`
- `--fairness-check-only`
- `--equivalence-contract <file>`
- `--profile-case <id> --profile-output <file>`

Profile a single case with VM/native counters:

- `enkai bench profile --case <id> --output bench/results/profiles/<id>.json`

## Suite Layout

- Suite specs: `bench/suites/*.json`
- Class matrix:
  - `bench/suites/official_v2_3_0_vm_compute.json`
  - `bench/suites/official_v2_3_0_native_bridge.json`
  - `bench/suites/official_v2_3_0_cli_workflows.json`
  - `bench/suites/official_v2_3_0_ai_data_workflows.json`
  - `bench/suites/official_v2_3_0_matrix.json`
- Class targets: `bench/suites/official_v2_3_0_targets.json`
- Fairness contract: `bench/contracts/workload_equivalence_v1.json`
- Frozen baseline: `bench/baselines/v2_2_0/pre_recovery_baseline.json`
- Enkai workloads: `bench/enkai/*.enk`
- Python baselines: `bench/python/*.py`
- Machine profiles: `bench/machines/*.json`
- Result artifacts: `bench/results/*.json`
- Per-case profile artifacts: `bench/results/profiles/*.json`

## Reporting Contract

Result JSON schema (`schema_version: 2`) records:

- per-case samples for Enkai and Python baseline
- per-case fairness metadata
- class summaries (median speedup and memory reduction)
- optional class target payload and class gate failures
- suite-level summary pass/fail

## Fairness Policy

All official cases must include workload-equivalence metadata and pass contract checks before execution:

- work units
- payload size
- batching mode
- warmup policy

If machine profile is provided, Python major.minor must exactly match the pinned profile value.

## Claim Policy

Performance claims are bounded to:

- declared class matrix suite(s)
- pinned machine profile(s)
- recorded tool/runtime versions

For release blocking in v2.3.0:

- every official case must satisfy class targets in `official_v2_3_0_targets.json`
- memory reduction floor is enforced per case

No universal cross-hardware claim is implied.
