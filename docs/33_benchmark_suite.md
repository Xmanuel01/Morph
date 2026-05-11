# 33. Benchmark Suite

This document defines the benchmark contract used for bounded Enkai performance claims.
Benchmarks are evidence artifacts, not marketing claims: every claim must name the suite, machine profile, runtime versions, and verifier result.

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
- AI optimization sweep matrix (includes official matrix + training/inference/algorithm kernels):
  - `bench/suites/ai_full_v2_4_0.json`
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

For release blocking on frozen suites:

- every official case must satisfy class targets in `official_v2_3_0_targets.json`
- memory reduction floor is enforced per case

No universal cross-hardware claim is implied.

## CUDA/PyTorch Claim Rule

The v3.9.0 CUDA-first LLM runtime line has its own stricter verifier. A CUDA performance claim is blocked unless the archived hardware run reaches the contracted PyTorch comparison thresholds and writes green readiness artifacts. Use `docs/tensor_api.md` and `docs/gpu_backend.md` for the GPU proof commands.

## AI Optimization Sweep

Use this suite to rank optimization priorities across AI-related workloads:

- `enkai bench run --suite ai_full_v2_4_0 --baseline python --iterations 2 --warmup 1 --machine-profile bench/machines/windows_ref.json --output bench/results/ai_full_v2_4_0.windows.json`
- `enkai bench profile --suite ai_full_v2_4_0 --case <id> --output bench/results/profiles/<id>.json`
