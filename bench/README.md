# Enkai Benchmark Harness (v2.2.0)

This harness provides reproducible Enkai-vs-Python benchmarks with structured JSON output.

## Command

- `enkai bench run --suite <name> --baseline python --output <file>`

Official suite for v2.2.0:
- `enkai bench run --suite official_v2_2_0 --baseline python --output bench/results/official_v2_2_0.json`

## Suites

- `bench/suites/core.json`: deterministic numeric/json/hash kernels.
- `bench/suites/train_step.json`: training-step style throughput kernel.
- `bench/suites/inference.json`: inference token-loop throughput kernel.
- `bench/suites/tokenizer_dataset.json`: tokenizer + dataset CLI throughput.
- `bench/suites/http_serving.json`: serving request-loop throughput.
- `bench/suites/db_ops.json`: SQLite throughput.
- `bench/suites/algorithm_kernels.json`: algorithm + ML utility stack throughput.
- `bench/suites/official_v2_1_0.json`: historical baseline suite.
- `bench/suites/official_v2_2_0.json`: current normative suite.

## Output Schema

`schema_version: 1` report includes:
- suite metadata
- per-case Enkai and Python sample series
- per-case deltas (`speedup_pct`, `memory_reduction_pct`)
- summary medians and pass/fail status

## Target Policy

For bounded performance claims, use:

- `--target-speedup 5 --target-memory 5 --enforce-target`

This enforces >=5% median speedup and >=5% median memory reduction on the suite.
For strict per-case target enforcement, add:

- `--enforce-all-cases`

## Machine Profiles

Use pinned host manifests for bounded claim reporting:
- `bench/machines/linux_ref.json`
- `bench/machines/windows_ref.json`

The performance claim in v2.1.x is bounded to:
- the official suite definition (`official_v2_2_0`)
- pinned machine profile manifests
- recorded report artifacts under `bench/results/*.json`
