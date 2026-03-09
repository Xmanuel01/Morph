# 33. Benchmark Suite (v2.1.8)

This document defines the benchmark contract used for bounded Enkai performance claims.

## CLI

Run benchmarks with:

`enkai bench run --suite <name> --baseline <python|none> --output <file>`

Optional controls:
- `--iterations <n>`
- `--warmup <n>`
- `--machine-profile <file>`
- `--target-speedup <pct>`
- `--target-memory <pct>`
- `--enforce-target`
- `--enforce-all-cases`
- `--python <command>`
- `--enkai-bin <path>`

## Suite Layout

- Suite specs: `bench/suites/*.json`
- Enkai workloads: `bench/enkai/*.enk`
- Python baselines: `bench/python/*.py`
- Machine profiles: `bench/machines/*.json`
- Result artifacts: `bench/results/*.json`

Official bounded claim suite for `v2.1.8`:
- `bench/suites/official_v2_1_8.json`

## Reporting Contract

Result JSON schema (`schema_version: 1`) records:
- per-case samples for Enkai and baseline
- median wall-clock time
- median peak RSS memory
- speedup and memory reduction deltas
- suite-level summary with pass/fail

## Claim Policy

Performance claims are bounded to:
- declared suite(s)
- pinned machine profile(s)
- recorded tool/runtime versions

For `v2.1.8`, target enforcement defaults to suite-level medians:
- `median_speedup_pct >= target_speedup_pct`
- `median_memory_reduction_pct >= target_memory_pct`

Use `--enforce-all-cases` when every benchmark case must individually satisfy targets.

No universal cross-hardware claim is implied.
