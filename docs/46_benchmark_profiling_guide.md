# 46. Benchmark And Profiling Guide

This guide defines how to interpret Enkai benchmark and VM/native profiling
artifacts.

## Purpose

Enkai benchmark results are bounded to official suites and pinned environments.
They are not universal language-wide claims.

## Command Surfaces

- suite run:
  - `enkai bench run --suite official_v2_3_0_matrix --baseline python --output <file>`
- fairness-only run:
  - `enkai bench run --suite official_v2_3_0_matrix --baseline python --fairness-check-only --output <file>`
- per-case profile:
  - `enkai bench profile --case <id> --output bench/results/profiles/<id>.json`

## What To Look At

- throughput / latency deltas
- memory reduction
- fairness contract status
- VM/native boundary counters:
  - `ffi_calls`
  - `native_calls`
  - marshal bytes
  - copy count
- simulation profile fields for sparse/native evidence

## Interpretation Rules

- If fairness fails, performance results are not sign-off quality.
- If native-backed paths are expected, profiles must prove the path was exercised.
- If a benchmark win exists only because a required verifier failed, the result is invalid.
- CPU-only and GPU-backed results must be interpreted separately.

## Release Use

For release sign-off:

- benchmark target gates must pass
- class-based thresholds must pass
- required profiling/evidence artifacts must be archived under `artifacts/release/v<version>/`

