# 36. Capability-Complete Report (v2.3.0)

This document defines the objective release evidence contract for the `v2.3.0`
stability cut.

## Goal

Produce a reproducible evidence bundle and a machine-parseable capability report
for release sign-off, instead of narrative-only claims.

## Evidence Bundle

Release evidence is archived with:

- `python3 scripts/collect_release_evidence.py --gpu-log-dir artifacts/gpu --require-gpu --strict`

The archive is written to:

- `artifacts/release/v<version>/`

Core categories:

- `dist/`:
  - release archive
  - checksum
  - SBOM
  - benchmark target result (`benchmark_official_<suite>_<platform>.json`)
- `selfhost/`:
  - `litec_selfhost_ci_report.json`
  - `litec_replace_check_report.json`
  - `litec_mainline_ci_report.json`
  - `litec_release_ci_report.json`
- `contracts/`:
  - `backend_api_v1.snapshot.json`
  - `sdk_api_v1.snapshot.json`
  - `conversation_state_v1.schema.json`
- `readiness/`:
  - `production.json` (`readiness/production.json`)
- `gpu/` (mandatory for full sign-off):
  - `single_gpu.log`, `single_gpu_evidence.json`
  - `multi_gpu.log`, `multi_gpu_evidence.json`
  - `soak_4gpu.log`, `soak_4gpu_evidence.json`

## Capability Report

Generate report artifacts from the archived manifest:

- `python3 scripts/generate_capability_report.py --require-gpu --strict`

Outputs:

- `artifacts/release/v<version>/capability_complete.json`
- `artifacts/release/v<version>/capability_complete.md`

`--strict` fails if any required check fails.

## RC Pipeline Contract

- Dry-run (CPU/CI): generates non-strict capability report.
- Full RC (release sign-off): requires GPU evidence and strict capability checks.

The RC wrappers for the current line are:

- `scripts/v2_3_0_rc_pipeline.ps1`
- `scripts/v2_3_0_rc_pipeline.sh`

