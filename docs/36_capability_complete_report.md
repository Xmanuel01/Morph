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
  - `full_platform.json` (`readiness/full_platform.json`) for the v2.5+ full-platform line
  - `full_platform_blockers.json` (`readiness/full_platform_blockers.json`) as the archived blocker-verification verdict for the current release line
  - `sim_smoke.json` (`readiness/sim_smoke.json`) summarizing the archived simulation smoke workflow
  - `sim_evidence_verify.json` (`readiness/sim_evidence_verify.json`) validating archived simulation evidence consistency
  - `sim_native_smoke.json` (`readiness/sim_native_smoke.json`) summarizing the archived native FFI simulation smoke workflow
  - `sim_native_evidence_verify.json` (`readiness/sim_native_evidence_verify.json`) validating archived native FFI simulation evidence consistency
  - `sim_stdlib_smoke.json` (`readiness/sim_stdlib_smoke.json`) summarizing the archived stdlib simulation primitive smoke workflow, including native-backed acceleration requirements
  - `sim_stdlib_evidence_verify.json` (`readiness/sim_stdlib_evidence_verify.json`) validating archived stdlib simulation evidence consistency
  - `snn_agent_kernel_smoke.json` (`readiness/snn_agent_kernel_smoke.json`) summarizing the archived SNN/agent kernel smoke workflow
  - `snn_agent_kernel_evidence_verify.json` (`readiness/snn_agent_kernel_evidence_verify.json`) validating archived SNN/agent kernel evidence consistency
  - `production.json` (`readiness/production.json`) as an optional compatibility/reference artifact
- `sim/`:
  - `sim/smoke_run.json`
  - `sim/smoke_profile.json`
  - `sim/smoke_replay.json`
  - `sim/native_smoke_run.json`
  - `sim/native_smoke_profile.json`
  - `sim/stdlib_smoke_run.json`
  - `sim/stdlib_smoke_profile.json`
  - `sim/snn_agent_kernel_run.json`
  - `sim/snn_agent_kernel_profile.json`
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
In strict mode the archived blocker report must also be present and must show:
- `all_passed: true`
- `skip_release_evidence: false`
- no missing/failed/skipped required checks
- no missing required release artifacts
- simulation std/runtime, stdlib primitive, native FFI, Adam-0, and SNN/agent kernel evidence artifacts must be present under both `readiness/` and `sim/`

## RC Pipeline Contract

- Dry-run (CPU/CI): generates non-strict capability report.
- Full RC (release sign-off): requires GPU evidence and strict capability checks.

The RC wrappers for the current line are:

- `scripts/v2_3_0_rc_pipeline.ps1`
- `scripts/v2_3_0_rc_pipeline.sh`

