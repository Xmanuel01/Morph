# 42. AGI Runbook

This runbook defines the bounded operational path for AGI-class simulation
workloads on Enkai.

## Scope

This runbook covers:

- `std::sim`
- `std::sparse`
- `std::event`
- `std::pool`
- `std::spatial`
- `std::snn`
- `std::agent`
- `enkai sim run|profile|replay`
- `enkai cluster validate|plan|run`
- Adam-0 bounded reference workloads

This runbook does not claim universal AGI readiness.
It defines the supported operational envelope for the current platform.

## Supported Envelope

- Single-node deterministic simulation:
  - 100-agent baseline
  - 1000-agent stress
  - 10000-agent bounded reference target
- Multi-node supervised simulation:
  - planned with `enkai cluster plan`
  - executed with `enkai cluster run`
  - bounded retry/recovery
  - snapshot/replay recovery windows
- Native acceleration:
  - sparse/event/pool hot paths
  - spatial queries
  - SNN batch update hooks
- Deterministic fallback:
  - `ENKAI_SIM_ACCEL=0`

## Required Preconditions

- `enkai readiness check --profile full_platform --json --output artifacts/readiness/full_platform.json`
- Adam-0 smoke/reference evidence present and verifier-clean:
  - `artifacts/readiness/adam0_100_smoke.json`
  - `artifacts/readiness/adam0_100_evidence_verify.json`
  - `artifacts/readiness/adam0_reference_suite.json`
  - `artifacts/readiness/adam0_reference_suite_verify.json`
- SNN/agent kernel evidence present and verifier-clean:
  - `artifacts/readiness/snn_agent_kernel_smoke.json`
  - `artifacts/readiness/snn_agent_kernel_evidence_verify.json`
- Cluster-scale evidence present and verifier-clean for multi-node usage:
  - `artifacts/readiness/cluster_scale_smoke.json`
  - `artifacts/readiness/cluster_scale_evidence_verify.json`

## Daily Operation

1. Validate the workload config.
   - `enkai cluster validate <config.enk> --json --output artifacts/cluster_scale/validate.json`
2. Generate a plan and inspect world partitioning.
   - `enkai cluster plan <config.enk> --json --output artifacts/cluster_scale/plan.json`
3. Run the workload under bounded supervision.
   - `enkai cluster run <config.enk> --json --output artifacts/cluster_scale/run.json`
4. On single-node runs, emit run/profile/snapshot artifacts.
   - `enkai sim run <entry> --json --output artifacts/sim/run.json --snapshot-output artifacts/sim/snapshot.json`
   - `enkai sim profile <entry> --output artifacts/sim/profile.json`
   - `enkai sim replay <entry> --json --output artifacts/sim/replay.json`

## Failure Handling

- Determinism failure:
  - compare run/snapshot/replay artifact hashes
  - reject rollout if replay diverges from snapshot lineage
- Native extension failure:
  - rerun with `ENKAI_SIM_ACCEL=0`
  - if fallback succeeds, isolate the native module before resuming production use
- Cluster window failure:
  - inspect `artifacts/cluster_scale/recovery/`
  - rerun the failed window from the latest valid snapshot
- Registry/signature failure:
  - use cached local artifacts only
  - do not promote or retire versions until signature verification is restored

## Release Sign-Off

AGI sign-off for the current line requires:

- full-platform non-hardware readiness green
- Adam-0 bounded suite verifier-clean
- SNN/agent kernel verifier-clean
- cluster-scale verifier-clean
- strict archived evidence bundle present under `artifacts/release/v<version>/`

