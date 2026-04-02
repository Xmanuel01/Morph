# 41. Scale And Reliability (v2.8.1)

This document defines the `v2.8.1` contract for bounded multi-node simulation
supervision and degraded registry fallback.

## Cluster Supervision

`enkai cluster` remains additive and syntax-neutral.

Supported `v2.8.1` planning/supervision fields:
- `dist.hosts`
- `dist.host_map`
- `workload = "simulation"`
- `simulation.target`
- `simulation.partition_count`
- `simulation.total_steps`
- `simulation.step_window`
- `simulation.snapshot_interval`
- `simulation.recovery_dir`
- `simulation.route_policy`

Command surface:
- `enkai cluster validate <config.enk> [--json] [--output <file>]`
- `enkai cluster plan <config.enk> [--json] [--output <file>]`
- `enkai cluster run <config.enk> [--dry-run] [--json] [--output <file>]`

`enkai cluster run` in `v2.8.1` executes bounded simulation workloads by:
- spawning rank-local `enkai sim run`
- continuing with windowed `enkai sim replay`
- persisting per-window reports, logs, manifests, and snapshots
- retrying failed windows up to `dist.retry_budget`
- resuming from the last completed snapshot when recovery is possible

Train multi-node execution remains operator-managed in this release line.

## Registry Degraded Fallback

Remote registry pull flows now have release-gated degraded-mode validation using:
- `enkai model pull <registry_dir> <name> <version> --registry <remote> --verify-signature --fallback-local`

`v2.8.1` readiness evidence requires proof that:
- a signed remote pull succeeds when the remote registry is available
- a later pull succeeds with `--fallback-local` when the remote registry is unavailable
- the local audit log records:
  - `operation = "pull_remote"`
  - `status = "fallback_local"`

## Release Evidence

Readiness artifacts:
- `artifacts/readiness/cluster_scale_smoke.json`
- `artifacts/readiness/cluster_scale_evidence_verify.json`
- `artifacts/readiness/registry_degraded_smoke.json`
- `artifacts/readiness/registry_degraded_evidence_verify.json`

Cluster supervision evidence:
- `artifacts/cluster_scale/validate.json`
- `artifacts/cluster_scale/plan.json`
- `artifacts/cluster_scale/run.json`
- `artifacts/cluster_scale/recovery/rank0/window_0000.run.json`
- `artifacts/cluster_scale/recovery/rank0/window_0000.snapshot.json`
- `artifacts/cluster_scale/recovery/rank1/window_0000.run.json`
- `artifacts/cluster_scale/recovery/rank1/window_0000.snapshot.json`

Registry degraded evidence:
- `artifacts/registry_degraded/cache/registry.json`
- `artifacts/registry_degraded/cache/audit.log.jsonl`
- `artifacts/registry_degraded/remote_offline/adam0-degraded/v2.8.1/remote.manifest.json`
- `artifacts/registry_degraded/remote_offline/adam0-degraded/v2.8.1/remote.manifest.sig`

## Exit Criteria

`v2.8.1` is complete only when:
- cluster scale smoke and semantic verification both pass
- at least one partition demonstrates snapshot-based retry recovery
- degraded registry fallback smoke and semantic verification both pass
- strict release evidence and capability reporting archive the required artifacts
