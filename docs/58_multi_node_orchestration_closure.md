# Multi-Node Orchestration Hardware Closure

This document defines the production gate for Enkai multi-node / distributed GPU orchestration.

## Production Rule

Multi-node orchestration is not closed by scaffold tests, single-GPU tests, simulated cluster runs, or contract-only evidence. It closes only when a real hardware run produces green 2+ GPU evidence.

The required artifact is:

```text
artifacts/gpu/multi_gpu_evidence.json
```

The closure verifier is:

```text
scripts/verify_v4_0_0_multi_node_orchestration_closure.py
```

The verifier rejects:

- missing evidence
- blocked or skipped evidence
- stale evidence older than the contract window
- evidence without at least two live CUDA devices
- evidence without distinct GPU UUIDs
- evidence whose GPU UUIDs do not match the live host
- missing rank logs, gradient files, or baseline logs
- non-zero rank process exits
- failed loss or gradient parity checks

## Required Hardware Run

On a machine with at least two CUDA GPUs:

```bash
export ENKAI_ENABLE_DIST=1
export ENKAI_RUN_MULTI_GPU_TESTS=1
export ENKAI_SINGLE_GPU_GREEN=1
python scripts/gpu_harness.py multi
python scripts/readiness_v3_9_0_distributed_gpu_execution.py --workspace .
python scripts/verify_v4_0_0_multi_node_orchestration_closure.py --workspace .
```

On Windows PowerShell:

```powershell
$env:ENKAI_ENABLE_DIST = "1"
$env:ENKAI_RUN_MULTI_GPU_TESTS = "1"
$env:ENKAI_SINGLE_GPU_GREEN = "1"
py scripts\gpu_harness.py multi
py scripts\readiness_v3_9_0_distributed_gpu_execution.py --workspace .
py scripts\verify_v4_0_0_multi_node_orchestration_closure.py --workspace .
```

## Evidence Requirements

The evidence must include:

- `status: PASS`
- `world_size: 2`
- two distinct GPU UUIDs from the live host
- rank 0 and rank 1 reports
- `loss_parity: true`
- `grad_parity: true`
- archived stdout/stderr for baseline and both ranks
- archived baseline, rank logs, and gradient summaries
- rank process exit codes equal to zero

## Claim Boundary

A green verifier closes the bounded 2-rank distributed GPU orchestration claim. A 4-GPU production claim still requires the separate 4-GPU soak artifact.

If the current host has fewer than two GPUs, the correct status is blocked, not production-grade.
