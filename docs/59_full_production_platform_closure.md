# Full Production Platform Closure

This is the umbrella production claim for Enkai v4.0 platform work.

## Rule

The full production platform claim is valid only when every required lower-level closure surface is green. A single partial, blocked, stale, smoke-only, or hardware-missing proof blocks the umbrella claim.

The verifier is:

```text
scripts/verify_v4_0_0_full_production_platform_closure.py
```

The output artifact is:

```text
artifacts/readiness/v4_0_0_full_production_platform_closure.json
```

## Required Surfaces

The umbrella gate requires these artifacts to be present and green:

- `artifacts/readiness/v4_0_0_tensor_ffi_opaque_handles.json`
- `artifacts/readiness/v4_0_0_native_snn_batched_kernels.json`
- `artifacts/readiness/v4_0_0_spatial_rtree_native.json`
- `artifacts/readiness/v4_0_0_llm_package_registry_verify.json`
- `artifacts/readiness/v4_0_0_app_platform_closure.json`
- `artifacts/readiness/v4_0_0_live_app_platform_closure.json`
- `artifacts/readiness/v4_0_0_multi_node_orchestration_closure.json`

## Live Evidence Requirements

The live app platform proof must include real external MySQL, deployed gRPC, and signed/mobile deployment evidence. The multi-node orchestration proof must include real 2+ GPU execution evidence with distinct GPU UUIDs, rank logs, parity checks, and process exit codes.

## Current Interpretation

If this verifier fails, Enkai may still have bounded production-grade slices, but the broad full production platform claim is not closed. The correct status is blocked until the failed surface artifacts become green.

## Commands

```powershell
py scripts\verify_v4_0_0_full_production_platform_closure.py --workspace .
```

If the verifier fails, inspect the `failures` array in the output artifact and close those lower-level proofs first.
