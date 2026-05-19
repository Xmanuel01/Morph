# Full Production Platform Closure

This is the umbrella production gate for Enkai. It prevents broad platform claims unless every required bounded, live external, and hardware-backed surface is green.

## Rule

A broad "full production platform" claim is allowed only when all required surface artifacts pass and all required production claims are true.

The verifier is intentionally strict. It does not allow a partial surface to be overridden by documentation wording, scaffold tests, smoke tests, or synthetic evidence.

## Required Surfaces

The umbrella closure currently requires these surfaces:

- Tensor FFI opaque handles
- Native SNN batched kernels
- Native spatial R-tree indexing
- LLM package registry closure
- Bounded app platform closure
- Live app platform closure
- Multi-node orchestration closure

## Live / Hardware Requirements

Some surfaces cannot be completed on a normal local workstation:

- Live app platform requires external MySQL, deployed gRPC, and signed/mobile deployment evidence.
- Multi-node orchestration requires a real 2+ GPU host with fresh distributed evidence.

If those are missing, the correct status is blocked, not production-grade.

## Verification

Run:

```powershell
py scripts\verify_v4_0_0_full_production_platform_closure.py --workspace .
```

The output artifact is:

```text
artifacts/readiness/v4_0_0_full_production_platform_closure.json
```

The artifact includes:

- `blocked_surfaces`
- `open_blockers`
- `closure_policy`
- `production_claims.full_production_platform_proven`

## Current Interpretation

If `full_production_platform_proven` is false, do not claim the whole platform is complete. Use the individual green surface claims only.
