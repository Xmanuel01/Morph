# 37. Production Readiness Matrix (v2.3.0 target)

This matrix defines the objective sign-off contract for "production-ready" in the
`v2.2.1 -> v2.3.0` cycle.

Bounded production envelope:
- single-node deployment (`Docker` + `systemd`) for backend/fullstack scaffolds
- 1-2 GPU training/serving path
- local filesystem model registry
- VM runtime as normative execution contract

## Non-GPU Readiness Gate Bundle

Run:

`enkai readiness check --profile production --json --output artifacts/readiness/production.json`

Manifest:
- `enkai/contracts/readiness_production_v2_3_0.json`

The command executes a deterministic gate bundle:
- format/lint/test
- docs/spec consistency
- frontend/backend contract snapshot test
- bootstrap mainline + Stage0 fallback lanes
- benchmark fairness smoke (`official_v2_3_0_matrix`, workload-equivalence contract)

## GPU Evidence (Release Blocking)

`v2.3.0` release sign-off requires operator evidence and verifier pass:
- single-GPU stability evidence
- 2-GPU loss/grad parity evidence
- 4-GPU soak evidence

Verification:
- `scripts/verify_gpu_gates.ps1`
- `scripts/verify_gpu_gates.sh`

## Sign-Off Rule

A release is marked "production-ready" only when:
- non-GPU readiness bundle is green, and
- RC/release artifact gates are green, and
- GPU evidence package is present and verifier-clean.
