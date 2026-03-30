# 37. Production Readiness Matrix (v2.5.0 full-platform line)

This matrix defines the objective sign-off contract for "production-ready" in the
`v2.5.0 -> v2.9.0` cycle.

Full-platform production envelope:
- single-node and multi-node training/serving validation paths
- web + mobile frontend contract targets
- multi-engine DB core gates
- bootstrap mainline default with Stage0 emergency fallback lane
- signed remote registry + local cache lifecycle contract
- VM runtime as normative execution contract

## Non-Hardware Readiness Gate Bundle

Run:

`enkai readiness check --profile full_platform --json --output artifacts/readiness/full_platform.json`

Selective pipeline reuse:
- `enkai readiness check ... --skip-check <id>` may omit checks already executed by a stronger release gate.
- Release pipelines use this to skip standalone self-host lanes when `enkai litec release-ci` is run separately.

Manifest:
- `enkai/contracts/readiness_full_platform_v2_5_0.json`
- release blocker matrix:
  - `enkai/contracts/full_platform_release_blockers_v2_5_0.json`

The command executes a deterministic gate bundle:
- format/lint/test
- docs/spec consistency
- frontend/backend contract snapshot tests
- backend HTTP contract smoke
- LLM runtime smoke
- DB core smoke
- bootstrap mainline + Stage0 fallback lanes
- benchmark fairness + target smoke enforcement (`official_v2_3_0_matrix`, workload-equivalence contract)

## GPU Evidence (Release Blocking)

`v2.9.0` release sign-off requires operator evidence and verifier pass:
- single-GPU stability evidence
- 2-GPU loss/grad parity evidence
- 4-GPU soak evidence

Verification:
- `scripts/verify_gpu_gates.ps1`
- `scripts/verify_gpu_gates.sh`

## Sign-Off Rule

A release is marked "production-ready" only when:
- full-platform non-hardware readiness bundle is green, and
- RC/release artifact gates are green, and
- GPU evidence package is present and verifier-clean.
