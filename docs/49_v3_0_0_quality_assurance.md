# 49. v3.0.0 Quality Assurance Findings

This document records the final CPU-side QA pass for the `v2.9.1 -> v3.0.0`
quality-recovery program and the remaining GPU/operator evidence blocker.

## Scope

The QA pass covers the complete CPU-side proof contract:

- correctness
- determinism
- performance baselines
- Adam-0 CPU reference validation
- runtime / FFI safety
- release evidence, capability reporting, and release dashboard publication

The QA pass does not claim final hardware sign-off. GPU/operator evidence remains
an external blocker until it is produced on a real GPU host.

## Commands Run

CPU-side gates:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
py -3 scripts/check_docs_consistency.py
powershell -ExecutionPolicy Bypass -File scripts/release_pipeline.ps1
```

GPU/operator readiness checks:

```powershell
py -3 scripts/gpu_preflight.py --profile full --output artifacts/gpu/preflight.json
cargo run -p enkai -- readiness verify-blockers --profile full_platform --report artifacts/readiness/full_platform.json --json --output artifacts/readiness/full_platform_blockers_gpu_required.json --require-gpu-evidence
```

## CPU-Side Findings

The CPU-side release gates are green when evaluated through
`scripts/release_pipeline.ps1`.

Observed result on `2026-04-04`:

- `scripts/release_pipeline.ps1`: passed
- `artifacts/release/v3.0.0/release_dashboard.json`: `cpu_complete = true`
- `artifacts/release/v3.0.0/release_dashboard.json`: `final_signoff_complete = false`

Archived proof sources for the final QA pass:

- `artifacts/release/v3.0.0/capability_complete.json`
- `artifacts/release/v3.0.0/release_dashboard.json`
- `artifacts/release/v3.0.0/release_dashboard.md`
- `artifacts/readiness/full_platform.json`
- `artifacts/readiness/full_platform_blockers.json`
- `artifacts/validation/ffi_correctness.json`
- `artifacts/validation/determinism_event_queue.json`
- `artifacts/validation/determinism_sim_replay.json`
- `artifacts/validation/determinism_adam0_reference_100.json`
- `artifacts/validation/pool_safety.json`
- `artifacts/validation/adam0_fake10.json`
- `artifacts/validation/adam0_ref100.json`
- `artifacts/validation/adam0_stress1000.json`
- `artifacts/validation/adam0_target10000.json`
- `artifacts/validation/perf_ffi_noop.json`
- `artifacts/validation/perf_sparse_dot.json`
- `artifacts/validation/perf_adam0_reference_100.json`

Quality conclusions:

1. Release claims are now proof-backed rather than artifact-presence-backed.
2. Determinism, CPU correctness, Adam-0 CPU validation, and runtime safety are
   all promoted into release-blocking proof groups.
3. The release dashboard makes the CPU-complete vs GPU-pending state explicit for
   operators.
4. The remaining blocker is not a code gap on this host; it is the absence of
   operator-run GPU evidence.

## GPU / Operator Findings

Expected blocker artifacts for final hardware sign-off:

- `artifacts/gpu/single_gpu_evidence.json`
- `artifacts/gpu/multi_gpu_evidence.json`
- `artifacts/gpu/soak_4gpu_evidence.json`

Expected blocker reports on a non-GPU host:

- `artifacts/gpu/preflight.json`
- `artifacts/readiness/full_platform_blockers_gpu_required.json`

Interpretation:

- if `scripts/gpu_preflight.py` reports missing `nvidia-smi`, insufficient GPU
  count, or missing CUDA-visible torch, the host is not eligible for final
  hardware sign-off
- if `enkai readiness verify-blockers --require-gpu-evidence` fails, the repo is
  still in the `CPU-complete / GPU sign-off pending` state

Observed result on `2026-04-04`:

- `artifacts/gpu/preflight.json` reported missing:
  - `nvidia-smi`
  - `gpu-count`
  - `torch-cuda`
- `artifacts/readiness/full_platform_blockers_gpu_required.json` failed only
  because these required GPU artifacts were absent:
  - `artifacts/gpu/single_gpu_evidence.json`
  - `artifacts/gpu/multi_gpu_evidence.json`
  - `artifacts/gpu/soak_4gpu_evidence.json`

## What Changed Across The Recovery Program

`v2.9.1`
- established proof-grade validation commands and archived validation artifacts

`v2.9.2`
- strengthened `std::sparse`, `std::event`, and `std::pool` correctness and
  equivalence requirements

`v2.9.3`
- made replay, snapshot, and coroutine behavior deterministic and auditable

`v2.9.4`
- completed the Adam-0 CPU validation ladder for `10`, `100`, `1000`, and
  `10000` agent scenarios

`v2.9.5`
- hardened FFI and runtime safety with release-blocking validation

`v2.9.6`
- moved release claims to proof-group summaries and added the release dashboard

`v3.0.0`
- finalized the publication assets, runbooks, and QA findings for the
  `CPU-complete / GPU sign-off pending` release state

## Release Status

Current state:

- code implementation: complete
- CPU-side proof gates: complete
- GPU/operator preflight tooling: complete
- final GPU hardware evidence: pending

Truthful release label:

- `v3.0.0` is CPU-complete and operator-ready for GPU sign-off
- final hardware sign-off still requires the real GPU evidence package
