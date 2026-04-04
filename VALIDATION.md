# Release Validation Matrix (v2.x)

This file is the strict verification matrix for the v2.x release line.

## Current Gate State

- CPU single-device soak: PASS (functional evidence captured).
- CPU proof-grade validation: in progress via `enkai validate ...` and `artifacts/validation/*.json`.
- CUDA single-GPU long soak: pending operator run.
- 2-GPU correctness gate: pending operator run.
- 4-GPU soak gate: pending operator run.

## CPU Validation Matrix

- `enkai validate ffi-correctness --json --output artifacts/validation/ffi_correctness.json`
- `enkai validate determinism --suite event_queue --runs 10 --json --output artifacts/validation/determinism_event_queue.json`
- `enkai validate determinism --suite sim_replay --runs 10 --json --output artifacts/validation/determinism_sim_replay.json`
- `enkai validate determinism --suite sim_coroutines --runs 10 --json --output artifacts/validation/determinism_sim_coroutines.json`
- `enkai validate determinism --suite adam0_reference_100 --runs 3 --json --output artifacts/validation/determinism_adam0_reference_100.json`
- `enkai validate pool-safety --json --output artifacts/validation/pool_safety.json`
- `enkai validate adam0-cpu --scenario fake10 --json --output artifacts/validation/adam0_fake10.json`
- `enkai validate adam0-cpu --scenario ref100 --json --output artifacts/validation/adam0_ref100.json`
- `enkai validate adam0-cpu --scenario stress1000 --json --output artifacts/validation/adam0_stress1000.json`
- `enkai validate adam0-cpu --scenario target10000 --json --output artifacts/validation/adam0_target10000.json`
- `enkai validate perf-baseline --suite ffi_noop --json --output artifacts/validation/perf_ffi_noop.json`
- `enkai validate perf-baseline --suite sparse_dot --json --output artifacts/validation/perf_sparse_dot.json`
- `enkai validate perf-baseline --suite adam0_reference_100 --json --output artifacts/validation/perf_adam0_reference_100.json`

These artifacts are required inputs to full-platform readiness and strict release evidence on the quality-recovery line.

Proof requirements on the `v2.9.2+` line:
- `ffi_correctness.json` must prove:
  - exact sparse/event/pool outputs
  - native-backed and VM-fallback results are identical
  - the native path was actually exercised
  - native handle live count returns to zero
- `determinism_event_queue.json` must prove:
  - byte-identical output hashes across 10 runs
  - insertion-order tie-breaking remains stable
  - native-backed and VM-fallback hashes/results are identical
  - the native event queue path was actually exercised
- `determinism_sim_replay.json` and `determinism_sim_coroutines.json` must prove:
  - repeated seed-based outputs are byte-identical
  - snapshot and replay hashes match when replay artifacts are present
  - coroutine/task counters are recorded in the audit payload
  - coroutine suites actually exercised spawn/emit/next-wait counters
- `pool_safety.json` must prove:
  - reusable pool stats plateau after warmup
  - reset clears available items without corrupting counters
  - native-backed and VM-fallback results are identical
  - the native pool path was actually exercised

Baseline and proof metadata:
- `enkai/contracts/validation_cpu_v3_0_0.json`
- `enkai/contracts/validation_release_blockers_v3_0_0.json`
- `bench/baselines/validation_cpu_v3_0_0.json`
- `bench/machines/windows_local.json`
- `bench/machines/linux_local.json`

## 0) v2.0 contract enforcement (config + checkpoints + CLI)

- [x] Train/Eval config schema v1 enforced:
  - `config_version: 1` required for all runtime-accepted configs.
  - legacy configs are rejected by default in v2.1.0 strict acceptance path.
  - `backend`, `dtype`, and `device` validation errors are explicit.
- [x] Checkpoint format v1 enforced:
  - `format_version: 1` included in metadata.
  - `model_sig`, `dtype`, `device`, `config_hash` in metadata.
  - legacy checkpoints (missing required v1 metadata) are rejected by default in v2.1.0.
- [x] `enkai --version` reports CLI + language version.

## 1) Real Transformer forward + cross-entropy replaces `forward_l2`

- [x] Runtime backend calls TinyLM CE forward:
  - `enkairt/src/backend.rs:168` calls `forward_tinylm`.
  - `enkairt/src/backend.rs:504` binds `enkai_tensor_forward_tinylm`.
  - `enkai_tensor/src/lib.rs:2500` defines `enkai_tensor_forward_tinylm`.
- [x] `forward_l2` is not used by `enkai train` / engine path:
  - `rg -n "forward_l2" enkai/src/train.rs enkairt/src/engine.rs enkairt/src/backend.rs`
  - Expected: no references in `enkai/src/train.rs` and `enkairt/src/engine.rs`.
- [x] CE sanity run script added:
  - `scripts/ce_loss_sanity.ps1`
  - Config: `configs/ce_sanity_300.enk`
  - Command:
    - `powershell -ExecutionPolicy Bypass -File scripts/ce_loss_sanity.ps1`
  - Expected output contains:
    - `PASS: CE loss shows expected non-divergent trend`

## 2) Single-GPU stability with checkpoint/resume

- [x] Soak script with automated kill+resume:
  - `scripts/soak_single_gpu.ps1`
  - `scripts/soak_single_gpu.sh`
  - Uses `ENKAI_KILL_STEP` and `ENKAI_POST_RESUME_STEPS`.
- [x] Final PASS/FAIL summary printed:
  - Output fields include `status`, `last_step`, `last_loss`, `resumed_from_step`, `nan_or_inf`, `checkpoint_verified`.
- [x] Integrity checker added and wired:
  - `scripts/check_ckpt_integrity.ps1`
  - Called from `scripts/soak_single_gpu.ps1`.
- [x] Structured evidence output emitted:
  - `artifacts/gpu/single_gpu.log`
  - `artifacts/gpu/single_gpu_evidence.json`

Operator run required:
- Command:
  - `powershell -ExecutionPolicy Bypass -File scripts/soak_single_gpu.ps1`
- Green criteria:
  - `status: PASS`
  - `nan_or_inf: False`
  - `checkpoint_verified: True`
  - Resume step is less than final step and monotonic.

## 3) 2-GPU data parallel correctness

- [x] Harness script exists and is gated:
  - `scripts/multi_gpu_harness.ps1`
  - `scripts/multi_gpu_harness.sh`
  - Skips unless `ENKAI_RUN_MULTI_GPU_TESTS=1`.
- [x] Harness uses first-party launcher flow on both Windows/Linux (no required ad-hoc launcher env command).
- [x] Harness includes deterministic setup and checks:
  - fixed deterministic dataset generation
  - loss comparison against 1-GPU baseline (tolerance)
  - grad equality comparison post-allreduce artifacts (tolerance)
- [x] Harness prints `PASS`/`FAIL`/`SKIPPED`.
- [x] Structured evidence output emitted:
  - `artifacts/gpu/multi_gpu.log`
  - `artifacts/gpu/multi_gpu_evidence.json`
  - parity artifacts (`baseline.jsonl`, `rank0.jsonl`, `rank1.jsonl`, `rank0_grads.json`, `rank1_grads.json`)

Operator run required:
- Command:
  - `$env:ENKAI_ENABLE_DIST=1; $env:ENKAI_RUN_MULTI_GPU_TESTS=1; $env:ENKAI_SINGLE_GPU_GREEN=1; powershell -ExecutionPolicy Bypass -File scripts/multi_gpu_harness.ps1`
  - `ENKAI_ENABLE_DIST=1 ENKAI_RUN_MULTI_GPU_TESTS=1 ENKAI_SINGLE_GPU_GREEN=1 sh scripts/multi_gpu_harness.sh`
- Green criteria:
  - `PASS: 2-GPU DP correctness validated`

## 4) 4-GPU soak reliability

- [x] 4-GPU harness script added:
  - `scripts/soak_4gpu.ps1`
  - `scripts/soak_4gpu.sh`
- [x] Script prints `PASS`/`FAIL`/`SKIPPED`.
- [x] Script supports first-party launcher flow and optional custom launcher override (`ENKAI_4GPU_LAUNCH_CMD`).
- [x] Structured evidence output emitted:
  - `artifacts/gpu/soak_4gpu.log`
  - `artifacts/gpu/soak_4gpu_evidence.json`

Operator run required:
- Command:
  - `$env:ENKAI_ENABLE_DIST=1; $env:ENKAI_RUN_MULTI_GPU_TESTS=1; $env:ENKAI_SINGLE_GPU_GREEN=1; powershell -ExecutionPolicy Bypass -File scripts/soak_4gpu.ps1`
  - `ENKAI_ENABLE_DIST=1 ENKAI_RUN_MULTI_GPU_TESTS=1 ENKAI_SINGLE_GPU_GREEN=1 sh scripts/soak_4gpu.sh`
- Green criteria:
  - `PASS: 4-GPU soak completed`
  - no hang
  - no NCCL timeout
  - no desync

## 5) Repository health checks

- [x] `cargo fmt`
- [x] `cargo clippy --workspace -- -D warnings`
- [x] `cargo test --workspace`
- [x] Backend protocol/runtime regression checks included in workspace tests:
  - WebSocket upgrade + frame send path (`http_websocket_upgrade_and_send_text`)
  - Postgres connector failure-mode contract (`std_db_postgres_open_failure_is_none`)

Expected result:
- all commands exit `0`.

## 6) v1.9.x compatibility + self-host gates

- [x] Legacy config compatibility test:
  - `legacy_config_without_config_version_still_trains`
- [x] Legacy checkpoint compatibility test:
  - `legacy_checkpoint_meta_without_format_version_loads`
- [x] Self-host CI corpus gate:
  - `enkai litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus`
- [x] Self-host mainline CI gate:
  - `enkai litec mainline-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost`
  - expected triage artifacts:
    - `artifacts/selfhost/litec_selfhost_ci_report.json`
    - `artifacts/selfhost/litec_replace_check_report.json`
    - `artifacts/selfhost/litec_mainline_ci_report.json`
- [x] Self-host consolidated release gate:
  - `enkai litec release-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost`
  - expected triage artifact:
    - `artifacts/selfhost/litec_release_ci_report.json`
- [x] Self-host replacement-readiness gate:
  - `enkai litec replace-check enkai/tools/bootstrap/selfhost_corpus --no-compare-stage0`
- [x] Production readiness report gate (non-GPU bundle):
  - `enkai readiness check --profile production --json --output artifacts/readiness/production.json`
- [x] Master pipeline smoke:
  - `master_pipeline_cpu_smoke`
- [x] Consolidated pipeline scripts:
  - `powershell -ExecutionPolicy Bypass -File scripts/release_pipeline.ps1`
  - `sh scripts/release_pipeline.sh`
- [x] RC pipeline scripts (GPU evidence mandatory by default):
  - `powershell -ExecutionPolicy Bypass -File scripts/rc_pipeline.ps1`
  - `sh scripts/rc_pipeline.sh`
- [x] RC evidence archive tooling:
  - `python3 scripts/collect_release_evidence.py --gpu-log-dir artifacts/gpu --require-gpu --strict`
- [x] Capability-complete report tooling:
  - `python3 scripts/generate_capability_report.py --require-gpu --strict`
- [ ] Optional GPU evidence verification (after operator runs):
  - `powershell -ExecutionPolicy Bypass -File scripts/verify_gpu_gates.ps1 -LogDir artifacts/gpu`
  - `sh scripts/verify_gpu_gates.sh artifacts/gpu`

## 7) Benchmark Target Gates (v2.3 target suite)

- [x] Official bounded benchmark suite updated:
  - `bench/suites/official_v2_3_0_matrix.json`
  - `bench/suites/official_v2_3_0_vm_compute.json`
  - `bench/suites/official_v2_3_0_native_bridge.json`
  - `bench/suites/official_v2_3_0_cli_workflows.json`
  - `bench/suites/official_v2_3_0_ai_data_workflows.json`
- [x] Benchmark fairness contract frozen:
  - `bench/contracts/workload_equivalence_v1.json`
- [x] Machine profiles pinned to official suite:
  - `bench/machines/linux_ref.json`
  - `bench/machines/windows_ref.json`
- [x] Benchmark target enforcement supports class-based policy:
  - `--enforce-target` validates global summary targets
  - `--enforce-class-targets --class-targets bench/suites/official_v2_3_0_targets.json` validates class medians + per-case targets
  - `--fairness-check-only` validates workload-equivalence metadata without executing workloads
- [x] CI regression blocker lane for benchmark targets:
  - `benchmark-target-gate` (Linux + Windows release binaries)

Suggested operator rerun command:
- `enkai bench run --suite official_v2_3_0_matrix --baseline python --iterations 2 --warmup 1 --machine-profile bench/machines/windows_ref.json --output bench/results/official_v2_3_0_matrix.windows.json --target-speedup 15 --target-memory 5 --enforce-target --enforce-class-targets --class-targets bench/suites/official_v2_3_0_targets.json`
