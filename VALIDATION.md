# v0.9.3 Validation Checklist

This file is the strict verification checklist for v0.9.3 completion.

## Current Gate State

- CPU single-device soak: PASS (functional evidence captured).
- CUDA single-GPU long soak: pending operator run.
- 2-GPU correctness gate: pending operator run.
- 4-GPU soak gate: pending operator run.

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
  - Uses `ENKAI_KILL_STEP` and `ENKAI_POST_RESUME_STEPS`.
- [x] Final PASS/FAIL summary printed:
  - Output fields include `status`, `last_step`, `last_loss`, `resumed_from_step`, `nan_or_inf`, `checkpoint_verified`.
- [x] Integrity checker added and wired:
  - `scripts/check_ckpt_integrity.ps1`
  - Called from `scripts/soak_single_gpu.ps1`.

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
  - Skips unless `ENKAI_RUN_MULTI_GPU_TESTS=1`.
- [x] Harness includes deterministic setup and checks:
  - fixed deterministic dataset generation
  - loss comparison against 1-GPU baseline (tolerance)
  - grad equality comparison post-allreduce artifacts (tolerance)
- [x] Harness prints `PASS`/`FAIL`/`SKIPPED`.

Operator run required:
- Command:
  - `$env:ENKAI_RUN_MULTI_GPU_TESTS=1; $env:ENKAI_SINGLE_GPU_GREEN=1; powershell -ExecutionPolicy Bypass -File scripts/multi_gpu_harness.ps1`
- Green criteria:
  - `PASS: 2-GPU DP correctness validated`

## 4) 4-GPU soak reliability

- [x] 4-GPU harness script added:
  - `scripts/soak_4gpu.ps1`
- [x] Script prints `PASS`/`FAIL`/`SKIPPED`.
- [x] Script outputs NCCL timeout guidance (`NCCL_ASYNC_ERROR_HANDLING`, `NCCL_TIMEOUT`).

Operator run required:
- Command:
  - `$env:ENKAI_RUN_MULTI_GPU_TESTS=1; $env:ENKAI_SINGLE_GPU_GREEN=1; powershell -ExecutionPolicy Bypass -File scripts/soak_4gpu.ps1`
- Green criteria:
  - `PASS: 4-GPU soak completed`
  - no hang
  - no NCCL timeout
  - no desync

## 5) Repository health checks

- [x] `cargo fmt`
- [x] `cargo clippy --workspace -- -D warnings`
- [x] `cargo test --workspace`

Expected result:
- all commands exit `0`.
