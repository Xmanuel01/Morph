# Single-GPU Stability Guide

This guide defines the single-GPU stability gate for Enkai v1.2.0.

## Goal

Run a 50M model for 8-12 hours on a single RTX 4090 with:

- checkpoint every 500 steps
- mid-run kill + resume
- no NaNs/Inf
- no checkpoint corruption

Multi-GPU work is blocked until this is stable.

## Quickstart (Windows)

```powershell
$env:ENKAI_CONFIG="configs/enkai_50m.enk"
$env:ENKAI_KILL_STEP=2000
$env:ENKAI_POST_RESUME_STEPS=500
scripts\soak_single_gpu.ps1
```

## Quickstart (Linux/macOS)

```bash
ENKAI_CONFIG=configs/enkai_50m.enk ENKAI_KILL_STEP=2000 ENKAI_POST_RESUME_STEPS=500 ./scripts/soak_single_gpu.sh
```

## Expected Output

- Training logs to `checkpoints/enkai_50m/train_log.jsonl`.
- Script kills the process at the configured step and restarts.
- Script confirms resume and validates log health (finite loss, monotonic steps).
- Final summary prints PASS/FAIL, last step/loss, resume step, and checkpoint verification.

## Failure Modes

- `non-finite loss detected`: NaN/Inf in loss or gradients. The run is aborted and an error checkpoint is attempted.
- `checkpoint config hash mismatch`: Config changed between runs.
- `checkpoint model signature mismatch`: Model shape changed between runs.
- `checkpoint dtype mismatch` / `device mismatch`: Resume attempted on different dtype or device.
- `timeout waiting for step`: Training did not progress (GPU/IO stall or deadlock).

## Gate Flags

- Set `ENKAI_SINGLE_GPU_GREEN=1` only after a successful CUDA single-GPU soak.
- Multi-GPU harnesses are additionally gated by `ENKAI_RUN_MULTI_GPU_TESTS=1`.

## Notes

- `configs/enkai_50m.enk` is the default template. You can override values by editing the file.
- Checkpoints are written atomically with integrity hashes.
- Resume uses the latest complete step checkpoint (`checkpoints/enkai_50m/step_*`).
- Multi-GPU harness is scaffolded but disabled by default. Enable via `ENKAI_RUN_MULTI_GPU_TESTS=1` only after single-GPU soak is green.
