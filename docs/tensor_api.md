# Tensor API (v1.9.4)

`std::tensor` is backed by the `enkai_tensor` native library. This doc reflects the
v1.9.4 surface, safety improvements, and backend behavior.

## Quick start
```
import std::tensor

fn main() ::
    let dev := tensor.device("cuda:0")
    let x := tensor.randn([2,2], "fp16", dev)
    let y := tensor.randn([2,2], "fp16", dev)
    let z := tensor.matmul(x, y)
    print(tensor.shape(z))
::
```

## Supported ops (selected)
- `tensor.device("cpu"|"cuda:N")`
- `tensor.randn/zeros`
- `tensor.add/mul/matmul/softmax/masked_softmax/relu/sigmoid/gelu/dropout`
- `tensor.sum/mean/reshape/transpose/view/slice/concat`
- `tensor.to_device/to_dtype/shape`
- `tensor.layernorm/embedding/linear` plus `layernorm_backward`, `masked_softmax_backward`
- Autograd: `tensor.cross_entropy`, `tensor.backward`, `tensor.require_grad`, `tensor.zero_grad`, `tensor.zero_grad_multi`
- Optimizer helpers: `tensor.adamw_step`, `tensor.adamw_step_multi`, `tensor.param_group`, `tensor.param_group_step`

## Backend selection (torch + CPU fallback)
- `enkai_backend_list()` -> JSON array, currently `["torch","cpu"]`.
- `enkai_backend_set("torch"|"cpu")` selects the active backend. **All extern ops are guarded**; a mismatched backend returns a sentinel and sets `enkai_tensor_last_error`.
- `enkai_backend_current()` -> active backend name.
- CPU backend currently routes through torch in CPU mode. It lets non-CUDA hosts run and provides graceful fallback, but it still depends on libtorch.

## Handle safety, ref-counts, and leak counters
- Tensors, devices, optimizers, and grad-scalers are ref-counted. `*_retain` increments; `*_free` decrements and detects double-free or stale handles.
- Live counters: `enkai_tensor_live_tensors/devices/opts/scalers()` expose current counts for leak smoke tests.
- Generation IDs prevent reuse of freed handles.

## Panic guard across FFI
- Every `extern "C"` entry is wrapped in a panic guard that catches Rust unwinds, writes the message to thread-local `enkai_tensor_last_error`, and returns a safe sentinel (`0`, `1`, `null_slice`, or null pointer). No Rust panic will cross the C ABI.
- Callers must still pass valid pointers/JSON; the guard does not make invalid inputs safe.

## Autocast and gradient scaling (AMP)
- GradScaler handles: `enkai_amp_scaler_create`, `enkai_amp_scaler_retain`, `enkai_amp_scaler_free`.
- Autocast scope: `enkai_autocast_enter(device)` / `enkai_autocast_exit()`.
- Helpers: `enkai_amp_scale_loss`, `enkai_amp_unscale_grads`, `enkai_amp_scaler_update`.
- Convenience: `enkai_amp_step(params, grads, scaler, loss, step_fn)` performs scale -> backward -> unscale -> update for torch backends.

## Distributed support (environment-gated)
- `enkai_dist_config(world_size, rank, device, seed)`:
  - validates `world_size/rank`,
  - requires explicit opt-in (`ENKAI_ENABLE_DIST=1`) for `world_size > 1`,
  - enforces rank-device mapping (`rank N -> cuda:N`) in multi-rank mode,
  - seeds per-rank execution deterministically (`seed + rank`).
- Multi-rank mode requires explicit opt-in: set `ENKAI_ENABLE_DIST=1`.
- `enkai_dist_allreduce_sum(handles)`:
  - validates distributed context and tensor handles,
  - enforces CUDA-device affinity per rank,
  - runs sum-allreduce and normalizes by `world_size`.
- Device-per-rank selection and explicit guardrails are covered by CUDA-gated tests
  (`backend_rank_device.rs`, `dist_guards.rs`).
- `enkai_dist_init` and `enkai_dist_allreduce_sum_multi` require
  `enkai_tensor` built with features `torch,dist`; otherwise they return explicit
  feature-missing errors.

## Checkpointing
- Single-rank saves parameters/optimizer state with SHA-256 integrity files; load verifies hashes before returning handles.
- Ranked saves: `enkai_checkpoint_save_ranked(dir, rank, world_size, params, opt, meta)` writes `params_rank{n}.bin` and `meta_rank{n}.json` (and optimizer shard if provided) with per-file hashes. `enkai_checkpoint_load_ranked` loads the shard for the given rank. Manifest/barrier coordination remains manual (one call per rank).

## Known gaps / roadmap
- Multi-rank execution remains environment-gated and launcher-dependent; operators
  must provide rank orchestration and GPU soak evidence for release sign-off.
- CPU backend still depends on libtorch; a pure CPU kernel set would remove that dependency.
- Mixed-precision support exists only for torch backends.
- Safety depends on callers honoring documented preconditions; invalid pointers or JSON can still cause undefined behavior.
