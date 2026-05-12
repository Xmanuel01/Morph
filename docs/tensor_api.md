# Tensor API (v2.3.0)

`std::tensor` has a first-party Enkai runtime path for deterministic CPU-safe
construction, shape inspection, flat access, NumPy-style broadcasting for
elementwise arithmetic, matrix multiply, reshape, transpose, slicing,
concatenation, reductions, selection/masking, gather/scatter, top-k/sort,
limited einsum, convolution, pooling, batchnorm, attention, deterministic
dropout, linear, embedding, layernorm, cross-entropy, and bounded tape-backed
autodiff. Larger accelerated kernels are still backed by the
`enkai_tensor` native library. This doc reflects the v2.3.0 production surface
plus v2.1.x hardening updates, safety improvements, and backend behavior.

## Quick Start

```enkai
import std::io
import std::json
import std::tensor

policy default ::
    allow io.write
::policy

fn main() ::
    let dev := tensor.device("cpu")
    let x := tensor.randn([2, 2], "f32", dev)
    let y := tensor.randn([2, 2], "f32", dev)
    let z := tensor.matmul(x, y)
    let _ := io.stdout_write_text(json.enkai(tensor.shape(z)) + "\n")
::fn
```

Use `cuda:0` only on machines with a CUDA-capable build and green preflight.
CPU examples are the safest learner path.

## Supported Ops (Selected)
- First-party bounded runtime path: `tensor.from_array`, `tensor.to_array`,
  `tensor.rank`, `tensor.len`, `tensor.get_flat`, `tensor.zeros`,
  `tensor.randn`, `tensor.shape`, `tensor.add`, `tensor.sub`, `tensor.mul`,
  `tensor.div`, `tensor.scale`, `tensor.broadcast_to`,
  `tensor.matmul`, `tensor.reshape`, `tensor.transpose`, `tensor.slice`,
  `tensor.concat`, `tensor.sum`, `tensor.mean`, `tensor.softmax`,
  `tensor.relu`, `tensor.sigmoid`, `tensor.gelu`, `tensor.exp`, `tensor.log`,
  `tensor.sqrt`, `tensor.tanh`, `tensor.dropout`, `tensor.linear`,
  `tensor.embedding`, `tensor.layernorm`, `tensor.cross_entropy`,
  `tensor.to_dtype`, `tensor.to_device`, `tensor.where`, `tensor.clip`,
  `tensor.argmax`, `tensor.sort`, `tensor.topk`, `tensor.gather`,
  `tensor.scatter`, `tensor.masked_fill`, `tensor.einsum`, `tensor.conv2d`,
  `tensor.max_pool2d`, `tensor.avg_pool2d`, `tensor.batchnorm1d`,
  `tensor.attention`.
- First-party autodiff path: `tensor.requires_grad`, `tensor.require_grad`,
  `tensor.backward`, `tensor.grad`, `tensor.zero_grad`, `tensor.detach`,
  `tensor.no_grad`, `tensor.grad_check`.

## Autodiff

The first-party runtime includes a deterministic tape for core differentiable
ops. Tracked tensors are created with `tensor.requires_grad(x)` or
`tensor.require_grad(x)`. Calling `tensor.backward(loss)` requires a scalar loss
and accumulates gradients into tracked leaf tensors. Use `tensor.grad(x)` to read
the accumulated gradient, `tensor.zero_grad(x)` to clear it, `tensor.detach(x)`
to cut graph history, and `tensor.no_grad(true|false)` to disable or re-enable
graph construction for a block of code.

Current first-party backward rules cover elementwise add/sub/mul/div, scale,
rank-2 matmul, reductions, softmax, linear, embedding, layernorm,
cross-entropy, conv2d, avg/max pool2d, attention, and core unary
activations/math functions. Unsupported operations remain deterministic forward
kernels but do not claim gradient coverage until a backward rule is added and
tested.
- `tensor.device("cpu"|"cuda:N")`
- `tensor.randn/zeros`
- `tensor.add/mul/matmul/softmax/masked_softmax/relu/sigmoid/gelu/dropout`
- `tensor.sum/mean/reshape/transpose/view/slice/concat`
- `tensor.to_device/to_dtype/shape`
- `tensor.layernorm/embedding/linear` plus `layernorm_backward`, `masked_softmax_backward`
- Autograd: `tensor.cross_entropy`, `tensor.backward`, `tensor.require_grad`, `tensor.requires_grad`, `tensor.grad`, `tensor.detach`, `tensor.no_grad`, `tensor.zero_grad`, `tensor.zero_grad_multi`, `tensor.grad_check`
- First-party optimizer helpers: `tensor.sgd_step`, `tensor.sgd_step_multi`, `tensor.clip_grad_norm`, `tensor.adamw_state`, `tensor.adamw_step`, `tensor.adamw_step_multi`
- First-party memory evidence helpers: `tensor.memory_current`, `tensor.memory_peak`, `tensor.memory_limit`, `tensor.memory_set_limit`, `tensor.memory_clear_limit`, `tensor.memory_reset_peak`
- Native compatibility optimizer helpers: `tensor.param_group`, `tensor.param_group_step`
- LM core FFI (native runtime integration):
  - `enkai_tensor_lm_init(spec_json, device_handle, seed, out_json, out_len)`
  - `enkai_tensor_forward_lm(params_json, spec_json, input, target, batch, seq, training)`
  - `enkai_tensor_tinylm_init/forward_tinylm` remain supported for compatibility fallback.

## Benchmark Evidence

The bounded CPU training runtime has a readiness benchmark/proof pair:

```powershell
py scripts/readiness_v3_8_0_tensor_training_bench.py --workspace . --enkai-bin target\debug\enkai.exe
py scripts/verify_v3_8_0_tensor_training_bench.py
```

The generated evidence is written to `artifacts/readiness/v3_8_0_tensor_training_bench.json` and the verifier writes `artifacts/readiness/v3_8_0_tensor_training_bench_verify.json`. The contract covers training/autodiff/optimizer execution, eval-only attention execution, peak memory evidence, and deterministic bounded-OOM behavior. This is a first-party CPU-runtime proof, not a GPU or PyTorch-parity benchmark.

## CUDA-first production LLM runtime foundation

v3.9.0 starts the CUDA-first production LLM runtime foundation. The first
contract freezes a bounded decoder-transformer train/eval/checkpoint benchmark
against PyTorch before Enkai widens claims to broader GPU, distributed, or
package-level compatibility.

- Contract: `enkai/contracts/v3_9_0_cuda_llm_runtime_foundation.json`
- Suite: `bench/suites/v3_9_0_cuda_llm_runtime_foundation.json`
- Readiness: `scripts/readiness_v3_9_0_cuda_llm_runtime_foundation.py`
- Verifier: `scripts/verify_v3_9_0_cuda_llm_runtime_foundation.py`

The backend catalog distinguishes `cpu`, `torch`, `cuda`, `rocm`, and
`metal`. `cuda` is the primary accelerated target for this line. ROCm and Metal
now have first-party source backends with build-gated kernel libraries:
`enkai_tensor/rocm/enkai_kernels.hip.cpp` for HIP/ROCm and
`enkai_tensor/metal/enkai_kernels.metal` for Apple Metal. The `rocm-kernels`
feature requires `hipcc` and ROCm runtime availability; the `metal-kernels`
feature requires macOS and the Xcode Metal toolchain. These backends remain
hardware-proof-gated until their hardware-backed verifier evidence and PyTorch
ROCm/MPS comparison suites are archived,
but they remain hardware-proof-gated. Full PyTorch parity, distributed training
closure, and broad GPU performance claims must not be claimed without green
hardware evidence.

The v3.9.0 GPU benchmark gate is intentionally strict. A CUDA performance claim
is blocked unless the frozen suite archives:

- Enkai CUDA train throughput at least `1.50x` the PyTorch CUDA baseline.
- Enkai CUDA eval throughput at least `1.50x` the PyTorch CUDA baseline.
- Enkai CUDA checkpoint write throughput at least `1.50x` the PyTorch CUDA baseline.
- Enkai CUDA peak memory no worse than the PyTorch CUDA baseline.
- Enkai CUDA checkpoint resume latency no worse than the PyTorch CUDA baseline.
- Bounded loss parity within the contract tolerance.

If any target misses, the verifier must fail. The next action is optimization on
the measured GPU profile, not a production claim.

The first-party CUDA kernel tranche adds guarded source-level kernels under
`enkai_tensor/cuda/enkai_kernels.cu` and the `cuda-kernels` Cargo feature. The
feature requires `nvcc`; builds without the feature keep the manifest available
but report `build_status = "not_compiled"`. Current first-party CUDA source
covers the bounded LLM-training kernel frontier: `fp32` elementwise
add/mul/scale, `fp16`/`bf16` elementwise variants, `bias_gelu`,
`matmul_bias`, `layernorm`, `softmax`, `masked_softmax`, fused
`cross_entropy` forward/backward, embedding forward/backward, causal attention
prefill, causal attention backward, KV-cache decode, fused `adamw_update`, and
gradient clipping. The manifest marks this source set as
`source_complete_hardware_gated`: production claims still require a green
CUDA/PyTorch verifier on real CUDA hardware. Remaining work before broad
PyTorch parity includes tiled Tensor Core kernels, deeper attention/kernel
fusion, multi-GPU CUDA/NCCL collectives, and archived tolerance/throughput proof
from a CUDA runner.

GPU proof after pushing a version should be run on the target hardware runner:

```powershell
scripts\run_v3_9_0_gpu_proof.ps1 -Python "py -3.11" -InstallPyTorchCuda -BuildFirstPartyCudaKernels
```

```bash
INSTALL_PYTORCH_CUDA=1 BUILD_FIRST_PARTY_CUDA_KERNELS=1 PYTHON=python3.11 scripts/run_v3_9_0_gpu_proof.sh
```

For a preflight-only check before a long GPU run:

```powershell
py -3.11 scripts\preflight_v3_9_0_gpu_test.py --workspace . --python "py -3.11" --require-nvcc
```

Distributed GPU production proof is a separate hard gate. It requires the
single-GPU CUDA proof to pass first, then runs a two-rank executable training
proof with `torch,dist`, archived rank logs, loss parity, gradient parity, and
rank artifact checks:

```powershell
scripts\run_v3_9_0_gpu_proof.ps1 -Python "py -3.11" -InstallPyTorchCuda -BuildFirstPartyCudaKernels -RunDistributedGpuProof
```

```bash
INSTALL_PYTORCH_CUDA=1 BUILD_FIRST_PARTY_CUDA_KERNELS=1 RUN_DISTRIBUTED_GPU_PROOF=1 PYTHON=python3.11 scripts/run_v3_9_0_gpu_proof.sh
```

Four-GPU production claims additionally require the soak gate:

```bash
INSTALL_PYTORCH_CUDA=1 BUILD_FIRST_PARTY_CUDA_KERNELS=1 RUN_DISTRIBUTED_GPU_PROOF=1 RUN_FOUR_GPU_SOAK=1 PYTHON=python3.11 scripts/run_v3_9_0_gpu_proof.sh
```

ROCm and Metal builds are intentionally separate hardware gates:

```bash
RUN_ROCM_SOURCE_BUILD=1 PYTHON=python3.11 scripts/run_v3_9_0_gpu_proof.sh
RUN_METAL_SOURCE_BUILD=1 PYTHON=python3.11 scripts/run_v3_9_0_gpu_proof.sh
```

The GPU memory planner has a deterministic v1 policy surface exposed through
`enkai_gpu_memory_planner_policy` and `enkai_gpu_memory_plan`. It estimates
parameter, optimizer, activation, workspace, KV-cache, and peak memory bytes for
the frozen transformer profile, applies 256-byte alignment, and returns
machine-parseable `E_GPU_PLAN_*` errors for invalid specs, unsupported dtypes,
and budget overflow. This is a production-shaped planning contract; release
claims still require CUDA hardware peak-memory measurement and fragmentation
stress evidence from the verifier.

Large-model checkpointing now has a v1 manifest/shard planning surface through
`enkai_large_checkpoint_format_policy` and `enkai_large_checkpoint_plan`. The
format contract covers model shards, optimizer shards, RNG state, data cursor,
SHA-256 hashes, atomic write expectations, and resume validation. The existing
checkpoint writer remains the execution path; full production closure requires
fault-injected corruption/replay evidence on the target filesystem.

Distributed training now has a bounded v1 planning surface through
`enkai_distributed_training_policy` and `enkai_distributed_training_plan`. The
contract freezes barrier/allreduce ownership, gradient bucket sizing,
multi-rank checkpoint merge/replay requirements, and deterministic fault cases
for rank disconnects, stale payloads, duplicate ranks, wrong tensor length, and
timeouts. Full distributed production status still requires real multi-rank
hardware evidence.

Mixed precision now has an explicit `enkai_amp_v1` policy manifest and guarded
loss-scaler validation. Invalid scale/growth/backoff/interval values produce
deterministic `E_AMP_*` errors. Production AMP still requires CUDA hardware
evidence, overflow fault injection, and PyTorch gradient-parity proof.

## First-party memory accounting

The bounded CPU tensor runtime tracks live tensor payload bytes and peak tensor
payload bytes using the runtime's internal `f64` storage size. The memory helpers
are deterministic and intended for readiness evidence, benchmark provenance, and
safety tests:

- `tensor.memory_current()` returns currently live first-party tensor payload bytes.
- `tensor.memory_peak()` returns the peak live first-party tensor payload bytes.
- `tensor.memory_reset_peak()` resets peak to current live payload bytes.
- `tensor.memory_set_limit(bytes)` installs a bounded allocation limit and returns the previous limit, or `-1` when no previous limit was set.
- `tensor.memory_clear_limit()` removes the limit and returns the previous limit, or `-1`.
- `tensor.memory_limit()` returns the active limit, or `-1`.

Creation paths such as `tensor.from_array`, `tensor.zeros`, and `tensor.randn`
fail deterministically when the active limit would be exceeded. This is a bounded
CPU-runtime OOM guard, not a GPU allocator or full lifetime planner.

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
- Distributed failures are machine-parseable with `E_DIST_*` prefixes
  (`E_DIST_ENV_GATE`, `E_DIST_FEATURE_MISSING`, `E_DIST_DEVICE_MAPPING`, etc.).
- Device-per-rank selection and explicit guardrails are covered by CUDA-gated tests
  (`backend_rank_device.rs`, `dist_guards.rs`).
- `enkai_dist_init` and `enkai_dist_allreduce_sum_multi` require
  `enkai_tensor` built with features `torch,dist`; otherwise they return explicit
  feature-missing errors with rebuild guidance (`E_DIST_FEATURE_MISSING`).

## Checkpointing
- Single-rank saves parameters/optimizer state with SHA-256 integrity files; load verifies hashes before returning handles.
- Ranked saves: `enkai_checkpoint_save_ranked(dir, rank, world_size, params, opt, meta)` writes `params_rank{n}.bin` and `meta_rank{n}.json` (and optimizer shard if provided) with per-file hashes. `enkai_checkpoint_load_ranked` loads the shard for the given rank. Manifest/barrier coordination remains manual (one call per rank).

## Known gaps / roadmap
- Multi-rank execution remains environment-gated. First-party harness wrappers now
  exist for Windows/Linux (`scripts/multi_gpu_harness.ps1/.sh`, `scripts/soak_4gpu.ps1/.sh`)
  and emit structured evidence under `artifacts/gpu`, but operator-run GPU soak
  evidence is still required for release sign-off.
- CPU backend still depends on libtorch; a pure CPU kernel set would remove that dependency.
- Mixed-precision support exists only for torch backends.
- Safety depends on callers honoring documented preconditions; invalid pointers or JSON can still cause undefined behavior.






## v4.0 Native Training Runtime Direction

The v4.0 native-training-runtime branch starts moving Enkai training execution
away from PyTorch as a core engine. PyTorch remains allowed only in reference
benchmark scripts and correctness comparison tests.

The first native tranche provides:
- `enkai_tensor::native_runtime::TensorGraph` and `GraphOp` as an Enkai-owned tensor graph IR.
- `CpuBackend` for native CPU execution of zeros, add, multiply, matmul, ReLU, softmax, cross-entropy, mean, and sum.
- `MemoryPlanner` metrics for peak bytes, allocated bytes, freed bytes, live bytes, and reuse count.
- Fusion helpers for add+ReLU, matmul+bias, and softmax+cross-entropy equivalence evidence.
- A basic MLP SGD training loop with manual backward rules for the bounded graph.
- `CudaBackendHook` as a non-PyTorch CUDA/Triton extension point. It is a hook only until CUDA execution has separate hardware-backed proof.

Performance language is gated by `artifacts/readiness/v4_0_native_training_runtime.json`.
Do not claim broad PyTorch or CUDA superiority from this tranche. The acceptable
wording after green evidence is:

`Enkai's native training runtime is up to X% faster than Python/PyTorch eager on selected benchmarked workloads.`
