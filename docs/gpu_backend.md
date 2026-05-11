# GPU Backend

Enkai's GPU work is centered on `enkai_tensor`, the native tensor backend and
first-party accelerator kernel source tree. CUDA is the first production proof
target. ROCm and Metal have source/build surfaces but require separate hardware
evidence before production claims.

Read this document as an operator guide, not as proof that a given GPU claim is
closed. A claim is closed only when its verifier artifact is green under
`artifacts/readiness/`.

## Design

- Enkai **never owns GPU memory** directly.
- Enkai holds **opaque Int handles** to native tensors/devices.
- The native library owns memory and lifetime.
- Errors are surfaced via `enkai_tensor_last_error()` (legacy `enkai_tensor_last_error()` is accepted).
- CUDA source kernels are build-gated through the `cuda-kernels` feature.
- Hardware benchmark claims are gated against PyTorch CUDA reference runs.

## C ABI overview

The backend exposes a C ABI so Enkai can call into it:

```
int64_t enkai_tensor_device(const char* spec);
int64_t enkai_tensor_randn(const char* shape_json, const char* dtype, int64_t device);
int64_t enkai_tensor_matmul(int64_t a, int64_t b);
...
```

Return value:
- `0` means error (call `enkai_tensor_last_error`).
- Non-zero is a valid handle.

## Ownership

Handles must be freed when no longer needed:

```
enkai_tensor_free(handle);
enkai_tensor_device_free(handle);
enkai_tensor_opt_free(handle);
```

## Notes

- If the `torch` feature is disabled, the backend will return errors.
- CUDA is optional in CI; CPU-only is supported if libtorch is installed.
- Builds without CUDA hardware can validate manifests and CPU behavior, but they
  cannot close CUDA production performance claims.

## Local verification

CPU (default, pinned to torch 2.2.0 for tch 0.15.x):

```bash
scripts/verify_torch.sh
```

CUDA (example for cu121 wheels):

```bash
TORCH_VERSION=2.2.0+cu121 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 scripts/verify_torch.sh
```

Windows PowerShell:

```powershell
scripts\\verify_torch.ps1
```

You can pass a CUDA wheel index on Windows as well:

```powershell
scripts\\verify_torch.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu121 -TorchVersion 2.2.0+cu121
```

Note: the scripts add the PyTorch `lib` directory to the process `PATH`/`LD_LIBRARY_PATH`
so `torch-sys` can locate the required DLLs/shared libs on Windows and Linux.

## CUDA troubleshooting

If CUDA validation fails:

- Run `python -c "import torch; print(torch.cuda.is_available())"` to confirm CUDA is available.
- Ensure your NVIDIA driver matches the CUDA wheel version (e.g. cu121).
- On Windows, verify `torch\\lib\\cudart64_*.dll` exists and that the script prepends it to `PATH`.
- If you see `STATUS_DLL_NOT_FOUND`, run `scripts\\verify_torch.ps1` from a new PowerShell session.

## Operator preflight

Before any real sign-off soak/parity run, execute the v3.9.0 GPU preflight:

```powershell
py -3.11 scripts\preflight_v3_9_0_gpu_test.py --workspace . --python "py -3.11" --require-nvcc
```

This validates:

- `enkai` binary resolution
- tensor native library presence
- train config presence
- required soak/parity scripts
- `nvidia-smi`
- PyTorch CUDA visibility
- required GPU count for the chosen profile

Run the full CUDA/PyTorch proof on the proof machine:

```powershell
scripts\run_v3_9_0_gpu_proof.ps1 -Python "py -3.11" -InstallPyTorchCuda -BuildFirstPartyCudaKernels
```

Distributed claims require the distributed proof flags and archived multi-rank
evidence. Do not mark distributed training production-closed from a single-GPU
run.
