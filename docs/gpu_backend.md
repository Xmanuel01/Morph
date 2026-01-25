# GPU Backend (v0.9)

Morph uses a native GPU backend (`morph_tensor`) built on libtorch + CUDA.

## Design

- Morph **never owns GPU memory** directly.
- Morph holds **opaque Int handles** to native tensors/devices.
- The native library owns memory and lifetime.
- Errors are surfaced via `morph_tensor_last_error()`.

## C ABI overview

The backend exposes a C ABI so Morph can call into it:

```
int64_t morph_tensor_device(const char* spec);
int64_t morph_tensor_randn(const char* shape_json, const char* dtype, int64_t device);
int64_t morph_tensor_matmul(int64_t a, int64_t b);
...
```

Return value:
- `0` means error (call `morph_tensor_last_error`).
- Non-zero is a valid handle.

## Ownership

Handles must be freed when no longer needed:

```
morph_tensor_free(handle);
morph_tensor_device_free(handle);
morph_tensor_opt_free(handle);
```

## Notes

- If the `torch` feature is disabled, the backend will return errors.
- CUDA is optional in CI; CPU-only is supported if libtorch is installed.

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
