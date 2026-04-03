# 47. GPU Operator Preflight

This document defines the operator preflight path for real GPU sign-off hosts.

## Goal

Make a GPU host deterministic before running:

- single-GPU soak
- 2-GPU parity
- 4-GPU soak
- RC sign-off with required GPU evidence

## Preflight Command

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu_preflight.ps1 -Profile full -Output artifacts/gpu/preflight.json
```

Linux/macOS:

```sh
sh scripts/gpu_preflight.sh full configs/enkai_50m.enk artifacts/gpu/preflight.json artifacts/gpu
```

## What It Checks

- `enkai` binary is resolvable
- tensor native library is present
- base train config exists
- required GPU scripts are present
- `nvidia-smi` is available
- required GPU count exists for the chosen profile
- Python `torch` sees CUDA and the expected device count
- artifact directory exists and is writable

Profiles:

- `single`
- `multi`
- `soak4`
- `full`

## Expected Next Steps

The generated JSON report includes the exact next commands for the chosen profile.

For full sign-off the sequence is:

1. `scripts/soak_single_gpu.ps1`
2. `scripts/multi_gpu_harness.ps1`
3. `scripts/soak_4gpu.ps1`
4. `scripts/verify_gpu_gates.ps1 -LogDir artifacts/gpu`
5. `scripts/v3_0_0_rc_pipeline.ps1`

## Required Evidence

- `artifacts/gpu/single_gpu.log`
- `artifacts/gpu/single_gpu_evidence.json`
- `artifacts/gpu/multi_gpu.log`
- `artifacts/gpu/multi_gpu_evidence.json`
- `artifacts/gpu/soak_4gpu.log`
- `artifacts/gpu/soak_4gpu_evidence.json`

Without these artifacts, the final hardware sign-off is not complete.
