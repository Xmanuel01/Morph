#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(command: list[str], cwd: Path, timeout: int = 900) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return {
        "command": command,
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "stdout_tail": proc.stdout[-12000:],
        "stderr_tail": proc.stderr[-12000:],
    }


def parse_evidence(stdout: str) -> dict[str, Any]:
    evidence = {"missing": True}
    for line in stdout.splitlines():
        if line.startswith("ENKAI_NATIVE_CUDA_EVIDENCE="):
            large = evidence.get("large_matmul")
            evidence = json.loads(line.split("=", 1)[1])
            if large is not None:
                evidence["large_matmul"] = large
        if line.startswith("ENKAI_NATIVE_CUDA_LARGE_MATMUL="):
            if evidence.get("missing"):
                evidence = {}
            evidence["large_matmul"] = json.loads(line.split("=", 1)[1])
        if line.startswith("ENKAI_NATIVE_CUDA_RESIDENT_MATMUL="):
            if evidence.get("missing"):
                evidence = {}
            evidence["resident_matmul"] = json.loads(line.split("=", 1)[1])
    return evidence


def torch_cuda_reference() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "reason": f"torch import failed: {exc}"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch CUDA unavailable", "version": getattr(torch, "__version__", "unknown")}
    def matmul_case(n: int, iters: int) -> dict[str, Any]:
        a = torch.arange(n * n, dtype=torch.float32, device="cuda").reshape(n, n).remainder(17).sub(8).mul(0.01)
        b = torch.arange(n * n, dtype=torch.float32, device="cuda").reshape(n, n).remainder(19).sub(9).mul(0.01)
        out = None
        for _ in range(3):
            out = a @ b
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            out = a @ b
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        checksum = float(out.sum().detach().cpu()) if out is not None else 0.0
        return {
            "shape": [n, n],
            "iterations": iters,
            "elapsed_ms": elapsed_ms,
            "per_iter_ms": elapsed_ms / iters,
            "checksum": checksum,
            "timing_scope": "resident_loop_with_device_sync_no_host_copy",
        }
    return {
        "available": True,
        "version": getattr(torch, "__version__", "unknown"),
        "device": torch.cuda.get_device_name(0),
        "matmul_32x32": matmul_case(32, 10),
        "matmul_512x512": matmul_case(512, 20),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_native_cuda_runtime.json")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    root = Path(args.workspace).resolve()

    nvcc = shutil.which("nvcc")
    nvidia_smi = shutil.which("nvidia-smi")
    failures: list[str] = []
    cargo = None
    evidence: dict[str, Any] = {"skipped": True, "reason": "cuda hardware/toolchain not requested"}

    if nvcc or args.require_cuda:
        cargo = run(
            [
                "cargo",
                "test",
                "-p",
                "enkai_tensor",
                "--release",
                "--features",
                "cuda-kernels",
                "--test",
                "native_cuda_runtime",
                "--",
                "--nocapture",
            ],
            root,
            timeout=900,
        )
        evidence = parse_evidence(cargo.get("stdout_tail", ""))
        if not cargo["passed"]:
            failures.append("native CUDA runtime cargo test failed")
        if evidence.get("missing"):
            failures.append("native CUDA evidence line missing")
        if args.require_cuda and evidence.get("skipped"):
            failures.append(f"native CUDA proof skipped: {evidence.get('reason', 'unknown')}")
        if not evidence.get("skipped", True):
            if evidence.get("pytorch_core_execution_dependency") is not False:
                failures.append("native CUDA runtime has a PyTorch core execution dependency")
            if evidence.get("cuda_memory", {}).get("peak_bytes", 0) <= 0:
                failures.append("native CUDA memory evidence missing")
            if "matmul_cublas" not in evidence.get("ops", []):
                failures.append("native CUDA cuBLAS matmul proof missing")
            if not evidence.get("large_matmul"):
                failures.append("native CUDA large matmul benchmark missing")
            if not evidence.get("resident_matmul"):
                failures.append("native CUDA resident matmul benchmark missing")
    elif args.require_cuda:
        failures.append("nvcc is required but was not found")

    torch_ref = torch_cuda_reference() if (nvcc or args.require_cuda) else {"available": False, "reason": "not requested"}
    if args.require_cuda and not torch_ref.get("available"):
        failures.append("PyTorch CUDA reference is required for comparison but unavailable")
    if evidence.get("large_matmul") and torch_ref.get("available"):
        torch_large = torch_ref.get("matmul_512x512", {})
        native_large = evidence["large_matmul"]
        if torch_large.get("elapsed_ms", 0) > 0 and native_large.get("elapsed_ms", 0) > 0:
            native_large["pytorch_eager_ms"] = torch_large["elapsed_ms"]
            native_large["speedup_vs_pytorch_eager"] = torch_large["elapsed_ms"] / native_large["elapsed_ms"]
    if evidence.get("resident_matmul") and torch_ref.get("available"):
        torch_large = torch_ref.get("matmul_512x512", {})
        resident = evidence["resident_matmul"]
        if torch_large.get("elapsed_ms", 0) > 0 and resident.get("elapsed_ms", 0) > 0:
            resident["pytorch_eager_ms"] = torch_large["elapsed_ms"]
            resident["speedup_vs_pytorch_eager"] = torch_large["elapsed_ms"] / resident["elapsed_ms"]
        if "checksum" in resident and "checksum" in torch_large:
            resident["checksum_abs_delta_vs_pytorch"] = abs(resident["checksum"] - torch_large["checksum"])

    payload = {
        "schema_version": 1,
        "contract_version": "v4.0-native-training-runtime",
        "scope": "enkai_native_cuda_runtime",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tools": {"nvcc": nvcc, "nvidia_smi": nvidia_smi},
        "require_cuda": args.require_cuda,
        "native_cuda_cargo_test": cargo,
        "enkai_native_cuda_evidence": evidence,
        "pytorch_reference_only": torch_ref,
        "claim_policy": {
            "pytorch_is_reference_only": True,
            "claim_without_require_cuda_green": False,
            "forbidden_claims": ["faster than CUDA", "full PyTorch parity"]
        },
        "all_passed": not failures,
        "failures": failures,
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": payload["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
