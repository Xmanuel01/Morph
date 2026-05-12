#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(command: list[str], cwd: Path, timeout: int = 300) -> dict[str, Any]:
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
    for line in stdout.splitlines():
        if line.startswith("ENKAI_NATIVE_RUNTIME_EVIDENCE="):
            return json.loads(line.split("=", 1)[1])
    return {"missing": True}


def py_vector_add(n: int, iters: int) -> dict[str, Any]:
    a = [float((i % 17) - 8) for i in range(n)]
    b = [float((i % 19) - 9) for i in range(n)]
    started = time.perf_counter()
    checksum = 0.0
    for _ in range(iters):
        out = [x + y for x, y in zip(a, b)]
        checksum += sum(out)
    return {"elapsed_ms": (time.perf_counter() - started) * 1000.0, "checksum": checksum}


def py_matmul(n: int, iters: int) -> dict[str, Any]:
    a = [[((r * n + c) % 17 - 8) * 0.01 for c in range(n)] for r in range(n)]
    b = [[((r * n + c) % 19 - 9) * 0.01 for c in range(n)] for r in range(n)]
    started = time.perf_counter()
    checksum = 0.0
    for _ in range(iters):
        out = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for k in range(n):
                aik = a[i][k]
                for j in range(n):
                    out[i][j] += aik * b[k][j]
        checksum += sum(sum(row) for row in out)
    return {"elapsed_ms": (time.perf_counter() - started) * 1000.0, "checksum": checksum}


def torch_reference(n: int, iters: int) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "reason": f"torch import failed: {exc}"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.arange(n * n, dtype=torch.float32, device=device).reshape(n, n).remainder(17).sub(8).mul(0.01)
    b = torch.arange(n * n, dtype=torch.float32, device=device).reshape(n, n).remainder(19).sub(9).mul(0.01)
    if device == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    checksum = 0.0
    for _ in range(iters):
        out = a @ b
        checksum += float(out.sum().detach().cpu())
    if device == "cuda": torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - started) * 1000.0
    compiled = {"available": False, "reason": "torch.compile unavailable or skipped"}
    if hasattr(torch, "compile"):
        try:
            fn = torch.compile(lambda x, y: x @ y)
            if device == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            compiled_checksum = 0.0
            for _ in range(iters):
                compiled_checksum += float(fn(a, b).sum().detach().cpu())
            if device == "cuda": torch.cuda.synchronize()
            compiled = {"available": True, "elapsed_ms": (time.perf_counter() - started) * 1000.0, "checksum": compiled_checksum}
        except Exception as exc:
            compiled = {"available": False, "reason": str(exc)}
    return {"available": True, "device": device, "version": getattr(torch, "__version__", "unknown"), "eager": {"elapsed_ms": eager_ms, "checksum": checksum}, "compile": compiled}


def source_checks(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    native = (root / "enkai_tensor" / "src" / "native_runtime.rs").read_text(encoding="utf-8")
    lib = (root / "enkai_tensor" / "src" / "lib.rs").read_text(encoding="utf-8")
    failures: list[str] = []
    for token in ["TensorGraph", "GraphOp", "CpuBackend", "MemoryPlanner", "fused_add_relu", "fused_matmul_bias_relu", "train_mlp_sgd", "train_mlp_adamw", "AdamWConfig", "CudaBackendHook", "benchmark_native_runtime"]:
        if token not in native:
            failures.append(f"missing native runtime token: {token}")
    forbidden_patterns = ["use tch", "tch::", "pyo3", "python::", "torch::"]
    for forbidden in forbidden_patterns:
        if forbidden in native.lower():
            failures.append(f"forbidden core execution dependency in native runtime: {forbidden}")
    if "pub mod native_runtime" not in lib:
        failures.append("native_runtime module is not exported")
    return {"passed": not failures, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_native_training_runtime.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_native_training_runtime.json")
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.workspace).resolve()
    contract = json.loads((root / args.contract).read_text(encoding="utf-8-sig"))
    cargo = run(["cargo", "test", "-p", "enkai_tensor", "--release", "--test", "native_training_runtime", "--", "--nocapture"], root, timeout=600)
    evidence = parse_evidence(cargo.get("stdout_tail", ""))
    checks = source_checks(root, contract)
    python = {"vector_add": py_vector_add(128 * 128, args.iterations), "matmul": py_matmul(128, args.iterations)}
    torch = torch_reference(128, args.iterations)
    comparison: dict[str, Any] = {}

    failures: list[str] = []
    if not cargo["passed"]: failures.append("native runtime cargo test failed")
    if evidence.get("missing"): failures.append("native runtime benchmark evidence missing")
    if not checks["passed"]: failures.extend(checks["failures"])
    if evidence and not evidence.get("missing"):
        if evidence.get("claims", {}).get("pytorch_core_execution_dependency") is not False:
            failures.append("native runtime still claims PyTorch core dependency")
        training = evidence.get("training", {})
        if not (training.get("loss_final", math.inf) < training.get("loss_initial", -math.inf)):
            failures.append("MLP SGD training did not reduce loss")
        memory = evidence.get("memory", {}).get("unfused", {})
        if memory.get("peak_bytes", 0) <= 0 or memory.get("allocated_bytes", 0) <= 0:
            failures.append("memory planner metrics missing")
        if memory.get("reuse_count", 0) <= 0:
            failures.append("memory planner did not demonstrate reuse")
        if evidence.get("matmul_bias_relu", {}).get("checksum_delta_abs", 1.0) > 1e-4:
            failures.append("fused matmul/bias/relu checksum mismatch")
        if evidence.get("softmax_cross_entropy", {}).get("loss_delta_abs", 1.0) > 1e-4:
            failures.append("fused softmax/cross_entropy loss mismatch")
        for workload in contract.get("required_metrics", []):
            if workload.startswith("benchmark_"):
                name = workload.removeprefix("benchmark_")
                if evidence.get("benchmarks", {}).get(name) is None:
                    failures.append(f"missing benchmark evidence: {name}")
        native_vector_ms = evidence.get("benchmarks", {}).get("vector_add", {}).get("elapsed_ms")
        native_matmul_ms = evidence.get("benchmarks", {}).get("matmul", {}).get("elapsed_ms")
        if native_vector_ms and native_vector_ms > 0:
            comparison["vector_add_vs_python"] = {
                "python_ms": python["vector_add"]["elapsed_ms"],
                "enkai_ms": native_vector_ms,
                "speedup": python["vector_add"]["elapsed_ms"] / native_vector_ms,
            }
            if comparison["vector_add_vs_python"]["speedup"] < 1.5:
                failures.append("native vector_add is not at least 1.5x faster than Python baseline")
        if native_matmul_ms and native_matmul_ms > 0:
            comparison["matmul_vs_python"] = {
                "python_ms": python["matmul"]["elapsed_ms"],
                "enkai_ms": native_matmul_ms,
                "speedup": python["matmul"]["elapsed_ms"] / native_matmul_ms,
            }
            if comparison["matmul_vs_python"]["speedup"] < 1.5:
                failures.append("native matmul is not at least 1.5x faster than Python baseline")
            if torch.get("available") and torch.get("eager", {}).get("elapsed_ms", 0) > 0:
                comparison["matmul_vs_pytorch_eager"] = {
                    "pytorch_eager_ms": torch["eager"]["elapsed_ms"],
                    "enkai_ms": native_matmul_ms,
                    "speedup": torch["eager"]["elapsed_ms"] / native_matmul_ms,
                    "claim_allowed": torch["eager"]["elapsed_ms"] > native_matmul_ms,
                }

    payload = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0], "cwd": str(root)},
        "contract": str((root / args.contract).resolve()),
        "suite": "bench/suites/v4_0_native_training_runtime.json",
        "source_checks": checks,
        "native_cargo_test": cargo,
        "enkai_native_evidence": evidence,
        "python_reference": python,
        "pytorch_reference_only": torch,
        "native_reference_comparison": comparison,
        "claim_policy": contract.get("claim_policy", {}),
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
