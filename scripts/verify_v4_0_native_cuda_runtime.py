#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--input", default="artifacts/readiness/v4_0_native_cuda_runtime.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_native_cuda_runtime_verify.json")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    root = Path(args.workspace).resolve()
    report = json.loads((root / args.input).read_text(encoding="utf-8-sig"))
    failures: list[str] = []
    if report.get("contract_version") != "v4.0-native-training-runtime":
        failures.append("contract version mismatch")
    if not report.get("all_passed"):
        failures.append("native CUDA readiness report is not green")
    evidence = report.get("enkai_native_cuda_evidence", {})
    if args.require_cuda:
        if evidence.get("skipped"):
            failures.append("native CUDA hardware proof was skipped")
        if evidence.get("pytorch_core_execution_dependency") is not False:
            failures.append("native CUDA proof did not establish PyTorch-free execution")
        for op in ["vec_add", "matmul", "softmax", "cross_entropy"]:
            if op not in evidence.get("ops", []):
                failures.append(f"missing CUDA op proof: {op}")
        if not evidence.get("resident_matmul"):
            failures.append("missing resident CUDA matmul proof")
        resident = evidence.get("resident_matmul", {})
        if resident:
            if resident.get("kernel") != "cublaslt_tensor_core_candidate":
                failures.append("resident CUDA matmul did not use the cuBLASLt/Tensor Core candidate path")
            if resident.get("speedup_vs_internal_cublas", 0.0) < 1.5:
                failures.append("resident CUDA cuBLASLt path did not clear 1.5x internal cuBLAS speedup gate")
            if resident.get("speedup_vs_pytorch_eager", 0.0) < 1.5:
                failures.append("resident CUDA cuBLASLt path did not clear 1.5x PyTorch eager speedup gate")
        if not report.get("pytorch_reference_only", {}).get("available"):
            failures.append("PyTorch CUDA reference comparison missing")
    payload = {
        "schema_version": 1,
        "input": str((root / args.input).resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "require_cuda": args.require_cuda,
        "all_passed": not failures,
        "failures": failures,
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
