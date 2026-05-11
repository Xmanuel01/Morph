#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(command: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except Exception as exc:
        return {
            "command": command,
            "exit_code": 1,
            "passed": False,
            "stdout": "",
            "stderr": repr(exc),
        }


def run_python_json(python_cmd: list[str], code: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            python_cmd + ["-"],
            input=code,
            text=True,
            capture_output=True,
            timeout=120,
        )
        raw = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
        payload = json.loads(raw)
        payload["process"] = {
            "command": python_cmd + ["-"],
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stderr": proc.stderr[-4000:],
        }
        return payload
    except Exception as exc:
        return {
            "available": False,
            "cuda_available": False,
            "error": repr(exc),
            "process": {"command": python_cmd + ["-"], "exit_code": 1, "passed": False},
        }


def parse_python(value: str | None) -> list[str]:
    if value:
        return value.split()
    if os.name == "nt":
        return ["py", "-3.11"]
    return [sys.executable]


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight v3.9.0 CUDA/GPU proof machine readiness.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--python", default=None)
    parser.add_argument("--output", default="artifacts/readiness/v3_9_0_gpu_preflight.json")
    parser.add_argument("--require-nvcc", action="store_true")
    parser.add_argument("--require-two-gpus", action="store_true")
    parser.add_argument("--require-four-gpus", action="store_true")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    python_cmd = parse_python(args.python)
    nvidia_smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")

    smi_query = (
        run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            timeout=60,
        )
        if nvidia_smi
        else {"passed": False, "stdout": "", "stderr": "nvidia-smi not found"}
    )
    gpus = []
    if smi_query.get("passed"):
        for line in smi_query.get("stdout", "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    {
                        "name": parts[0],
                        "driver_version": parts[1],
                        "memory_total": parts[2],
                        "compute_capability": parts[3],
                    }
                )

    torch = run_python_json(
        python_cmd,
        """
import json
payload = {"available": False, "cuda_available": False}
try:
    import torch
    payload.update({
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    })
except Exception as exc:
    payload["error"] = str(exc)
print(json.dumps(payload))
""",
    )

    cargo = run(["cargo", "--version"], timeout=30)
    rustc = run(["rustc", "--version"], timeout=30)
    build = run(["cargo", "build", "-p", "enkai_tensor", "--features", "torch"], timeout=900)

    failures: list[str] = []
    if not nvidia_smi:
        failures.append("nvidia-smi not found")
    if not gpus:
        failures.append("no NVIDIA GPUs detected by nvidia-smi")
    if args.require_nvcc and not nvcc:
        failures.append("nvcc not found but --require-nvcc was set")
    required_gpu_count = 4 if args.require_four_gpus else 2 if args.require_two_gpus else 1
    if len(gpus) < required_gpu_count:
        failures.append(f"expected at least {required_gpu_count} CUDA GPUs, found {len(gpus)}")
    if torch.get("available") is not True:
        failures.append("PyTorch is not importable in selected Python")
    if torch.get("cuda_available") is not True:
        failures.append("PyTorch CUDA is not available")
    if int(torch.get("device_count") or 0) < required_gpu_count:
        failures.append(
            f"PyTorch sees fewer than {required_gpu_count} CUDA GPUs: {torch.get('device_count')}"
        )
    if cargo.get("passed") is not True:
        failures.append("cargo is unavailable")
    if rustc.get("passed") is not True:
        failures.append("rustc is unavailable")
    if build.get("passed") is not True:
        failures.append("cargo build -p enkai_tensor --features torch failed")

    payload = {
        "schema_version": 1,
        "contract_version": "v3.9.0",
        "scope": "gpu_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "cwd": str(root),
            "python_command": python_cmd,
        },
        "tools": {
            "nvidia_smi": nvidia_smi,
            "nvcc": nvcc,
            "cargo": cargo,
            "rustc": rustc,
        },
        "gpus": gpus,
        "torch": torch,
        "build_torch_feature": build,
        "requirements": {
            "required_gpu_count": required_gpu_count,
            "require_nvcc": bool(args.require_nvcc),
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
