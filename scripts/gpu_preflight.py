#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def resolve_enkai() -> pathlib.Path | None:
    env_path = os.environ.get("ENKAI_EXE")
    if env_path:
        path = pathlib.Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
    which = shutil.which("enkai")
    if which:
        return pathlib.Path(which).resolve()
    local_candidates = [
        ROOT / "target" / "release" / ("enkai.exe" if os.name == "nt" else "enkai"),
        ROOT / "target" / "debug" / ("enkai.exe" if os.name == "nt" else "enkai"),
    ]
    for candidate in local_candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_tensor_lib() -> pathlib.Path | None:
    env_path = os.environ.get("ENKAI_TENSOR_PATH")
    if env_path:
        path = pathlib.Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
    names: list[str] = []
    if os.name == "nt":
        names.append("enkai_tensor.dll")
    elif sys.platform == "darwin":
        names.append("libenkai_tensor.dylib")
    else:
        names.append("libenkai_tensor.so")
    candidates: list[pathlib.Path] = []
    for name in names:
        candidates.extend(
            [
                ROOT / "target" / "release" / name,
                ROOT / "target" / "debug" / name,
                ROOT / "target" / "release" / "deps" / name,
                ROOT / "target" / "debug" / "deps" / name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight a host for Enkai GPU sign-off execution."
    )
    parser.add_argument(
        "--profile",
        choices=["single", "multi", "soak4", "full"],
        default="full",
        help="GPU sign-off profile to validate",
    )
    parser.add_argument(
        "--config",
        default="configs/enkai_50m.enk",
        help="Base train config expected by GPU soak/harness scripts",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSON report output path",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/gpu",
        help="GPU evidence directory to verify/create",
    )
    return parser.parse_args()


def run_checked(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return False, "command_not_found"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        return False, stderr or f"exit_{proc.returncode}"
    return True, (proc.stdout or "").strip()


def torch_cuda_probe() -> dict[str, Any]:
    python = shutil.which("py") or shutil.which("python3") or shutil.which("python")
    if python is None:
        return {"ok": False, "reason": "python_not_found"}
    cmd = [python]
    if pathlib.Path(python).name.lower() == "py.exe" or pathlib.Path(python).name.lower() == "py":
        cmd.append("-3")
    cmd.extend(
        [
            "-c",
            (
                "import json,sys\n"
                "try:\n"
                " import torch\n"
                " print(json.dumps({"
                "\"torch_version\": getattr(torch, '__version__', None),"
                "\"cuda_available\": bool(torch.cuda.is_available()),"
                "\"device_count\": int(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
                "}))\n"
                "except Exception as exc:\n"
                " print(json.dumps({\"error\": str(exc)}))\n"
                " sys.exit(2)\n"
            ),
        ]
    )
    ok, output = run_checked(cmd)
    if not ok:
        return {"ok": False, "reason": output}
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return {"ok": False, "reason": "invalid_python_probe_output", "raw": output}
    if "error" in payload:
        return {"ok": False, "reason": payload["error"]}
    payload["ok"] = True
    return payload


def nvidia_probe() -> dict[str, Any]:
    ok, output = run_checked(["nvidia-smi", "-L"])
    if not ok:
        return {"ok": False, "reason": output, "gpu_count": 0, "gpus": []}
    gpus = [line.strip() for line in output.splitlines() if line.strip()]
    return {"ok": True, "gpu_count": len(gpus), "gpus": gpus}


def required_gpu_count(profile: str) -> int:
    return {
        "single": 1,
        "multi": 2,
        "soak4": 4,
        "full": 4,
    }[profile]


def expected_scripts(profile: str) -> list[str]:
    common = [
        "scripts/soak_single_gpu.ps1",
        "scripts/soak_single_gpu.sh",
        "scripts/verify_gpu_gates.ps1",
        "scripts/verify_gpu_gates.sh",
        "scripts/gpu_harness.py",
    ]
    if profile in {"multi", "soak4", "full"}:
        common.extend(
            [
                "scripts/multi_gpu_harness.ps1",
                "scripts/multi_gpu_harness.sh",
            ]
        )
    if profile in {"soak4", "full"}:
        common.extend(
            [
                "scripts/soak_4gpu.ps1",
                "scripts/soak_4gpu.sh",
            ]
        )
    return common


def profile_commands(profile: str, artifact_dir: pathlib.Path) -> list[str]:
    commands = [
        f"powershell -ExecutionPolicy Bypass -File scripts/soak_single_gpu.ps1",
    ]
    if profile in {"multi", "soak4", "full"}:
        commands.append(
            "powershell -ExecutionPolicy Bypass -File scripts/multi_gpu_harness.ps1"
        )
    if profile in {"soak4", "full"}:
        commands.append("powershell -ExecutionPolicy Bypass -File scripts/soak_4gpu.ps1")
    if profile == "full":
        commands.append(
            f"powershell -ExecutionPolicy Bypass -File scripts/verify_gpu_gates.ps1 -LogDir {artifact_dir}"
        )
        commands.append(
            f"powershell -ExecutionPolicy Bypass -File scripts/v3_0_0_rc_pipeline.ps1"
        )
    return commands


def main() -> int:
    args = parse_args()
    output_path = pathlib.Path(args.output)
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    artifact_dir = pathlib.Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = (ROOT / artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    enkai_bin = resolve_enkai()
    tensor_lib = resolve_tensor_lib()
    nvidia = nvidia_probe()
    torch_probe = torch_cuda_probe()
    scripts = expected_scripts(args.profile)

    checks: list[dict[str, Any]] = []

    checks.append(
        {
            "id": "enkai-bin",
            "ok": enkai_bin is not None,
            "details": str(enkai_bin) if enkai_bin is not None else "not_found",
        }
    )
    checks.append(
        {
            "id": "tensor-lib",
            "ok": tensor_lib is not None,
            "details": str(tensor_lib) if tensor_lib is not None else "not_found",
        }
    )
    checks.append(
        {
            "id": "config",
            "ok": config_path.is_file(),
            "details": str(config_path),
        }
    )
    checks.append(
        {
            "id": "artifact-dir",
            "ok": artifact_dir.is_dir(),
            "details": str(artifact_dir),
        }
    )
    for script in scripts:
        path = ROOT / script
        checks.append({"id": f"script:{script}", "ok": path.is_file(), "details": str(path)})

    checks.append(
        {
            "id": "nvidia-smi",
            "ok": bool(nvidia["ok"]),
            "details": nvidia,
        }
    )
    checks.append(
        {
            "id": "gpu-count",
            "ok": int(nvidia.get("gpu_count", 0)) >= required_gpu_count(args.profile),
            "details": {
                "required": required_gpu_count(args.profile),
                "detected": int(nvidia.get("gpu_count", 0)),
            },
        }
    )
    checks.append({"id": "torch-cuda", "ok": bool(torch_probe.get("ok")), "details": torch_probe})
    if torch_probe.get("ok"):
        checks.append(
            {
                "id": "torch-device-count",
                "ok": int(torch_probe.get("device_count", 0)) >= required_gpu_count(args.profile),
                "details": {
                    "required": required_gpu_count(args.profile),
                    "detected": int(torch_probe.get("device_count", 0)),
                    "torch_version": torch_probe.get("torch_version"),
                },
            }
        )

    missing = [check["id"] for check in checks if not check["ok"]]
    report = {
        "schema_version": 1,
        "profile": args.profile,
        "status": "PASS" if not missing else "FAIL",
        "required_gpu_count": required_gpu_count(args.profile),
        "enkai_bin": str(enkai_bin) if enkai_bin is not None else None,
        "tensor_lib": str(tensor_lib) if tensor_lib is not None else None,
        "config": str(config_path),
        "artifact_dir": str(artifact_dir),
        "checks": checks,
        "missing": missing,
        "next_steps": profile_commands(args.profile, artifact_dir),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output_path), "missing": missing}))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
