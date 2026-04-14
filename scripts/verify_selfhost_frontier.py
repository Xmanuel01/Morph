#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and verify a frozen self-host frontend audit contract."
    )
    parser.add_argument("--enkai-bin", required=True, help="Path to enkai executable")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument(
        "--contract",
        default="enkai/contracts/selfhost_frontier_v3_1_1.json",
        help="Path to the self-host frontier contract JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the verification summary",
    )
    return parser.parse_args()


def read_json(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def native_library_filename() -> str:
    if sys.platform.startswith("win"):
        return "enkai_native.dll"
    if sys.platform == "darwin":
        return "libenkai_native.dylib"
    return "libenkai_native.so"


def ensure_release_native_runtime(root: pathlib.Path, enkai_bin: pathlib.Path) -> list[str]:
    """
    Ensure a release-mode enkai binary has its cdylib/native companion available.

    This keeps strict self-host verification stable after cache clears, where
    `cargo build --release -p enkai` produces the executable but not the FFI
    cdylib that the bootstrap path loads at runtime.
    """
    build_command: list[str] = []
    bin_parts = {part.lower() for part in enkai_bin.parts}
    if "target" not in bin_parts or "release" not in bin_parts:
        return build_command

    candidates = [
        enkai_bin.parent / native_library_filename(),
        enkai_bin.parent / "deps" / native_library_filename(),
    ]
    if any(candidate.is_file() for candidate in candidates):
        return build_command

    build_command = ["cargo", "build", "--release", "-p", "enkai_native"]
    proc = subprocess.run(
        build_command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "failed to build release native runtime companion:\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if not any(candidate.is_file() for candidate in candidates):
        raise RuntimeError(
            f"release native runtime companion still missing after build: {candidates[0].name}"
        )
    return build_command


def subprocess_env(enkai_bin: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    path_entries = [str(enkai_bin.parent), str(enkai_bin.parent / "deps")]
    current_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_entries + ([current_path] if current_path else []))
    return env


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.workspace).resolve()
    enkai_bin = pathlib.Path(args.enkai_bin).resolve()
    contract_path = (root / args.contract).resolve() if not pathlib.Path(args.contract).is_absolute() else pathlib.Path(args.contract)
    output_path = (root / args.output).resolve() if not pathlib.Path(args.output).is_absolute() else pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    native_build_command = ensure_release_native_runtime(root, enkai_bin)

    contract = read_json(contract_path)
    corpus = root / contract["corpus"]
    report_path = root / contract["report_file"]
    triage_dir = report_path.parent
    triage_dir.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        report_path.unlink()

    audit_command = str(contract.get("command", "frontend-audit"))
    if audit_command not in {"frontend-audit", "negative-audit"}:
        raise RuntimeError(f"unsupported audit command: {audit_command}")

    command = [
        str(enkai_bin),
        "litec",
        audit_command,
        str(corpus),
        "--triage-dir",
        str(triage_dir),
    ]
    proc = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        env=subprocess_env(enkai_bin),
    )
    fallback_command: list[str] | None = None
    if proc.returncode != 0 and f"unknown litec subcommand: {audit_command}" in proc.stderr:
        fallback_command = [
            "cargo",
            "run",
            "-p",
            "enkai",
            "--",
            "litec",
            audit_command,
            str(corpus),
            "--triage-dir",
            str(triage_dir),
        ]
        proc = subprocess.run(
            fallback_command,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            env=subprocess_env(enkai_bin),
        )

    if not report_path.is_file():
        raise RuntimeError(f"frontend audit report not found: {report_path}")
    report = read_json(report_path)
    expected = contract.get("expected", {})
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("contract expected map must be a non-empty object")

    actual_by_name: dict[str, str] = {}
    for entry in report.get("entries", []):
        if not isinstance(entry, dict):
            continue
        raw_file = str(entry.get("file", "")).replace("\\", "/")
        if not raw_file:
            continue
        status = str(entry.get("status", "unknown"))
        actual_by_name[raw_file] = status
        actual_by_name[pathlib.Path(raw_file).name] = status
        try:
            raw_path = pathlib.Path(raw_file)
            if raw_path.is_absolute():
                relative = raw_path.resolve().relative_to(root).as_posix()
                actual_by_name[relative] = status
        except Exception:
            pass

    mismatches = []
    for name, expected_status in expected.items():
        actual_status = actual_by_name.get(name)
        if actual_status != expected_status:
            mismatches.append(
                {
                    "file": name,
                    "expected": expected_status,
                    "actual": actual_status,
                }
            )

    summary = {
        "schema_version": 1,
        "profile": str(contract.get("profile", "strict_selfhost_frontend")),
        "status": "ok" if proc.returncode == 0 and not mismatches else "failed",
        "command": command,
        "native_build_command": native_build_command,
        "fallback_command": fallback_command,
        "contract": str(contract_path.relative_to(root)),
        "report": str(report_path.relative_to(root)),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "exit_code": proc.returncode,
        "expected_files": sorted(expected.keys()),
        "mismatches": mismatches,
        "full_support_ready": bool(report.get("full_support_ready", False)),
        "frontier_gap_files": int(report.get("frontier_gap_files", 0)),
        "invalid_files": int(report.get("invalid_files", 0)),
        "selfhost_frontend_files": int(report.get("selfhost_frontend_files", 0)),
        "rust_frontend_files": int(report.get("rust_frontend_files", 0)),
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(output_path.relative_to(root)),
                "frontier_gap_files": summary["frontier_gap_files"],
                "invalid_files": summary["invalid_files"],
            },
            separators=(",", ":"),
        )
    )
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
