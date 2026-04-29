#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify manifest-backed validation suite dispatch.")
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--contract", default="enkai/contracts/selfhost_validation_suite_dispatch_v3_3_0.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    enkai_bin = Path(args.enkai_bin).resolve()
    contract_path = (root / args.contract).resolve() if not Path(args.contract).is_absolute() else Path(args.contract).resolve()
    output_path = Path(args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    proof_dir = output_path.parent / "validation_suite_dispatch"
    proof_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, object]] = []
    all_passed = True
    for required in contract.get("required_suite_dispatches", []):
        subcommand = required["subcommand"]
        manifest_path = proof_dir / f"{subcommand.replace('-', '_')}_manifest.json"
        result_path = proof_dir / required["result_output"]
        manifest_cmd = [str(enkai_bin), "validate", "suite-manifest", subcommand]
        if subcommand == "determinism":
            manifest_cmd.extend(["--suite", required["suite"], "--runs", str(required["runs"])])
        manifest_cmd.extend(["--json", "--output", str(result_path), "--manifest-output", str(manifest_path)])
        manifest_proc = run_cmd(manifest_cmd, root)
        manifest_ok = manifest_proc.returncode == 0 and manifest_path.is_file()
        manifest_payload = None
        manifest_shape_ok = False
        if manifest_ok:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_shape_ok = manifest_payload.get("command", {}).get("subcommand") == subcommand
        exec_proc = run_cmd([str(enkai_bin), "validate", "suite-exec", "--manifest", str(manifest_path)], root)
        result_ok = exec_proc.returncode == 0 and result_path.is_file()
        result_payload = None
        result_passed = False
        if result_ok:
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            result_passed = bool(result_payload.get("passed"))
            if subcommand == "determinism":
                result_passed = result_passed and result_payload.get("suite") == required["suite"]
        check_passed = manifest_ok and manifest_shape_ok and result_ok and result_passed
        all_passed = all_passed and check_passed
        checks.append({
            "subcommand": subcommand,
            "manifest_command": manifest_cmd,
            "manifest_returncode": manifest_proc.returncode,
            "manifest_stdout": manifest_proc.stdout,
            "manifest_stderr": manifest_proc.stderr,
            "manifest_path": str(manifest_path),
            "manifest_shape_ok": manifest_shape_ok,
            "exec_returncode": exec_proc.returncode,
            "exec_stdout": exec_proc.stdout,
            "exec_stderr": exec_proc.stderr,
            "result_path": str(result_path),
            "result_passed": result_passed,
            "passed": check_passed,
        })

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "output_dir": str(proof_dir),
        "all_passed": all_passed,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok" if all_passed else "failed", "output": str(output_path), "all_passed": all_passed}, separators=(",", ":")))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
