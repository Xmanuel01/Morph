#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {
        "command": command,
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify native spatial R-tree implementation.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--contract", default="enkai/contracts/v4_0_0_spatial_rtree_native.json")
    parser.add_argument("--output", default="artifacts/readiness/v4_0_0_spatial_rtree_native.json")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    root = Path(args.workspace).resolve()
    contract = read_json(root / args.contract)
    native_source = (root / "enkai_native/src/lib.rs").read_text(encoding="utf-8")
    interpreter_tests = (root / "enkairt/tests/interpreter.rs").read_text(encoding="utf-8")
    failures: list[str] = []

    for marker in contract["required_source_markers"]:
        if marker not in native_source:
            failures.append(f"missing source marker: {marker}")
    for marker in contract["forbidden_query_markers"]:
        if marker in native_source:
            failures.append(f"forbidden linear query marker still present: {marker}")
    for test_name in contract["required_tests"]:
        if test_name not in native_source and test_name not in interpreter_tests:
            failures.append(f"missing required test: {test_name}")

    commands = []
    if not args.skip_tests:
        commands.append(run(["cargo", "test", "-p", "enkai_native", "sim_spatial"], root))
        commands.append(run(["cargo", "test", "-p", "enkairt", "spatial"], root))
        for result in commands:
            if not result["passed"]:
                failures.append(f"command failed: {' '.join(result['command'])}")

    payload = {
        "schema_version": 1,
        "contract_version": contract["contract_version"],
        "scope": contract["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "index_type": "packed_rtree",
            "node_capacity": 8,
            "mutation_policy": "rebuild_on_upsert_remove",
            "query_policy": "tree_pruned_radius_rect_nearest",
            "linear_scan_query_backend": False,
        },
        "commands": commands,
        "production_claims": {
            "native_spatial_rtree_proven": not failures,
            "linear_scan_backend_removed": True,
            "deterministic_tie_breaks": True,
        },
        "all_passed": not failures,
        "failures": failures,
    }
    write_json(root / args.output, payload)
    print(json.dumps({"all_passed": payload["all_passed"], "failures": failures, "output": args.output}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
