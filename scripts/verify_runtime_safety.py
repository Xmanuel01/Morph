#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ffi-safety", required=True)
    parser.add_argument("--pool-safety", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = read_json(Path(args.summary))
    ffi_safety = read_json(Path(args.ffi_safety))
    pool_safety = read_json(Path(args.pool_safety))

    if summary.get("schema_version") != 1:
        fail("runtime safety summary schema_version must be 1")
    checks = summary.get("checks")
    if not isinstance(checks, list) or not checks:
        fail("runtime safety summary must contain checks")
    for check in checks:
        if check.get("passed") is not True:
            fail(f"runtime safety check failed: {check.get('id')}")
    if summary.get("all_passed") is not True:
        fail("runtime safety summary all_passed must be true")

    if ffi_safety.get("passed") is not True:
        fail("ffi safety validation must pass")
    proof = ffi_safety.get("proof_checks", {})
    for key in (
        "result_matches_expected",
        "null_return_error_stable",
        "oversized_buffer_error_stable",
        "invalid_utf8_error_stable",
        "handle_live_count_zero",
    ):
        if proof.get(key) is not True:
            fail(f"ffi safety proof check failed: {key}")

    if pool_safety.get("passed") is not True:
        fail("pool safety validation must pass")
    pool_proof = pool_safety.get("proof_checks", {})
    for key in ("native_vm_equal", "high_watermark_matches_capacity", "handle_live_count_zero"):
        if pool_proof.get(key) is not True:
            fail(f"pool safety proof check failed: {key}")

    payload = {
        "schema_version": 1,
        "summary": args.summary,
        "ffi_safety": args.ffi_safety,
        "pool_safety": args.pool_safety,
        "all_passed": True,
    }
    write_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
