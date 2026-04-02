#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_VALUE = 77
EXPECTED_BUFFER_LEN = 3
PROFILE_CASE = "readiness_sim_native_smoke"


def run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin)
    output = workspace / args.output

    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    sim_dir = workspace / "artifacts" / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    run_report = sim_dir / "native_smoke_run.json"
    profile_report = sim_dir / "native_smoke_profile.json"

    run(["cargo", "build", "-p", "enkai_native"], workspace)

    with tempfile.TemporaryDirectory(prefix="enkai_sim_native_smoke_") as tmp:
        root = Path(tmp)
        script = root / "sim_native_smoke.enk"
        script.write_text(
            (
                "native::import \"enkai_native\" ::\n"
                "    fn handle_new(value: Int) -> Handle\n"
                "    fn handle_read(handle: Handle) -> Int\n"
                "    fn buffer_from_string(data: String) -> Buffer\n"
                "    fn buffer_len(data: Buffer) -> Int\n"
                "::\n"
                "fn main() ::\n"
                "    let handle := handle_new(77)\n"
                "    let payload := buffer_from_string(\"abc\")\n"
                "    return [handle_read(handle), buffer_len(payload)]\n"
                "::\n"
                "main()\n"
            ),
            encoding="utf-8",
        )

        run(
            [str(enkai_bin), "sim", "run", "--output", str(run_report), str(script)],
            workspace,
        )
        run_payload = read_json(run_report)
        if run_payload.get("command") != "sim.run":
            raise SystemExit("simulation native smoke run report command mismatch")
        result = run_payload.get("result")
        if result != [EXPECTED_VALUE, EXPECTED_BUFFER_LEN]:
            raise SystemExit("simulation native smoke run result mismatch")

        run(
            [
                str(enkai_bin),
                "sim",
                "profile",
                "--case",
                PROFILE_CASE,
                "--output",
                str(profile_report),
                str(script),
            ],
            workspace,
        )
        profile_payload = read_json(profile_report)
        if profile_payload.get("case") != PROFILE_CASE:
            raise SystemExit("simulation native smoke profile case mismatch")
        if profile_payload.get("status") != "ok":
            raise SystemExit("simulation native smoke profile status was not ok")

        summary = {
            "schema_version": 1,
            "script": str(script),
            "run_report": str(run_report),
            "profile_report": str(profile_report),
            "expected_value": EXPECTED_VALUE,
            "expected_buffer_len": EXPECTED_BUFFER_LEN,
            "profile_case": PROFILE_CASE,
        }
        write_json(output, summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
