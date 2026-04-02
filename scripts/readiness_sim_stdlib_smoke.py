#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_RESULT = [
    18.0,
    [5, 10, 20],
    [True, False, 7, 1],
    [2, 2],
]
PROFILE_CASE = "readiness_sim_stdlib_smoke"


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env)
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
    env = dict(os.environ)
    env["ENKAI_SIM_ACCEL"] = "1"

    if not enkai_bin.exists():
        raise SystemExit(f"enkai binary not found: {enkai_bin}")

    sim_dir = workspace / "artifacts" / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    run_report = sim_dir / "stdlib_smoke_run.json"
    profile_report = sim_dir / "stdlib_smoke_profile.json"

    with tempfile.TemporaryDirectory(prefix="enkai_sim_stdlib_smoke_") as tmp:
        root = Path(tmp)
        script = root / "sim_stdlib_smoke.enk"
        script.write_text(
            (
                "import std::sparse\n"
                "import std::event\n"
                "import std::pool\n"
                "fn main() ::\n"
                "    let v := sparse.vector()\n"
                "    sparse.set_vector(v, 0, 1.5)\n"
                "    sparse.set_vector(v, 3, 2.0)\n"
                "    let dot := sparse.dot(v, [4.0, 1.0, 9.0, 5.0]) + sparse.nnz(v)\n"
                "    let q := event.make()\n"
                "    event.push(q, 1.0, 10)\n"
                "    event.push(q, 1.0, 20)\n"
                "    event.push(q, 0.5, 5)\n"
                "    let a := event.pop(q)?\n"
                "    let b := event.pop(q)?\n"
                "    let c := event.pop(q)?\n"
                "    let p := pool.make(1)\n"
                "    let first := pool.release(p, 7)\n"
                "    let second := pool.release(p, 9)\n"
                "    let got := pool.acquire(p)?\n"
                "    let stats := pool.stats(p)\n"
                "    let g := pool.make_growable(1)\n"
                "    pool.release(g, 1)\n"
                "    pool.release(g, 2)\n"
                "    let grow_stats := pool.stats(g)\n"
                "    return [dot, [a.event, b.event, c.event], [first, second, got, stats.dropped_on_full], [grow_stats.capacity, grow_stats.available]]\n"
                "::\n"
                "main()\n"
            ),
            encoding="utf-8",
        )

        run(
            [str(enkai_bin), "sim", "run", "--output", str(run_report), str(script)],
            workspace,
            env,
        )
        run_payload = read_json(run_report)
        if run_payload.get("command") != "sim.run":
            raise SystemExit("simulation stdlib smoke run report command mismatch")
        if run_payload.get("result") != EXPECTED_RESULT:
            raise SystemExit("simulation stdlib smoke run result mismatch")

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
            env,
        )
        profile_payload = read_json(profile_report)
        if profile_payload.get("case") != PROFILE_CASE:
            raise SystemExit("simulation stdlib smoke profile case mismatch")
        if profile_payload.get("status") != "ok":
            raise SystemExit("simulation stdlib smoke profile status was not ok")

        summary = {
            "schema_version": 1,
            "script": str(script),
            "run_report": str(run_report),
            "profile_report": str(profile_report),
            "expected_result": EXPECTED_RESULT,
            "profile_case": PROFILE_CASE,
            "require_native_accel": True,
        }
        write_json(output, summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
