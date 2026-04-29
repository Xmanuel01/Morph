#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the v3.3.0 self-host worker runtime surface."
    )
    parser.add_argument("--enkai-bin", required=True, help="Path to enkai executable")
    parser.add_argument(
        "--contract",
        default="enkai/contracts/selfhost_worker_runtime_v3_3_0.json",
        help="Path to the worker runtime surface contract JSON",
    )
    parser.add_argument(
        "--controlplane-report",
        default="artifacts/readiness/strict_selfhost_systems_controlplane_slice.json",
        help="Path to the control-plane slice report JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/readiness/strict_selfhost_worker_runtime_surface.json",
        help="Path to write the verification report JSON",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def ensure(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def parse_worker_exec_report(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        return {}
    if text.startswith("{"):
        return json.loads(text)
    match = re.search(
        r"status=(?P<status>\S+) acked=(?P<acked>\d+) requeued=(?P<requeued>\d+) dead_lettered=(?P<dead>\d+)",
        text,
    )
    if not match:
        return {}
    return {
        "status": match.group("status"),
        "processed": 1,
        "acked": int(match.group("acked")),
        "requeued": int(match.group("requeued")),
        "dead_lettered": int(match.group("dead")),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    enkai_bin = Path(args.enkai_bin).resolve()
    contract_path = (repo_root / args.contract).resolve()
    controlplane_report_path = (repo_root / args.controlplane_report).resolve()
    output_path = (repo_root / args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    controlplane_report = json.loads(controlplane_report_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="enkai_worker_surface_", dir=str(repo_root)) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        state_dir = temp_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        fail_handler = temp_dir / "handler_fail.enk"
        fail_handler.write_text(
            "fn main() ::\n    missing_symbol()\n::\nmain()\n",
            encoding="utf-8",
        )
        ack_handler = temp_dir / "handler_ack.enk"
        ack_handler.write_text("fn main() ::\n    return 0\n::\nmain()\n", encoding="utf-8")

        enqueue_manifest = temp_dir / "worker_enqueue.json"
        enqueue_manifest_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "worker-manifest",
                "enqueue",
                "--queue",
                "jobs",
                "--dir",
                str(state_dir),
                "--payload",
                "{\"job\":1}",
                "--id",
                "retry-job",
                "--max-attempts",
                "2",
                "--json",
                "--output",
                str(enqueue_manifest),
            ],
            repo_root,
        )
        enqueue_exec_result = run_command(
            [str(enkai_bin), "systems", "worker-exec", "--manifest", str(enqueue_manifest)],
            repo_root,
        )

        run_manifest = temp_dir / "worker_run.json"
        run_manifest_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "worker-manifest",
                "run",
                "--queue",
                "jobs",
                "--dir",
                str(state_dir),
                "--handler",
                str(fail_handler),
                "--once",
                "--json",
                "--output",
                str(run_manifest),
            ],
            repo_root,
        )
        run_exec_result = run_command(
            [str(enkai_bin), "systems", "worker-exec", "--manifest", str(run_manifest)],
            repo_root,
        )
        ack_run_manifest = temp_dir / "worker_run_ack.json"
        ack_run_manifest_result = run_command(
            [
                str(enkai_bin),
                "systems",
                "worker-manifest",
                "run",
                "--queue",
                "jobs",
                "--dir",
                str(state_dir),
                "--handler",
                str(ack_handler),
                "--once",
                "--json",
                "--output",
                str(ack_run_manifest),
            ],
            repo_root,
        )
        ack_run_exec_result = run_command(
            [str(enkai_bin), "systems", "worker-exec", "--manifest", str(ack_run_manifest)],
            repo_root,
        )

        queue_root = state_dir / "queues" / "jobs"
        queue_files = {name: queue_root / name for name in contract["required_queue_files"]}

        report = {
            "schema_version": 1,
            "profile": contract.get("profile"),
            "contract": str(contract_path),
            "contract_version": contract.get("contract_version"),
            "controlplane_report": str(controlplane_report_path),
            "all_passed": False,
            "cases": {
                "worker_manifest_enqueue": enqueue_manifest_result,
                "worker_exec_enqueue": enqueue_exec_result,
                "worker_manifest_run": run_manifest_result,
                "worker_exec_run": run_exec_result,
                "worker_manifest_ack_run": ack_run_manifest_result,
                "worker_exec_ack_run": ack_run_exec_result,
            },
            "validations": {},
        }

        failures: list[str] = []
        ensure(controlplane_report_path.is_file(), "control-plane report is missing", failures)
        ensure(enqueue_manifest_result["ok"], "worker enqueue manifest command failed", failures)
        ensure(enqueue_exec_result["ok"], "worker enqueue exec command failed", failures)
        ensure(run_manifest_result["ok"], "worker run manifest command failed", failures)
        ensure(run_exec_result["ok"], "worker run exec command failed", failures)
        ensure(ack_run_manifest_result["ok"], "worker ack manifest command failed", failures)
        ensure(ack_run_exec_result["ok"], "worker ack exec command failed", failures)

        if not failures:
            enqueue_payload = json.loads(enqueue_manifest.read_text(encoding="utf-8"))
            run_report = parse_worker_exec_report(run_exec_result["stdout"])
            ack_run_report = parse_worker_exec_report(ack_run_exec_result["stdout"])
            queue_state = json.loads(queue_files["queue_state.json"].read_text(encoding="utf-8"))
            pending_items = []
            inflight_items = []
            scheduled_items = []
            dead_letter_items = []
            for name, path in queue_files.items():
                ensure(path.is_file(), f"missing queue backend file {name}", failures)
            if not failures:
                pending_items = [
                    line for line in queue_files["pending.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()
                ]
                inflight_items = [
                    line for line in queue_files["inflight.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()
                ]
                scheduled_items = [
                    line for line in queue_files["scheduled.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()
                ]
                dead_letter_items = [
                    line for line in queue_files["dead_letter.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()
                ]

            report["validations"] = {
                "enqueue_manifest": enqueue_payload,
                "run_report": run_report,
                "ack_run_report": ack_run_report,
                "queue_state": queue_state,
                "pending_items": pending_items,
                "inflight_items": inflight_items,
                "scheduled_items": scheduled_items,
                "dead_letter_items": dead_letter_items,
            }

            ensure(
                enqueue_payload["backend_kind"] == contract["required_backend_kind"],
                "worker enqueue manifest backend_kind mismatch",
                failures,
            )
            ensure(bool(run_report), "worker retry run did not emit a parseable report", failures)
            ensure(bool(ack_run_report), "worker ack run did not emit a parseable report", failures)
            for field, expected in contract["required_retry_counts"].items():
                ensure(
                    queue_state.get(field) == expected,
                    f"queue_state.{field} expected {expected}, got {queue_state.get(field)}",
                    failures,
                )
            for field, expected in contract["required_retry_report"].items():
                actual = run_report.get(field, 0) + ack_run_report.get(field, 0)
                ensure(
                    actual == expected,
                    f"aggregated worker reports for {field} expected {expected}, got {actual}",
                    failures,
                )
            ensure(len(pending_items) == 0, "pending queue should be empty after retry/ack drain", failures)
            ensure(len(inflight_items) == 0, "inflight queue should be empty after retry/ack drain", failures)
            ensure(len(scheduled_items) == 0, "scheduled queue should be empty after retry/ack drain", failures)
            ensure(len(dead_letter_items) == 0, "dead letter queue should remain empty for retry/ack path", failures)

        report["all_passed"] = not failures
        report["failures"] = failures
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "ok" if report["all_passed"] else "failed",
                    "output": str(output_path),
                    "all_passed": report["all_passed"],
                },
                separators=(",", ":"),
            )
        )
        return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
