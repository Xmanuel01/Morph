#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeterminismSuite:
    id: str
    path: str


SUITES = (
    DeterminismSuite("event_queue", "examples/validation/determinism_event_queue.enk"),
    DeterminismSuite("sim_coroutines", "examples/validation/determinism_sim_coroutines.enk"),
    DeterminismSuite("sim_replay", "examples/validation/determinism_sim_replay.enk"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify repeated self-host runtime determinism for the audited simulation/event corpus."
    )
    parser.add_argument("--enkai-bin", required=True, help="Path to the enkai CLI binary.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--output", required=True, help="Path to write the verification report JSON.")
    parser.add_argument(
        "--triage-dir",
        default="artifacts/selfhost/runtime_determinism",
        help="Directory to write per-run runtime-audit reports.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated runs per suite.")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def entry_supported(entry: dict[str, Any]) -> bool:
    selfhost_runtime = entry.get("selfhost_runtime")
    if isinstance(selfhost_runtime, dict):
        return selfhost_runtime.get("supported") is True
    return selfhost_runtime is True


def run_runtime_audit(
    enkai_bin: Path,
    workspace: Path,
    target: Path,
    triage_dir: Path,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    triage_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(enkai_bin),
        "litec",
        "runtime-audit",
        str(target),
        "--triage-dir",
        str(triage_dir),
        "--require-full-support",
    ]
    completed = subprocess.run(
        command,
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    report_path = triage_dir / "litec_runtime_audit_report.json"
    if not report_path.is_file():
        raise RuntimeError(
            f"runtime-audit did not produce {report_path} (exit={completed.returncode})"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return completed, report


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    enkai_bin = Path(args.enkai_bin).resolve()
    output_path = Path(args.output).resolve()
    triage_root = Path(args.triage_dir).resolve()
    triage_root.mkdir(parents=True, exist_ok=True)

    all_passed = True
    suite_reports: list[dict[str, Any]] = []
    suite_state = {
        suite.id: {
            "suite": suite,
            "run_reports": [],
            "selfhost_hashes": [],
            "rust_hashes": [],
            "stage2_hashes": [],
            "suite_ok": True,
        }
        for suite in SUITES
    }

    for suite in SUITES:
        target = (workspace / suite.path).resolve()
        if not target.is_file():
            raise RuntimeError(f"missing determinism suite source: {target}")
        state = suite_state[suite.id]
        for run_idx in range(args.runs):
            run_dir = triage_root / suite.id / f"run_{run_idx + 1:02d}"
            completed, report = run_runtime_audit(enkai_bin, workspace, target, run_dir)
            if len(report.get("entries", [])) != 1:
                raise RuntimeError(
                    f"suite {suite.id} expected a single runtime-audit entry, found {len(report.get('entries', []))}"
                )
            entry = report["entries"][0]

            selfhost_result = entry.get("selfhost_result")
            rust_result = entry.get("rust_result")
            parity_ok = selfhost_result == rust_result
            support_ok = (
                completed.returncode == 0
                and report.get("status") == "ok"
                and report.get("runtime_gap_files") == 0
                and report.get("full_support_ready") is True
                and report.get("stage1_stage2_fixed_point") is True
                and entry.get("status") == "pass"
                and entry_supported(entry)
            )
            if not support_ok:
                state["suite_ok"] = False

            selfhost_hash = value_hash(selfhost_result)
            rust_hash = value_hash(rust_result)
            stage2_hash = entry.get("stage2", {}).get("sha256", "")
            state["selfhost_hashes"].append(selfhost_hash)
            state["rust_hashes"].append(rust_hash)
            state["stage2_hashes"].append(stage2_hash)
            state["run_reports"].append(
                {
                    "run": run_idx + 1,
                    "exit_code": completed.returncode,
                    "report_status": report.get("status"),
                    "entry_status": entry.get("status"),
                    "support_ok": support_ok,
                    "parity_ok": parity_ok,
                    "selfhost_hash": selfhost_hash,
                    "rust_hash": rust_hash,
                    "stage2_hash": stage2_hash,
                    "report_path": str((run_dir / "litec_runtime_audit_report.json").resolve()),
                }
            )

    for state in suite_state.values():
        suite = state["suite"]
        selfhost_hashes = state["selfhost_hashes"]
        rust_hashes = state["rust_hashes"]
        stage2_hashes = state["stage2_hashes"]
        deterministic_selfhost = len(set(selfhost_hashes)) == 1
        deterministic_rust = len(set(rust_hashes)) == 1
        deterministic_stage2 = len(set(stage2_hashes)) == 1
        suite_passed = (
            state["suite_ok"] and deterministic_selfhost and deterministic_rust and deterministic_stage2
        )
        if not suite_passed:
            all_passed = False

        suite_reports.append(
            {
                "id": suite.id,
                "target": suite.path,
                "runs": args.runs,
                "passed": suite_passed,
                "selfhost_hashes_identical": deterministic_selfhost,
                "rust_hashes_identical": deterministic_rust,
                "stage2_hashes_identical": deterministic_stage2,
                "selfhost_hash": selfhost_hashes[0] if selfhost_hashes else "",
                "rust_hash": rust_hashes[0] if rust_hashes else "",
                "stage2_hash": stage2_hashes[0] if stage2_hashes else "",
                "run_reports": state["run_reports"],
            }
        )

    payload = {
        "schema_version": 1,
        "profile": "strict_selfhost_runtime_determinism",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "enkai_bin": str(enkai_bin),
        "workspace": str(workspace),
        "runs_per_suite": args.runs,
        "all_passed": all_passed,
        "suite_count": len(SUITES),
        "suites": suite_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok" if all_passed else "failed", "output": str(output_path), "all_passed": all_passed}))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
