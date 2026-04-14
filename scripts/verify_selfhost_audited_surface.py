#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the audited self-host executable surface with frontend-audit and mainline-ci."
    )
    parser.add_argument("--enkai-bin", required=True, help="Path to enkai executable")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument(
        "--contract",
        default="enkai/contracts/selfhost_audited_surface_v3_1_1.json",
        help="Path to the audited surface contract JSON",
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


def subprocess_env(enkai_bin: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    path_entries = [str(enkai_bin.parent), str(enkai_bin.parent / "deps")]
    current_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_entries + ([current_path] if current_path else []))
    return env


def run_command(command: list[str], cwd: pathlib.Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def materialized_destination(rel_path: pathlib.Path) -> pathlib.Path:
    parts = rel_path.parts
    if len(parts) >= 2 and parts[0] == "examples":
        return pathlib.Path(*parts[1:])
    if len(parts) >= 4 and parts[:4] == ("enkai", "tools", "bootstrap", "full_frontier_corpus"):
        return pathlib.Path("full_frontier_corpus", *parts[4:])
    if len(parts) >= 4 and parts[:4] == ("enkai", "tools", "bootstrap", "selfhost_corpus"):
        return pathlib.Path("selfhost_corpus", *parts[4:])
    return rel_path


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.workspace).resolve()
    enkai_bin = pathlib.Path(args.enkai_bin).resolve()
    contract_path = (root / args.contract).resolve() if not pathlib.Path(args.contract).is_absolute() else pathlib.Path(args.contract)
    output_path = (root / args.output).resolve() if not pathlib.Path(args.output).is_absolute() else pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    contract = read_json(contract_path)
    files = contract.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("contract files must be a non-empty list")

    triage_dir_value = contract.get("triage_dir", "artifacts/selfhost/audited_surface")
    triage_dir = (root / triage_dir_value).resolve() if not pathlib.Path(triage_dir_value).is_absolute() else pathlib.Path(triage_dir_value)
    triage_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="enkai_audited_surface_") as tmpdir_raw:
        corpus_root = pathlib.Path(tmpdir_raw)
        for rel in files:
            rel_path = pathlib.Path(str(rel))
            src = (root / rel_path).resolve()
            if not src.is_file():
                raise RuntimeError(f"contract file missing: {rel_path}")
            dst = corpus_root / materialized_destination(rel_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        env = subprocess_env(enkai_bin)
        frontend_command = [
            str(enkai_bin),
            "litec",
            "frontend-audit",
            str(corpus_root),
            "--triage-dir",
            str(triage_dir),
            "--require-full-support",
        ]
        frontend = run_command(frontend_command, root, env)

        mainline_command = [
            str(enkai_bin),
            "litec",
            "mainline-ci",
            str(corpus_root),
            "--triage-dir",
            str(triage_dir),
        ]
        mainline = run_command(mainline_command, root, env)

    frontend_report = read_json(triage_dir / "litec_frontend_audit_report.json")
    replace_report = read_json(triage_dir / "litec_replace_check_report.json")
    selfhost_ci_report = read_json(triage_dir / "litec_selfhost_ci_report.json")
    mainline_report = read_json(triage_dir / "litec_mainline_ci_report.json")

    ok = (
        frontend.returncode == 0
        and mainline.returncode == 0
        and bool(frontend_report.get("full_support_ready", False))
        and int(frontend_report.get("frontier_gap_files", 0)) == 0
        and int(frontend_report.get("invalid_files", 0)) == 0
        and replace_report.get("status") == "ok"
        and bool(replace_report.get("stage2_stage3_fixed_point", False))
        and int(replace_report.get("files_failed", 0)) == 0
        and selfhost_ci_report.get("status") == "ok"
        and mainline_report.get("status") == "ok"
    )

    summary = {
        "schema_version": 1,
        "profile": str(contract.get("profile", "strict_selfhost_audited_surface")),
        "status": "ok" if ok else "failed",
        "contract": str(contract_path.relative_to(root)),
        "triage_dir": str(triage_dir.relative_to(root)),
        "files_total": len(files),
        "frontend_command": frontend_command,
        "frontend_exit_code": frontend.returncode,
        "frontend_stdout": frontend.stdout,
        "frontend_stderr": frontend.stderr,
        "mainline_command": mainline_command,
        "mainline_exit_code": mainline.returncode,
        "mainline_stdout": mainline.stdout,
        "mainline_stderr": mainline.stderr,
        "frontend_full_support_ready": bool(frontend_report.get("full_support_ready", False)),
        "frontend_frontier_gap_files": int(frontend_report.get("frontier_gap_files", 0)),
        "frontend_invalid_files": int(frontend_report.get("invalid_files", 0)),
        "replace_check_status": replace_report.get("status"),
        "replace_check_stage2_stage3_fixed_point": bool(
            replace_report.get("stage2_stage3_fixed_point", False)
        ),
        "replace_check_files_failed": int(replace_report.get("files_failed", 0)),
        "selfhost_ci_status": selfhost_ci_report.get("status"),
        "mainline_status": mainline_report.get("status"),
        "mainline_release_default_build_path": mainline_report.get("release_default_build_path"),
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(output_path.relative_to(root)),
                "files_total": summary["files_total"],
                "stage2_stage3_fixed_point": summary["replace_check_stage2_stage3_fixed_point"],
            },
            separators=(",", ":"),
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
