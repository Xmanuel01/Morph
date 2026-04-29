#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify self-host project entrypoints ownership slice."
    )
    parser.add_argument("--enkai-bin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--contract",
        default=str(
            Path(__file__).resolve().parents[1]
            / "enkai"
            / "contracts"
            / "selfhost_project_entrypoints_v3_3_0.json"
        ),
    )
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_backend(report_path: Path, key: str, expected: str) -> dict:
    payload = load_json(report_path)
    actual = payload.get(key)
    if actual != expected:
        raise AssertionError(f"{report_path.name}: expected {key}={expected}, got {actual}")
    return payload


def main() -> int:
    args = parse_args()
    enkai_bin = Path(args.enkai_bin).resolve()
    output_path = Path(args.output).resolve()
    contract = load_json(Path(args.contract).resolve())
    expected_backend = contract["required_backend"]
    build_cache_backend = contract["require_build_cache_backend"]

    workspace = Path(tempfile.mkdtemp(prefix="enkai_project_entrypoints_"))
    try:
        project = workspace / "demo"
        src = project / "src"
        tests = project / "tests"
        src.mkdir(parents=True)
        tests.mkdir(parents=True)
        (project / "enkai.toml").write_text(
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
            encoding="utf-8",
        )
        program = "fn main() -> Int ::\n    return 0\n::\n"
        (src / "main.enk").write_text(program, encoding="utf-8")
        (tests / "main.enk").write_text(program, encoding="utf-8")

        env = os.environ.copy()
        report_dir = workspace / "reports"
        report_dir.mkdir()
        run_report = report_dir / "run.json"
        check_report = report_dir / "check.json"
        build_report = report_dir / "build.json"
        test_report = report_dir / "test.json"
        env["ENKAI_RUN_BACKEND_REPORT"] = str(run_report)
        env["ENKAI_CHECK_BACKEND_REPORT"] = str(check_report)
        env["ENKAI_BUILD_BACKEND_REPORT"] = str(build_report)
        env["ENKAI_TEST_BACKEND_REPORT"] = str(test_report)

        results: dict[str, object] = {"contract": str(Path(args.contract).resolve())}

        check_proc = run([str(enkai_bin), "check", str(project)], env=env, cwd=project)
        if check_proc.returncode != 0:
            raise AssertionError(f"check failed: {check_proc.stderr or check_proc.stdout}")
        results["check"] = assert_backend(check_report, "backend", expected_backend)

        build_proc = run([str(enkai_bin), "build", str(project)], env=env, cwd=project)
        if build_proc.returncode != 0:
            raise AssertionError(f"build failed: {build_proc.stderr or build_proc.stdout}")
        results["build"] = assert_backend(build_report, "backend", expected_backend)

        build_meta = load_json(project / "target" / "enkai" / "build.json")
        if build_meta.get("compiler_backend") != build_cache_backend:
            raise AssertionError(
                f"build cache backend expected {build_cache_backend}, got {build_meta.get('compiler_backend')}"
            )
        results["build_cache"] = build_meta

        if contract.get("require_lockfile"):
            lockfile = project / "enkai.lock"
            if not lockfile.is_file():
                raise AssertionError("expected enkai.lock to be written")
            results["lockfile"] = {"path": str(lockfile), "present": True}

        run_proc = run([str(enkai_bin), "run", str(project)], env=env, cwd=project)
        if run_proc.returncode != 0:
            raise AssertionError(f"run failed: {run_proc.stderr or run_proc.stdout}")
        results["run"] = assert_backend(run_report, "backend", expected_backend)

        test_proc = run([str(enkai_bin), "test", str(project)], env=env, cwd=project)
        if test_proc.returncode != 0:
            raise AssertionError(f"test failed: {test_proc.stderr or test_proc.stdout}")
        test_payload = load_json(test_report)
        entries = test_payload.get("entries", [])
        if not entries:
            raise AssertionError("test backend report contained no entries")
        for entry in entries:
            if entry.get("compiler_backend") != expected_backend:
                raise AssertionError(
                    f"test compiler backend expected {expected_backend}, got {entry.get('compiler_backend')}"
                )
            if entry.get("runtime_backend") != expected_backend:
                raise AssertionError(
                    f"test runtime backend expected {expected_backend}, got {entry.get('runtime_backend')}"
                )
            if entry.get("status") != "pass":
                raise AssertionError(f"test entry failed: {entry}")
        results["test"] = test_payload

        results["all_passed"] = True
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "ok", "output": str(output_path)}, separators=(",", ":")))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {"all_passed": False, "error": str(exc), "contract": str(Path(args.contract).resolve())}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "error", "output": str(output_path), "error": str(exc)}, separators=(",", ":")))
        return 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
