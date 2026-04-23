#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import package_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a portable Enkai bundle, smoke it on a clean-ish environment, and emit closure evidence."
    )
    parser.add_argument(
        "--enkai-bin",
        required=True,
        help="Path to the built enkai executable to stage into the bundle",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the install smoke and closure reports will be written",
    )
    parser.add_argument(
        "--target-os",
        choices=["windows", "linux", "macos"],
        help="Bundle target OS label (defaults to the current host OS)",
    )
    parser.add_argument(
        "--native",
        action="append",
        default=[],
        help="Optional native library payloads to stage into the bundle",
    )
    parser.add_argument(
        "--bundle-contract",
        default="enkai/contracts/install_bundle_v3_2_1.json",
        help="Path to the versioned install-bundle contract",
    )
    parser.add_argument(
        "--closure-contract",
        default="enkai/contracts/zero_rust_closure_v3_2_1.json",
        help="Path to the versioned zero-Rust closure contract",
    )
    return parser.parse_args()


def detect_target_os() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def native_library_name(target_os: str) -> str:
    if target_os == "windows":
        return "enkai_native.dll"
    if target_os == "macos":
        return "libenkai_native.dylib"
    return "libenkai_native.so"


def scrubbed_path_entries(path_value: str) -> list[str]:
    entries: list[str] = []
    for raw in path_value.split(os.pathsep):
        entry = raw.strip()
        if not entry:
            continue
        lowered = entry.lower()
        if ".cargo" in lowered or "rustup" in lowered:
            continue
        entries.append(entry)
    return entries


def auto_detect_native_payloads(root: pathlib.Path, bin_path: pathlib.Path, target_os: str) -> list[pathlib.Path]:
    library_name = native_library_name(target_os)
    candidates: list[pathlib.Path] = []
    if bin_path.parent.is_dir():
        candidates.append(bin_path.parent / library_name)
        candidates.append(bin_path.parent / "deps" / library_name)
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("target"):
            candidates.append(child / "debug" / library_name)
            candidates.append(child / "debug" / "deps" / library_name)
    seen: set[pathlib.Path] = set()
    payloads: list[pathlib.Path] = []
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                payloads.append(resolved)
    return payloads


def scrubbed_env(install_dir: pathlib.Path) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    scrubbed_entries = scrubbed_path_entries(env.get("PATH", ""))
    install_entry = str(install_dir)
    if install_entry not in scrubbed_entries:
        scrubbed_entries.insert(0, install_entry)
    env["PATH"] = os.pathsep.join(scrubbed_entries)
    env["CARGO_HOME"] = str(install_dir / "_no_cargo_home")
    env["RUSTUP_HOME"] = str(install_dir / "_no_rustup_home")
    env.pop("RUSTC", None)
    env.pop("CARGO", None)
    return env, scrubbed_entries


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def staged_file_rows(stage: pathlib.Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(stage)
        rows.append(
            {
                "path": rel.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def run_checked(command: list[str], cwd: pathlib.Path, env: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "ok": completed.returncode == 0,
    }


def read_json_if_exists(path: pathlib.Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_entrypoint_project(root: pathlib.Path) -> pathlib.Path:
    project = root.parent / "entrypoint_project"
    if project.exists():
        shutil.rmtree(project)
    write_text(project / "enkai.toml", "name = 'entrypoint-proof'\nversion = '0.1.0'\n")
    write_text(project / "src" / "main.enk", "fn main() -> Int ::\n    return 0\n::\n\nmain()\n")
    write_text(project / "tests" / "main.enk", "fn main() -> Int ::\n    return 0\n::\n")
    return project


def run_entrypoint_proofs(
    install_dir: pathlib.Path,
    exe_name: str,
    env: dict[str, str],
    required_backend: object,
) -> dict[str, object]:
    binary = install_dir / exe_name
    project = prepare_entrypoint_project(install_dir)
    reports = install_dir / "artifacts" / "entrypoint_reports"
    reports.mkdir(parents=True, exist_ok=True)
    proof_env = env.copy()

    run_report = reports / "run.json"
    proof_env["ENKAI_RUN_BACKEND_REPORT"] = str(run_report)
    run_result = run_checked([str(binary), "run", str(install_dir / "examples" / "hello" / "main.enk")], install_dir, proof_env)
    proof_env.pop("ENKAI_RUN_BACKEND_REPORT", None)

    check_report = reports / "check.json"
    proof_env["ENKAI_CHECK_BACKEND_REPORT"] = str(check_report)
    check_result = run_checked([str(binary), "check", str(project / "src" / "main.enk")], install_dir, proof_env)
    proof_env.pop("ENKAI_CHECK_BACKEND_REPORT", None)

    build_report = reports / "build.json"
    proof_env["ENKAI_BUILD_BACKEND_REPORT"] = str(build_report)
    build_result = run_checked([str(binary), "build", str(project)], install_dir, proof_env)
    proof_env.pop("ENKAI_BUILD_BACKEND_REPORT", None)

    test_report = reports / "test.json"
    proof_env["ENKAI_TEST_BACKEND_REPORT"] = str(test_report)
    test_result = run_checked([str(binary), "test", str(project)], install_dir, proof_env)
    proof_env.pop("ENKAI_TEST_BACKEND_REPORT", None)

    run_backend = read_json_if_exists(run_report)
    check_backend = read_json_if_exists(check_report)
    build_backend = read_json_if_exists(build_report)
    test_backend = read_json_if_exists(test_report)
    required = str(required_backend)
    test_entries = [] if test_backend is None else test_backend.get("entries", [])
    test_backends_ok = bool(
        test_backend is not None
        and test_backend.get("failed") == 0
        and test_entries
        and all(
            row.get("compiler_backend") == required and row.get("runtime_backend") == required
            for row in test_entries
            if isinstance(row, dict)
        )
    )
    return {
        "run": {"result": run_result, "backend": run_backend, "backend_ok": run_backend is not None and run_backend.get("backend") == required},
        "check": {"result": check_result, "backend": check_backend, "backend_ok": check_backend is not None and check_backend.get("backend") == required},
        "build": {"result": build_result, "backend": build_backend, "backend_ok": build_backend is not None and build_backend.get("backend") == required},
        "test": {"result": test_result, "backend": test_backend, "backend_ok": test_backends_ok},
        "all_passed": bool(
            run_result["ok"]
            and check_result["ok"]
            and build_result["ok"]
            and test_result["ok"]
            and run_backend is not None
            and run_backend.get("backend") == required
            and check_backend is not None
            and check_backend.get("backend") == required
            and build_backend is not None
            and build_backend.get("backend") == required
            and test_backends_ok
        ),
    }


def write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path_value: str, root: pathlib.Path) -> dict[str, object]:
    path = pathlib.Path(path_value)
    if not path.is_absolute():
        path = (root / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def has_required_paths(stage: pathlib.Path, required_paths: list[object]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for item in required_paths:
        rel = str(item)
        if not (stage / rel).exists():
            missing.append(rel)
    return (not missing, missing)


def diagnostics_match_contract(
    diagnostics: dict[str, object] | None, contract: dict[str, object]
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if diagnostics is None:
        return False, ["missing install diagnostics report"]
    expected = contract.get("required_diagnostics", {})
    if not isinstance(expected, dict):
        return True, failures
    for key, value in expected.items():
        actual = diagnostics.get(key)
        if isinstance(value, bool):
            if actual is not value:
                failures.append(f"diagnostics.{key} expected {value!r} but got {actual!r}")
        elif isinstance(value, int):
            if not isinstance(actual, int) or actual < value:
                failures.append(f"diagnostics.{key} expected >= {value} but got {actual!r}")
        else:
            if actual != value:
                failures.append(f"diagnostics.{key} expected {value!r} but got {actual!r}")
    return (not failures, failures)


def bundle_manifest_diagnostics_ok(diagnostics: dict[str, object] | None) -> bool:
    if diagnostics is None:
        return False
    bundle_manifest = diagnostics.get("bundle_manifest", {})
    return bool(
        isinstance(bundle_manifest, dict)
        and bundle_manifest.get("present") is True
        and bundle_manifest.get("parse_ok") is True
        and bundle_manifest.get("version_matches_cli") is True
        and bundle_manifest.get("target_os_matches_host") is True
        and bundle_manifest.get("entrypoint_matches_host") is True
        and not bundle_manifest.get("missing_required_paths", [])
    )


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve() if not pathlib.Path(args.output_dir).is_absolute() else pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_contract = load_json(args.bundle_contract, root)
    closure_contract = load_json(args.closure_contract, root)

    target_os = args.target_os or detect_target_os()
    exe_name = "enkai.exe" if target_os == "windows" else "enkai"
    bin_path = pathlib.Path(args.enkai_bin)
    if not bin_path.is_absolute():
        bin_path = (root / bin_path).resolve()
    if not bin_path.is_file():
        raise RuntimeError(f"enkai binary not found: {bin_path}")

    native_paths: list[pathlib.Path] = []
    for value in args.native:
        candidate = pathlib.Path(value)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        native_paths.append(candidate)
    if not native_paths:
        native_paths = auto_detect_native_payloads(root, bin_path, target_os)

    with tempfile.TemporaryDirectory(prefix="enkai_install_bundle_") as tmp_dir:
        tmp_root = pathlib.Path(tmp_dir)
        install_dir = tmp_root / "install"
        package_release.stage_release_tree(root, install_dir, exe_name, bin_path, native_paths)
        version = package_release.read_version_from_cargo(root)
        archive_format = "zip" if target_os == "windows" else "tar.gz"
        package_release.write_bundle_manifest(
            install_dir, version, target_os, "x86_64", archive_format, exe_name, native_paths
        )
        package_release.verify_required_layout(install_dir, exe_name)

        env, scrubbed_entries = scrubbed_env(install_dir)
        rustc_visible = shutil.which("rustc", path=env["PATH"])
        cargo_visible = shutil.which("cargo", path=env["PATH"])
        version_run = run_checked([str(install_dir / exe_name), "--version"], install_dir, env)
        diagnostics_report = install_dir / "artifacts" / "install_diagnostics.json"
        diagnostics_run = run_checked(
            [
                str(install_dir / exe_name),
                "install-diagnostics",
                "--json",
                "--output",
                str(diagnostics_report),
            ],
            install_dir,
            env,
        )
        install_diagnostics = read_json_if_exists(diagnostics_report)
        hello_program = install_dir / "examples" / "hello" / "main.enk"
        hello_backend_report = install_dir / "artifacts" / "run_backend_hello.json"
        env["ENKAI_RUN_BACKEND_REPORT"] = str(hello_backend_report)
        hello_run = run_checked(
            [str(install_dir / exe_name), "run", str(hello_program)],
            install_dir,
            env,
        )
        env.pop("ENKAI_RUN_BACKEND_REPORT", None)
        hello_backend = read_json_if_exists(hello_backend_report)

        bundle_paths_ok, missing_bundle_paths = has_required_paths(
            install_dir, list(bundle_contract.get("required_paths", []))
        )
        diagnostics_ok, diagnostics_failures = diagnostics_match_contract(
            install_diagnostics, bundle_contract
        )
        closure_diagnostics_ok, closure_diagnostics_failures = diagnostics_match_contract(
            install_diagnostics, closure_contract
        )
        required_entrypoints = [
            str(item) for item in bundle_contract.get("required_entrypoints", [])
        ]
        actual_entrypoints = []
        if install_diagnostics is not None:
            actual_entrypoints = [
                str(item) for item in install_diagnostics.get("selfhost_entrypoints", [])
            ]
        missing_entrypoints = [
            item for item in required_entrypoints if item not in actual_entrypoints
        ]
        required_backend = bundle_contract.get("required_runtime_backend")
        closure_required_backend = closure_contract.get("required_runtime_backend")
        require_bundle_manifest = bool(bundle_contract.get("require_bundle_manifest_diagnostics", False))
        closure_require_bundle_manifest = bool(
            closure_contract.get("require_bundle_manifest_diagnostics", False)
        )
        require_entrypoint_proofs = bool(
            bundle_contract.get("require_entrypoint_execution_proofs", False)
        )
        bundle_manifest_ok = (not require_bundle_manifest) or bundle_manifest_diagnostics_ok(install_diagnostics)
        closure_bundle_manifest_ok = (
            not closure_require_bundle_manifest
        ) or bundle_manifest_diagnostics_ok(install_diagnostics)
        hello_backend_name = None if hello_backend is None else hello_backend.get("backend")
        entrypoint_proofs = run_entrypoint_proofs(install_dir, exe_name, env, required_backend)
        entrypoint_proofs_ok = (not require_entrypoint_proofs) or bool(entrypoint_proofs["all_passed"])

        manifest_report = {
            "schema_version": 1,
            "contract": str(pathlib.Path(args.bundle_contract).as_posix()),
            "contract_version": bundle_contract.get("contract_version"),
            "profile": bundle_contract.get("profile", "install_bundle_smoke"),
            "target_os": target_os,
            "bundle_root": str(install_dir),
            "required_paths": bundle_contract.get("required_paths", []),
            "missing_paths": missing_bundle_paths,
            "required_entrypoints": required_entrypoints,
            "missing_entrypoints": missing_entrypoints,
            "required_runtime_backend": required_backend,
            "observed_runtime_backend": hello_backend_name,
            "entrypoint_proofs": entrypoint_proofs,
            "native_payloads": [path.name for path in native_paths if path.is_file()],
            "staged_files": staged_file_rows(install_dir),
            "install_diagnostics": install_diagnostics,
            "contract_checks": {
                "paths_ok": bundle_paths_ok,
                "diagnostics_ok": diagnostics_ok,
                "bundle_manifest_ok": bundle_manifest_ok,
                "runtime_backend_ok": hello_backend_name == required_backend,
                "entrypoints_ok": not missing_entrypoints,
                "entrypoint_proofs_ok": entrypoint_proofs_ok,
            },
            "all_passed": bool(
                bundle_paths_ok
                and diagnostics_ok
                and bundle_manifest_ok
                and not missing_entrypoints
                and entrypoint_proofs_ok
                and hello_backend_name == required_backend
            ),
        }

        closure_report = {
            "schema_version": 1,
            "contract": str(pathlib.Path(args.closure_contract).as_posix()),
            "contract_version": closure_contract.get("contract_version"),
            "profile": closure_contract.get("profile", "zero_rust_closure"),
            "target_os": target_os,
            "bundle_root": str(install_dir),
            "staged_files": staged_file_rows(install_dir),
            "native_payloads": [path.name for path in native_paths if path.is_file()],
            "scrubbed_path_entries": scrubbed_entries,
            "rustc_visible_after_scrub": rustc_visible,
            "cargo_visible_after_scrub": cargo_visible,
            "rust_toolchain_required": bool(rustc_visible or cargo_visible),
            "bundle_operational_without_rust_toolchain": version_run["ok"] and hello_run["ok"],
            "install_diagnostics": install_diagnostics,
            "hello_runtime_backend": hello_backend,
            "entrypoint_proofs": entrypoint_proofs,
            "contract_checks": {
                "toolchain_hidden": not bool(rustc_visible or cargo_visible),
                "diagnostics_ok": closure_diagnostics_ok,
                "bundle_manifest_ok": closure_bundle_manifest_ok,
                "runtime_backend_ok": hello_backend_name == closure_required_backend,
                "operational_without_rust_ok": bool(version_run["ok"] and hello_run["ok"]),
                "entrypoint_proofs_ok": entrypoint_proofs_ok,
            },
        }
        install_report = {
            "schema_version": 1,
            "contract": str(pathlib.Path(args.bundle_contract).as_posix()),
            "contract_version": bundle_contract.get("contract_version"),
            "profile": bundle_contract.get("profile", "install_bundle_smoke"),
            "target_os": target_os,
            "bundle_root": str(install_dir),
            "version_check": version_run,
            "diagnostics_check": diagnostics_run,
            "install_diagnostics": install_diagnostics,
            "hello_check": hello_run,
            "hello_runtime_backend": hello_backend,
            "entrypoint_proofs": entrypoint_proofs,
            "manifest_report": "install_bundle_manifest.json",
            "closure_report": "zero_rust_closure.json",
            "contract_failures": {
                "missing_paths": missing_bundle_paths,
                "missing_entrypoints": missing_entrypoints,
                "diagnostics": diagnostics_failures,
                "closure_diagnostics": closure_diagnostics_failures,
            },
            "all_passed": bool(
                version_run["ok"]
                and diagnostics_run["ok"]
                and hello_run["ok"]
                and manifest_report["all_passed"]
                and not closure_report["rust_toolchain_required"]
                and closure_diagnostics_ok
                and closure_bundle_manifest_ok
                and entrypoint_proofs_ok
                and hello_backend_name == closure_required_backend
            ),
        }

        write_json(output_dir / "install_bundle_manifest.json", manifest_report)
        write_json(output_dir / "zero_rust_closure.json", closure_report)
        write_json(output_dir / "install_bundle_smoke.json", install_report)

    print(
        json.dumps(
            {
                "status": "ok" if install_report["all_passed"] else "failed",
                "output_dir": str(output_dir.relative_to(root)),
                "install_bundle_manifest": "install_bundle_manifest.json",
                "install_bundle_smoke": "install_bundle_smoke.json",
                "zero_rust_closure": "zero_rust_closure.json",
            },
            separators=(",", ":"),
        )
    )
    return 0 if install_report["all_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover - script entrypoint
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
