#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import shutil

import package_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove actual install, upgrade, and uninstall flows for the current host installer."
    )
    parser.add_argument("--enkai-bin", required=True, help="Path to the built enkai executable")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument(
        "--contract",
        default="enkai/contracts/install_flow_v3_3_0.json",
        help="Path to the install flow contract",
    )
    parser.add_argument(
        "--native",
        action="append",
        default=[],
        help="Optional native library payloads to stage into the bundle",
    )
    parser.add_argument(
        "--release-artifact-output",
        help="Optional JSON output path for deterministic release-archive verification",
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
    payloads: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                payloads.append(resolved)
    return payloads


def load_json(path_value: str, root: pathlib.Path) -> dict[str, object]:
    path = pathlib.Path(path_value)
    if not path.is_absolute():
        path = (root / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_contract(path_value: str, root: pathlib.Path, target_os: str) -> tuple[dict[str, object], str]:
    contract = load_json(path_value, root)
    os_contracts = contract.get("os_contracts")
    if isinstance(os_contracts, dict):
        selected = os_contracts.get(target_os)
        if not selected:
            raise RuntimeError(f"no install-flow contract configured for target OS: {target_os}")
        return load_json(str(selected), root), str(pathlib.Path(str(selected)).as_posix())
    return contract, str(pathlib.Path(path_value).as_posix())


def write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def parse_version(stdout: str) -> str | None:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Enkai v"):
            token = line.split(" ", 2)[1]
            return token.lstrip("v")
    return None


def invoke_installer(
    target_os: str,
    root: pathlib.Path,
    bundle_path: pathlib.Path,
    install_dir: pathlib.Path,
    uninstall: bool,
) -> dict[str, object]:
    if target_os == "windows":
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "install" / "install.ps1"),
            "-InstallDir",
            str(install_dir),
            "-NoPathUpdate",
        ]
        if uninstall:
            command.append("-Uninstall")
        else:
            command.extend(["-BundlePath", str(bundle_path), "-Version", "v3.3.0"])
    else:
        command = [
            "bash",
            str(root / "install" / "install.sh"),
            "--install-dir",
            str(install_dir),
            "--no-path-update",
        ]
        if uninstall:
            command.append("--uninstall")
        else:
            command.extend(["--bundle-path", str(bundle_path), "--version", "v3.3.0"])
    env = os.environ.copy()
    return run_checked(command, root, env)


def collect_runtime_proof(install_dir: pathlib.Path, target_os: str) -> dict[str, object]:
    exe_name = "enkai.exe" if target_os == "windows" else "enkai"
    binary = install_dir / exe_name
    diagnostics_path = install_dir / "artifacts" / "install_diagnostics.json"
    backend_path = install_dir / "artifacts" / "run_backend_hello.json"
    env, scrubbed_entries = scrubbed_env(install_dir)
    env["ENKAI_RUN_BACKEND_REPORT"] = str(backend_path)
    version_run = run_checked([str(binary), "--version"], install_dir, env)
    diagnostics_run = run_checked(
        [str(binary), "install-diagnostics", "--json", "--output", str(diagnostics_path)],
        install_dir,
        env,
    )
    hello_run = run_checked(
        [str(binary), "run", str(install_dir / "examples" / "hello" / "main.enk")],
        install_dir,
        env,
    )
    env.pop("ENKAI_RUN_BACKEND_REPORT", None)
    install_manifest = read_json_if_exists(install_dir / "install_manifest.json")
    parsed_version = parse_version(version_run["stdout"])
    return {
        "version_check": version_run,
        "diagnostics_check": diagnostics_run,
        "hello_check": hello_run,
        "install_diagnostics": read_json_if_exists(diagnostics_path),
        "hello_runtime_backend": read_json_if_exists(backend_path),
        "install_manifest": install_manifest,
        "scrubbed_path_entries": scrubbed_entries,
        "parsed_version": parsed_version,
    }


def run_entrypoint_proofs(
    install_dir: pathlib.Path,
    target_os: str,
    required_backend: object,
) -> dict[str, object]:
    exe_name = "enkai.exe" if target_os == "windows" else "enkai"
    binary = install_dir / exe_name
    project = prepare_entrypoint_project(install_dir)
    reports = install_dir / "artifacts" / "entrypoint_reports"
    reports.mkdir(parents=True, exist_ok=True)
    proof_env, _ = scrubbed_env(install_dir)

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


def paths_exist(root: pathlib.Path, rel_paths: list[object]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for value in rel_paths:
        rel = str(value)
        if (root / rel).exists():
            present.append(rel)
        else:
            missing.append(rel)
    return present, missing


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    output_path = pathlib.Path(args.output)
    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()

    target_os = detect_target_os()
    contract, selected_contract_path = resolve_contract(args.contract, root, target_os)
    exe_name = "enkai.exe" if target_os == "windows" else "enkai"
    bin_path = pathlib.Path(args.enkai_bin)
    if not bin_path.is_absolute():
        bin_path = (root / bin_path).resolve()
    native_paths: list[pathlib.Path] = []
    for value in args.native:
        candidate = pathlib.Path(value)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        native_paths.append(candidate)
    if not native_paths:
        native_paths = auto_detect_native_payloads(root, bin_path, target_os)

    release_artifact_proof: dict[str, object] | None = None
    if args.release_artifact_output:
        release_output = pathlib.Path(args.release_artifact_output)
        if not release_output.is_absolute():
            release_output = (root / release_output).resolve()
        dist_dir = release_output.parent / "dist"
        archive_format = "zip" if target_os == "windows" else "tar.gz"
        package_command = [
            sys.executable,
            str(root / "scripts" / "package_release.py"),
            "--target-os",
            target_os,
            "--arch",
            "x86_64",
            "--archive-format",
            archive_format,
            "--bin",
            str(bin_path),
            "--out-dir",
            str(dist_dir),
            "--check-deterministic",
        ]
        for native_path in native_paths:
            package_command.extend(["--native", str(native_path)])
        package_run = run_checked(package_command, root, os.environ.copy())
        archive_name = package_release.expected_archive_name(
            package_release.read_version_from_cargo(root), target_os, "x86_64", archive_format
        )
        verify_command = [
            sys.executable,
            str(root / "scripts" / "verify_release_artifact.py"),
            "--archive",
            str(dist_dir / archive_name),
            "--target-os",
            target_os,
            "--arch",
            "x86_64",
            "--version",
            package_release.read_version_from_cargo(root),
            "--smoke",
        ]
        verify_run = run_checked(verify_command, root, os.environ.copy()) if package_run["ok"] else {}
        release_artifact_proof = {
            "package": package_run,
            "verify": verify_run,
            "all_passed": bool(package_run["ok"] and verify_run.get("ok") is True),
        }
        write_json(release_output, release_artifact_proof)

    with tempfile.TemporaryDirectory(prefix="enkai_install_flow_") as tmp_dir:
        temp_root = pathlib.Path(tmp_dir)
        stage = temp_root / "stage"
        install_dir = temp_root / "install_root"
        package_release.stage_release_tree(root, stage, exe_name, bin_path, native_paths)
        version = package_release.read_version_from_cargo(root)
        archive_format = "zip" if target_os == "windows" else "tar.gz"
        package_release.write_bundle_manifest(
            stage, version, target_os, "x86_64", archive_format, exe_name, native_paths
        )
        package_release.verify_required_layout(stage, exe_name)

        archive_path = temp_root / (
            "enkai-v3.3.0-windows-x86_64.zip" if target_os == "windows" else "enkai-v3.3.0-host.tar.gz"
        )
        if target_os == "windows":
            first_bytes = package_release.build_zip_bytes(stage, exe_name)
            second_bytes = package_release.build_zip_bytes(stage, exe_name)
        else:
            first_bytes = package_release.build_tar_gz_bytes(stage, exe_name)
            second_bytes = package_release.build_tar_gz_bytes(stage, exe_name)
        deterministic_archive = package_release.sha256_bytes(first_bytes) == package_release.sha256_bytes(second_bytes)
        if target_os == "windows":
            archive_path.write_bytes(first_bytes)
        else:
            archive_path.write_bytes(first_bytes)

        install_step = invoke_installer(target_os, root, archive_path, install_dir, uninstall=False)
        install_proof = collect_runtime_proof(install_dir, target_os) if install_step["ok"] else {}
        required_backend = contract.get("required_runtime_backend")
        install_entrypoint_proofs = (
            run_entrypoint_proofs(install_dir, target_os, required_backend) if install_step["ok"] else {}
        )
        required_post = list(contract.get("required_post_install_paths", []))
        present_after_install, missing_after_install = paths_exist(install_dir, required_post)

        stale_std = install_dir / "std" / "stale_should_disappear.enk"
        stale_examples = install_dir / "examples" / "stale_should_disappear.txt"
        stale_std.parent.mkdir(parents=True, exist_ok=True)
        stale_examples.parent.mkdir(parents=True, exist_ok=True)
        stale_std.write_text("stale", encoding="utf-8")
        stale_examples.write_text("stale", encoding="utf-8")

        upgrade_step = invoke_installer(target_os, root, archive_path, install_dir, uninstall=False)
        upgrade_proof = collect_runtime_proof(install_dir, target_os) if upgrade_step["ok"] else {}
        upgrade_entrypoint_proofs = (
            run_entrypoint_proofs(install_dir, target_os, required_backend) if upgrade_step["ok"] else {}
        )
        stale_removed = (not stale_std.exists()) and (not stale_examples.exists())

        uninstall_step = invoke_installer(target_os, root, archive_path, install_dir, uninstall=True)
        _, missing_after_uninstall = paths_exist(
            install_dir, list(contract.get("required_uninstall_absent_paths", []))
        )
        uninstall_absent_ok = len(missing_after_uninstall) == len(
            list(contract.get("required_uninstall_absent_paths", []))
        )

        require_entrypoint_proofs = bool(contract.get("require_entrypoint_execution_proofs", False))
        require_hidden = bool(contract.get("require_rust_toolchain_hidden_in_runtime_proof", False))
        require_manifest = bool(contract.get("require_install_manifest", False))
        require_diagnostics_manifest = bool(
            contract.get("require_install_diagnostics_manifest_ok", False)
        )
        require_bundle_manifest = bool(contract.get("require_bundle_manifest_diagnostics", False))
        require_deterministic_archive = bool(contract.get("require_deterministic_archive", False))
        install_backend = None
        if install_proof.get("hello_runtime_backend"):
            install_backend = install_proof["hello_runtime_backend"].get("backend")
        upgrade_backend = None
        if upgrade_proof.get("hello_runtime_backend"):
            upgrade_backend = upgrade_proof["hello_runtime_backend"].get("backend")
        install_diag = install_proof.get("install_diagnostics") if isinstance(install_proof, dict) else None
        upgrade_diag = upgrade_proof.get("install_diagnostics") if isinstance(upgrade_proof, dict) else None
        install_hidden_ok = True
        upgrade_hidden_ok = True
        if require_hidden:
            install_hidden_ok = bool(
                install_diag is not None
                and install_diag.get("rust_toolchain_visible", {}).get("cargo") is False
                and install_diag.get("rust_toolchain_visible", {}).get("rustc") is False
            )
            upgrade_hidden_ok = bool(
                upgrade_diag is not None
                and upgrade_diag.get("rust_toolchain_visible", {}).get("cargo") is False
                and upgrade_diag.get("rust_toolchain_visible", {}).get("rustc") is False
            )
        install_manifest = install_proof.get("install_manifest") if isinstance(install_proof, dict) else None
        upgrade_manifest = upgrade_proof.get("install_manifest") if isinstance(upgrade_proof, dict) else None
        install_manifest_ok = True
        upgrade_manifest_ok = True
        if require_manifest:
            install_manifest_ok = bool(
                install_manifest is not None
                and install_manifest.get("installed_version") == install_proof.get("parsed_version")
            )
            upgrade_manifest_ok = bool(
                upgrade_manifest is not None
                and upgrade_manifest.get("installed_version") == upgrade_proof.get("parsed_version")
            )
        install_diagnostics_manifest_ok = True
        upgrade_diagnostics_manifest_ok = True
        if require_diagnostics_manifest:
            install_manifest_diag = (
                install_diag.get("install_manifest", {}) if install_diag is not None else {}
            )
            upgrade_manifest_diag = (
                upgrade_diag.get("install_manifest", {}) if upgrade_diag is not None else {}
            )
            install_diagnostics_manifest_ok = bool(
                install_manifest_diag.get("present") is True
                and install_manifest_diag.get("parse_ok") is True
                and install_manifest_diag.get("version_matches_cli") is True
                and not install_manifest_diag.get("missing_managed_entries", [])
            )
            upgrade_diagnostics_manifest_ok = bool(
                upgrade_manifest_diag.get("present") is True
                and upgrade_manifest_diag.get("parse_ok") is True
                and upgrade_manifest_diag.get("version_matches_cli") is True
                and not upgrade_manifest_diag.get("missing_managed_entries", [])
            )
        install_bundle_manifest_ok = True
        upgrade_bundle_manifest_ok = True
        if require_bundle_manifest:
            install_bundle_manifest_ok = bundle_manifest_diagnostics_ok(install_diag)
            upgrade_bundle_manifest_ok = bundle_manifest_diagnostics_ok(upgrade_diag)
        install_entrypoint_proofs_ok = (not require_entrypoint_proofs) or bool(
            install_entrypoint_proofs.get("all_passed")
        )
        upgrade_entrypoint_proofs_ok = (not require_entrypoint_proofs) or bool(
            upgrade_entrypoint_proofs.get("all_passed")
        )

        report = {
            "schema_version": 1,
            "contract": selected_contract_path,
            "contract_version": contract.get("contract_version"),
            "profile": contract.get("profile", "install_flow_proof"),
            "target_os": target_os,
            "release_artifact_proof": None
            if release_artifact_proof is None
            else str(pathlib.Path(args.release_artifact_output).as_posix()),
            "bundle_archive": str(archive_path),
            "archive_determinism": {
                "format": archive_format,
                "first_sha256": package_release.sha256_bytes(first_bytes),
                "second_sha256": package_release.sha256_bytes(second_bytes),
                "deterministic": deterministic_archive,
                "required": require_deterministic_archive,
            },
            "install_dir": str(install_dir),
            "phases": {
                "install": {
                    "command": install_step,
                    "runtime_proof": install_proof,
                    "entrypoint_proofs": install_entrypoint_proofs,
                    "present_paths": present_after_install,
                    "missing_paths": missing_after_install,
                    "runtime_backend_ok": install_backend == required_backend,
                    "rust_toolchain_hidden_ok": install_hidden_ok,
                    "install_manifest_ok": install_manifest_ok,
                    "install_diagnostics_manifest_ok": install_diagnostics_manifest_ok,
                    "bundle_manifest_ok": install_bundle_manifest_ok,
                    "entrypoint_proofs_ok": install_entrypoint_proofs_ok,
                },
                "upgrade": {
                    "command": upgrade_step,
                    "runtime_proof": upgrade_proof,
                    "entrypoint_proofs": upgrade_entrypoint_proofs,
                    "stale_paths_removed": stale_removed,
                    "runtime_backend_ok": upgrade_backend == required_backend,
                    "rust_toolchain_hidden_ok": upgrade_hidden_ok,
                    "install_manifest_ok": upgrade_manifest_ok,
                    "install_diagnostics_manifest_ok": upgrade_diagnostics_manifest_ok,
                    "bundle_manifest_ok": upgrade_bundle_manifest_ok,
                    "entrypoint_proofs_ok": upgrade_entrypoint_proofs_ok,
                },
                "uninstall": {
                    "command": uninstall_step,
                    "remaining_forbidden_paths": [
                        item
                        for item in contract.get("required_uninstall_absent_paths", [])
                        if (install_dir / str(item)).exists()
                    ],
                    "all_forbidden_paths_absent": uninstall_absent_ok,
                },
            },
            "all_passed": bool(
                install_step["ok"]
                and (not require_deterministic_archive or deterministic_archive)
                and not missing_after_install
                and install_backend == required_backend
                and install_hidden_ok
                and install_manifest_ok
                and install_diagnostics_manifest_ok
                and install_bundle_manifest_ok
                and install_entrypoint_proofs_ok
                and upgrade_step["ok"]
                and stale_removed
                and upgrade_backend == required_backend
                and upgrade_hidden_ok
                and upgrade_manifest_ok
                and upgrade_diagnostics_manifest_ok
                and upgrade_bundle_manifest_ok
                and upgrade_entrypoint_proofs_ok
                and uninstall_step["ok"]
                and uninstall_absent_ok
                and (release_artifact_proof is None or release_artifact_proof["all_passed"])
            ),
        }

        write_json(output_path, report)
        print(
            json.dumps(
                {
                    "status": "ok" if report["all_passed"] else "failed",
                    "output": str(output_path.relative_to(root)),
                },
                separators=(",", ":"),
            )
        )
        return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
