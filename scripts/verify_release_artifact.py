#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def parse_checksum_file(path: pathlib.Path) -> tuple[str, str]:
    text = path.read_text(encoding="ascii").strip()
    parts = text.split()
    if len(parts) < 2:
        raise RuntimeError(f"invalid checksum file format: {path}")
    return parts[0].strip().lower(), parts[-1].strip()


def verify_checksum(archive: pathlib.Path, checksum_file: pathlib.Path) -> str:
    expected_hash, expected_name = parse_checksum_file(checksum_file)
    if expected_name != archive.name:
        raise RuntimeError(
            f"checksum filename mismatch: expected {archive.name}, file records {expected_name}"
        )
    actual_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"checksum mismatch for {archive.name}: expected {expected_hash}, got {actual_hash}"
        )
    return actual_hash


def archive_entries(archive: pathlib.Path) -> list[str]:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            return sorted(info.filename for info in zf.infolist() if not info.is_dir())
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            return sorted(member.name for member in tf.getmembers() if member.isfile())
    raise RuntimeError(f"unsupported archive extension: {archive}")


def read_archive_text(archive: pathlib.Path, entry: str) -> str:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            return zf.read(entry).decode("utf-8-sig")
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            member = tf.getmember(entry)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"archive entry is not readable: {entry}")
            return extracted.read().decode("utf-8-sig")
    raise RuntimeError(f"unsupported archive extension: {archive}")


def extract_archive(archive: pathlib.Path, out_dir: pathlib.Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
        return
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
        return
    raise RuntimeError(f"unsupported archive extension: {archive}")


def validate_layout(entries: list[str], exe_name: str) -> None:
    required = {exe_name, "README.txt", "examples/hello/main.enk", "bundle_manifest.json"}
    missing = sorted(item for item in required if item not in entries)
    if missing:
        raise RuntimeError(f"archive missing required entries: {', '.join(missing)}")
    if not any(name.startswith("std/") for name in entries):
        raise RuntimeError("archive missing std/ contents")


def validate_bundle_manifest(
    archive: pathlib.Path,
    entries: list[str],
    version: str,
    target_os: str,
    arch: str,
    exe_name: str,
) -> dict[str, object]:
    manifest = json.loads(read_archive_text(archive, "bundle_manifest.json"))
    failures: list[str] = []
    expected = {
        "version": version,
        "target_os": target_os,
        "arch": arch,
        "entrypoint": exe_name,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            failures.append(f"bundle_manifest.{key} expected {value!r} got {manifest.get(key)!r}")
    required_paths = [str(item) for item in manifest.get("required_paths", [])]
    entry_set = set(entries)
    missing_required_paths = sorted(
        item
        for item in required_paths
        if item not in entry_set and not any(name.startswith(f"{item}/") for name in entries)
    )
    if missing_required_paths:
        failures.append(
            "bundle_manifest.required_paths missing from archive: "
            + ", ".join(missing_required_paths)
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "version": manifest.get("version"),
        "target_os": manifest.get("target_os"),
        "arch": manifest.get("arch"),
        "archive_format": manifest.get("archive_format"),
        "entrypoint": manifest.get("entrypoint"),
        "required_paths": required_paths,
        "native_payloads": manifest.get("native_payloads", []),
        "selfhost_entrypoints": manifest.get("selfhost_entrypoints", []),
        "missing_required_paths": missing_required_paths,
    }


def smoke_test(extract_dir: pathlib.Path, exe_name: str) -> None:
    exe_path = extract_dir / exe_name
    if not exe_path.exists():
        raise RuntimeError(f"smoke test executable not found: {exe_path}")
    subprocess.run([str(exe_path), "--version"], cwd=extract_dir, check=True)
    subprocess.run(
        [str(exe_path), "run", "examples/hello/main.enk"],
        cwd=extract_dir,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify release archive checksum/layout/smoke.")
    parser.add_argument("--archive", required=True, help="Path to archive (.tar.gz or .zip)")
    parser.add_argument(
        "--checksum-file",
        help="Path to checksum file. Defaults to <archive>.sha256",
    )
    parser.add_argument(
        "--target-os",
        required=True,
        choices=["linux", "macos", "windows"],
        help="Determines expected executable name",
    )
    parser.add_argument("--version", required=True, help="Expected Enkai semantic version")
    parser.add_argument("--arch", required=True, help="Expected architecture label")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run extracted smoke commands: --version + run examples/hello/main.enk",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]

    archive = pathlib.Path(args.archive)
    if not archive.is_absolute():
        archive = (root / archive).resolve()
    if not archive.is_file():
        raise RuntimeError(f"archive not found: {archive}")

    checksum_file = pathlib.Path(args.checksum_file) if args.checksum_file else pathlib.Path(f"{archive}.sha256")
    if not checksum_file.is_absolute():
        checksum_file = (root / checksum_file).resolve()
    if not checksum_file.is_file():
        raise RuntimeError(f"checksum file not found: {checksum_file}")

    exe_name = "enkai.exe" if args.target_os == "windows" else "enkai"

    checksum = verify_checksum(archive, checksum_file)
    entries = archive_entries(archive)
    validate_layout(entries, exe_name)
    bundle_manifest = validate_bundle_manifest(
        archive, entries, args.version, args.target_os, args.arch, exe_name
    )

    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="enkai_release_extract_") as tmp_dir:
            extract_dir = pathlib.Path(tmp_dir)
            extract_archive(archive, extract_dir)
            smoke_test(extract_dir, exe_name)

    payload = {
        "status": "ok",
        "archive": str(archive.relative_to(root)),
        "sha256": checksum,
        "entries": len(entries),
        "bundle_manifest": bundle_manifest,
        "smoke": args.smoke,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover - script entrypoint
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
