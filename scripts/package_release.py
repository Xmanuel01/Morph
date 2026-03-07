#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import zipfile


FIXED_MTIME = 0
FIXED_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ensure_file(path: pathlib.Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")


def copy_tree(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def stage_release_tree(
    root: pathlib.Path,
    stage: pathlib.Path,
    exe_name: str,
    bin_path: pathlib.Path,
    native_paths: list[pathlib.Path],
) -> None:
    ensure_file(bin_path, "binary")

    std_dir = root / "std"
    hello_dir = root / "examples" / "hello"
    readme = root / "install" / "README.txt"
    if not std_dir.is_dir():
        raise RuntimeError(f"missing std directory: {std_dir}")
    if not hello_dir.is_dir():
        raise RuntimeError(f"missing hello example directory: {hello_dir}")
    ensure_file(readme, "install README")

    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bin_path, stage / exe_name)
    copy_tree(std_dir, stage / "std")
    copy_tree(hello_dir, stage / "examples" / "hello")
    shutil.copy2(readme, stage / "README.txt")

    seen_native: set[str] = set()
    for candidate in native_paths:
        if not candidate:
            continue
        if not candidate.is_file():
            continue
        name = candidate.name
        if name in seen_native:
            continue
        seen_native.add(name)
        shutil.copy2(candidate, stage / name)


def iter_files(stage: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in stage.rglob("*"):
        if path.is_file():
            files.append(path.relative_to(stage))
    files.sort(key=lambda item: item.as_posix())
    return files


def is_executable(rel: pathlib.Path, exe_name: str) -> bool:
    return rel.as_posix() == exe_name


def build_tar_gz_bytes(stage: pathlib.Path, exe_name: str) -> bytes:
    out = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=FIXED_MTIME) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for rel in iter_files(stage):
                source = stage / rel
                data = source.read_bytes()
                info = tarfile.TarInfo(rel.as_posix())
                info.size = len(data)
                info.mode = 0o755 if is_executable(rel, exe_name) else 0o644
                info.mtime = FIXED_MTIME
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


def build_zip_bytes(stage: pathlib.Path, exe_name: str) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in iter_files(stage):
            source = stage / rel
            data = source.read_bytes()
            info = zipfile.ZipInfo(rel.as_posix(), date_time=FIXED_ZIP_DATE)
            mode = 0o755 if is_executable(rel, exe_name) else 0o644
            info.external_attr = (mode & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return out.getvalue()


def expected_archive_name(version: str, target_os: str, arch: str, archive_format: str) -> str:
    ext = "zip" if archive_format == "zip" else "tar.gz"
    return f"enkai-{version}-{target_os}-{arch}.{ext}"


def write_checksum(out_path: pathlib.Path) -> pathlib.Path:
    checksum = hashlib.sha256(out_path.read_bytes()).hexdigest()
    checksum_path = out_path.with_name(f"{out_path.name}.sha256")
    checksum_path.write_text(f"{checksum}  {out_path.name}\n", encoding="ascii")
    return checksum_path


def verify_required_layout(stage: pathlib.Path, exe_name: str) -> None:
    required = [
        stage / exe_name,
        stage / "README.txt",
        stage / "examples" / "hello" / "main.enk",
    ]
    for item in required:
        if not item.exists():
            raise RuntimeError(f"missing staged path: {item}")
    std_files = list((stage / "std").rglob("*.enk"))
    if not std_files:
        raise RuntimeError("staged std directory does not contain any .enk files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Enkai release archives.")
    parser.add_argument(
        "--version",
        help="Semantic version without leading v (default: parse from enkai/Cargo.toml)",
    )
    parser.add_argument(
        "--target-os",
        required=True,
        choices=["linux", "macos", "windows"],
        help="Asset OS label used in the artifact name",
    )
    parser.add_argument("--arch", required=True, help="Asset architecture label")
    parser.add_argument(
        "--archive-format",
        choices=["tar.gz", "zip"],
        help="Archive format (default: tar.gz for unix, zip for windows)",
    )
    parser.add_argument("--bin", required=True, help="Path to built enkai binary")
    parser.add_argument(
        "--native",
        action="append",
        default=[],
        help="Optional native library path(s) to include when present",
    )
    parser.add_argument("--out-dir", default="dist", help="Output directory for release assets")
    parser.add_argument(
        "--check-deterministic",
        action="store_true",
        help="Build archive bytes twice and fail if hashes differ",
    )
    return parser.parse_args()


def read_version_from_cargo(root: pathlib.Path) -> str:
    cargo = (root / "enkai" / "Cargo.toml").read_text(encoding="utf-8")
    for line in cargo.splitlines():
        line = line.strip()
        if line.startswith("version"):
            parts = line.split("=", 1)
            if len(parts) != 2:
                continue
            return parts[1].strip().strip('"')
    raise RuntimeError("failed to parse version from enkai/Cargo.toml")


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    version = args.version or read_version_from_cargo(root)

    archive_format = args.archive_format
    if archive_format is None:
        archive_format = "zip" if args.target_os == "windows" else "tar.gz"

    bin_path = pathlib.Path(args.bin)
    if not bin_path.is_absolute():
        bin_path = (root / bin_path).resolve()
    native_paths: list[pathlib.Path] = []
    for item in args.native:
        candidate = pathlib.Path(item)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        native_paths.append(candidate)

    exe_name = "enkai.exe" if args.target_os == "windows" else "enkai"

    with tempfile.TemporaryDirectory(prefix="enkai_release_stage_") as tmp_dir:
        stage = pathlib.Path(tmp_dir) / "stage"
        stage_release_tree(root, stage, exe_name, bin_path, native_paths)
        verify_required_layout(stage, exe_name)

        if archive_format == "tar.gz":
            first = build_tar_gz_bytes(stage, exe_name)
            second = build_tar_gz_bytes(stage, exe_name) if args.check_deterministic else first
        else:
            first = build_zip_bytes(stage, exe_name)
            second = build_zip_bytes(stage, exe_name) if args.check_deterministic else first

        first_hash = sha256_bytes(first)
        second_hash = sha256_bytes(second)
        if args.check_deterministic and first_hash != second_hash:
            raise RuntimeError(
                "determinism check failed: archive hashes differ between consecutive builds"
            )

        archive_name = expected_archive_name(version, args.target_os, args.arch, archive_format)
        out_path = out_dir / archive_name
        out_path.write_bytes(first)
        checksum_path = write_checksum(out_path)

        payload = {
            "status": "ok",
            "archive": str(out_path.relative_to(root)),
            "sha256": first_hash,
            "checksum_file": str(checksum_path.relative_to(root)),
            "format": archive_format,
            "deterministic": args.check_deterministic,
            "native_included": [path.name for path in native_paths if path.is_file()],
        }
        print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover - script entrypoint
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
