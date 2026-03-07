#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import sys


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_version(root: pathlib.Path) -> str:
    cargo = (root / "enkai" / "Cargo.toml").read_text(encoding="utf-8")
    for line in cargo.splitlines():
        line = line.strip()
        if line.startswith("version"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"')
    raise RuntimeError("failed to parse version from enkai/Cargo.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and archive release evidence bundle.")
    parser.add_argument("--version", help="Release version (defaults to enkai/Cargo.toml)")
    parser.add_argument("--gpu-log-dir", default="artifacts/gpu", help="GPU evidence source directory")
    parser.add_argument("--dist-dir", default="dist", help="Built artifacts source directory")
    parser.add_argument(
        "--out-dir",
        default="artifacts/release",
        help="Output directory for evidence bundle",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if required GPU evidence files are missing",
    )
    return parser.parse_args()


def ensure_required_gpu_files(gpu_dir: pathlib.Path) -> list[pathlib.Path]:
    required = [
        "single_gpu.log",
        "single_gpu_evidence.json",
        "multi_gpu.log",
        "multi_gpu_evidence.json",
        "soak_4gpu.log",
        "soak_4gpu_evidence.json",
    ]
    missing = [name for name in required if not (gpu_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "missing required GPU evidence files: " + ", ".join(sorted(missing))
        )
    return [gpu_dir / name for name in required]


def copy_files(
    root: pathlib.Path,
    source_base: pathlib.Path,
    destination_base: pathlib.Path,
    files: list[pathlib.Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sorted(files):
        if not source.is_file():
            continue
        rel = source.relative_to(source_base)
        target = destination_base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append(
            {
                "source": str(source.relative_to(root)),
                "copied_to": str(target.relative_to(root)),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    version = args.version or parse_version(root)

    gpu_dir = (root / args.gpu_log_dir).resolve()
    dist_dir = (root / args.dist_dir).resolve()
    out_root = (root / args.out_dir / f"v{version}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "schema": "enkai-release-evidence-v1",
        "version": f"v{version}",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_required": args.require_gpu,
        "files": [],
    }

    file_rows: list[dict[str, object]] = []
    if gpu_dir.is_dir():
        if args.require_gpu:
            gpu_files = ensure_required_gpu_files(gpu_dir)
        else:
            gpu_files = sorted(path for path in gpu_dir.glob("*") if path.is_file())
        file_rows.extend(copy_files(root, gpu_dir, out_root / "gpu", gpu_files))
    elif args.require_gpu:
        raise RuntimeError(f"GPU evidence directory not found: {gpu_dir}")

    if dist_dir.is_dir():
        dist_files = sorted(path for path in dist_dir.glob("*") if path.is_file())
        file_rows.extend(copy_files(root, dist_dir, out_root / "dist", dist_files))

    manifest["files"] = file_rows
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "status": "ok",
        "version": f"v{version}",
        "out_dir": str(out_root.relative_to(root)),
        "files": len(file_rows),
        "manifest": str(manifest_path.relative_to(root)),
        "gpu_required": args.require_gpu,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover - script entrypoint
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
