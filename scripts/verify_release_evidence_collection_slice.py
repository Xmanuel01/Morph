#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify contract-backed release evidence collection.")
    parser.add_argument("--contract", default="enkai/contracts/selfhost_release_evidence_collection_v3_3_0.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    contract_path = (root / args.contract).resolve() if not Path(args.contract).is_absolute() else Path(args.contract).resolve()
    output_path = Path(args.output).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    proc = subprocess.run(
        ["py", "-3", "scripts/collect_release_evidence.py", "--contract", str(contract_path)],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    collector_payload = None
    bundle_root = None
    manifest_ok = False
    required_archived = []
    if proc.returncode == 0:
        collector_payload = json.loads(proc.stdout.strip())
        bundle_root = root / collector_payload["out_dir"]
        manifest_ok = (bundle_root / "manifest.json").is_file()
        for rel in contract.get("required_archived_files", []):
            path = bundle_root / rel
            required_archived.append({"path": str(path), "present": path.is_file()})
    all_required_present = all(item["present"] for item in required_archived)
    all_passed = proc.returncode == 0 and manifest_ok and all_required_present

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(contract_path),
        "collector_returncode": proc.returncode,
        "collector_stdout": proc.stdout,
        "collector_stderr": proc.stderr,
        "bundle_root": str(bundle_root) if bundle_root else None,
        "manifest_ok": manifest_ok,
        "required_archived_files": required_archived,
        "all_passed": all_passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok" if all_passed else "failed", "output": str(output_path), "all_passed": all_passed}, separators=(",", ":")))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
