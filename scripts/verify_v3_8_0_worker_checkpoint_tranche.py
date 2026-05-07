#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify v3.8.0 worker/checkpoint parent tranche.")
    parser.add_argument("--contract", default="enkai/contracts/v3_8_0_worker_checkpoint_tranche.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_8_0_worker_checkpoint_tranche.json")
    return parser.parse_args()

def version_from(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not m: raise RuntimeError(f"no version in {path}")
    return m.group(1)

def main() -> int:
    args = parse_args(); root = Path(__file__).resolve().parents[1]
    contract = read_json(root / args.contract); failures = []
    required_version = contract["required_version"]
    for rel in ["enkai/Cargo.toml", "enkaic/Cargo.toml", "enkairt/Cargo.toml", "enkai_native/Cargo.toml", "enkai_tensor/Cargo.toml"]:
        if version_from(root / rel) != required_version: failures.append(f"{rel} version mismatch")
    if f'version = "{required_version}"' not in (root / "enkai.toml").read_text(encoding="utf-8-sig"):
        failures.append("enkai.toml version mismatch")
    reports = []
    for rel in contract["required_reports"]:
        payload = read_json(root / rel); reports.append(rel)
        if payload.get("all_passed") is not True: failures.append(f"{rel} all_passed expected True")
    for rel, snippets in contract["required_text_snippets"].items():
        text = (root / rel).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text: failures.append(f"missing snippet in {rel}: {snippet}")
    output = {"schema_version": 1, "contract": str((root / args.contract).resolve()), "generated_at_utc": datetime.now(timezone.utc).isoformat(), "all_passed": not failures, "required_reports": reports, "failures": failures}
    out = root / args.output; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1
if __name__ == "__main__": raise SystemExit(main())
