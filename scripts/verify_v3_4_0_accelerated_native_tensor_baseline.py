#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the v3.4.0 accelerated native/tensor baseline scope.")
    parser.add_argument('--contract', default='enkai/contracts/v3_4_0_accelerated_native_tensor_baseline.json')
    parser.add_argument('--output', default='artifacts/readiness/v3_4_0_accelerated_native_tensor_baseline.json')
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    output_path = (repo_root / args.output).resolve()
    contract = load_json(contract_path)
    failures: list[str] = []

    for rel_path in contract.get('required_files', []):
        if not (repo_root / rel_path).is_file():
            failures.append(f'missing required file {rel_path}')

    inv_path = repo_root / 'artifacts/readiness/strict_selfhost_dependency_inventory.json'
    if inv_path.is_file():
        inv = load_json(inv_path)
        summary = inv.get('summary', {})
        if summary.get('strict_selfhost_cpu_complete') is not True:
            failures.append('strict_selfhost dependency inventory no longer reports cpu complete')
        comps = {c.get('id'): c for c in inv.get('components', []) if isinstance(c, dict)}
        for comp_id in ('tensor_backend', 'native_std_and_accel'):
            comp = comps.get(comp_id)
            if comp is None:
                failures.append(f'missing component {comp_id} in dependency inventory')
                continue
            if comp.get('status') != 'done':
                failures.append(f'{comp_id} is not done in dependency inventory')
        tensor_notes = str(comps.get('tensor_backend', {}).get('notes', ''))
        if 'Broader future work to replace the underlying accelerated Rust tensor backend globally remains roadmap work' not in tensor_notes:
            failures.append('tensor_backend notes no longer preserve the broader global replacement boundary')
        native_notes = str(comps.get('native_std_and_accel', {}).get('notes', ''))
        if 'Broader future native acceleration work outside the shipped strict-selfhost surface remains roadmap work' not in native_notes:
            failures.append('native_std_and_accel notes no longer preserve the broader future native-acceleration boundary')

    for rel_path in [
        'artifacts/readiness/strict_selfhost_tensor_backend_surface.json',
        'artifacts/readiness/strict_selfhost_native_std_and_accel_surface.json',
        'artifacts/readiness/strict_selfhost.json',
        'artifacts/readiness/strict_selfhost_blockers.json',
        'artifacts/readiness/v3_4_0_zero_rust_next_step_baseline.json',
    ]:
        path = repo_root / rel_path
        if path.is_file() and load_json(path).get('all_passed') is not True:
            failures.append(f'{rel_path} is not green')

    for rel_path, snippets in contract.get('required_text_snippets', {}).items():
        path = repo_root / rel_path
        if not path.is_file():
            failures.append(f'missing required text file {rel_path}')
            continue
        text = path.read_text(encoding='utf-8')
        for snippet in snippets:
            if snippet not in text:
                failures.append(f'{rel_path} missing required snippet: {snippet}')

    report = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'contract': str(contract_path),
        'all_passed': not failures,
        'failures': failures,
        'verified_scope': contract.get('scope'),
        'verified_contract_version': contract.get('contract_version'),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'ok' if report['all_passed'] else 'failed', 'output': str(output_path), 'all_passed': report['all_passed']}, separators=(',', ':')))
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
