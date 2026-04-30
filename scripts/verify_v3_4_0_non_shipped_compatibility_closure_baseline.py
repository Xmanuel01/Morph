#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the v3.4.0 non-shipped compatibility-closure baseline scope.")
    parser.add_argument('--contract', default='enkai/contracts/v3_4_0_non_shipped_compatibility_closure_baseline.json')
    parser.add_argument('--output', default='artifacts/readiness/v3_4_0_non_shipped_compatibility_closure_baseline.json')
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

    readiness_files = [
        'artifacts/readiness/v3_4_0_zero_rust_next_step_baseline.json',
        'artifacts/readiness/v3_4_0_install_host_matrix_baseline.json',
        'artifacts/readiness/v3_4_0_compatibility_storage_data_baseline.json',
        'artifacts/readiness/v3_4_0_accelerated_native_tensor_baseline.json',
        'artifacts/readiness/strict_selfhost.json',
        'artifacts/readiness/strict_selfhost_blockers.json',
    ]
    for rel_path in readiness_files:
        path = repo_root / rel_path
        if path.is_file() and load_json(path).get('all_passed') is not True:
            failures.append(f'{rel_path} is not green')

    inv_path = repo_root / 'artifacts/readiness/strict_selfhost_dependency_inventory.json'
    if inv_path.is_file():
        inv = load_json(inv_path)
        summary = inv.get('summary', {})
        if summary.get('done_components') != 6:
            failures.append('strict_selfhost dependency inventory no longer reports 6 done components')
        if summary.get('partial_components') != 0 or summary.get('blocked_components') != 0:
            failures.append('strict_selfhost dependency inventory is no longer fully closed for the shipped surface')

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
