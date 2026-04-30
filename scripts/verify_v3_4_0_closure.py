#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Verify the v3.4.0 closure surface.')
    parser.add_argument('--contract', default='enkai/contracts/v3_4_0_closure.json')
    parser.add_argument('--output', default='artifacts/readiness/v3_4_0_closure.json')
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
        path = repo_root / rel_path
        if not path.is_file():
            failures.append(f'missing required file {rel_path}')
            continue
        payload = load_json(path)
        if payload.get('all_passed') is not True:
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
