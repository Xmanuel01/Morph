#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8-sig'))

def run(contract_rel: str, output_rel: str, require_text: bool = True, extra=None):
    root = Path(__file__).resolve().parents[1]
    contract = load_json((root / contract_rel).resolve())
    out = (root / output_rel).resolve()
    failures=[]
    for rel in contract.get('required_files', []):
        path = root / rel
        if not path.is_file():
            failures.append(f'missing required file {rel}')
            continue
        if rel.endswith('.json') and rel.startswith('artifacts/readiness/') and rel not in ('artifacts/readiness/strict_selfhost_dependency_inventory.json',):
            payload = load_json(path)
            if payload.get('all_passed') is not True:
                failures.append(f'{rel} is not green')
    if require_text:
        for rel, snippets in contract.get('required_text_snippets', {}).items():
            path = root / rel
            if not path.is_file():
                failures.append(f'missing required text file {rel}')
                continue
            text = path.read_text(encoding='utf-8')
            for s in snippets:
                if s not in text:
                    failures.append(f'{rel} missing required snippet: {s}')
    if extra is not None:
        extra(root, failures)
    report={'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'contract':str((root/contract_rel).resolve()),'all_passed':not failures,'failures':failures,'verified_scope':contract.get('scope'),'verified_contract_version':contract.get('contract_version')}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'status':'ok' if report['all_passed'] else 'failed','output':str(out),'all_passed':report['all_passed']}, separators=(',',':')))
    return 0 if report['all_passed'] else 1

def extra_start(root: Path, failures: list[str]):
    for rel in ['enkai/Cargo.toml','enkaic/Cargo.toml','enkairt/Cargo.toml','enkai_native/Cargo.toml','enkai_tensor/Cargo.toml','enkai.toml']:
        if '3.6.0' not in (root/rel).read_text(encoding='utf-8'):
            failures.append(f'{rel} does not contain version 3.6.0')

def extra_evidence(root: Path, failures: list[str]):
    inv = load_json(root / 'artifacts/readiness/strict_selfhost_dependency_inventory.json')
    summary = inv.get('summary', {})
    if summary.get('done_components') != 6 or summary.get('partial_components') != 0 or summary.get('blocked_components') != 0:
        failures.append('strict selfhost dependency inventory is no longer fully green for shipped surface')

if __name__ == '__main__':
    import sys
    name = Path(__file__).name
    if name == 'verify_v3_6_0_release_line_start_baseline.py':
        raise SystemExit(run('enkai/contracts/v3_6_0_release_line_start_baseline.json','artifacts/readiness/v3_6_0_release_line_start_baseline.json', True, extra_start))
    if name == 'verify_v3_6_0_objective_set_freeze.py':
        raise SystemExit(run('enkai/contracts/v3_6_0_objective_set_freeze.json','artifacts/readiness/v3_6_0_objective_set_freeze.json'))
    if name == 'verify_v3_6_0_evidence_continuity_baseline.py':
        raise SystemExit(run('enkai/contracts/v3_6_0_evidence_continuity_baseline.json','artifacts/readiness/v3_6_0_evidence_continuity_baseline.json', False, extra_evidence))
    if name == 'verify_v3_6_0_closure.py':
        raise SystemExit(run('enkai/contracts/v3_6_0_closure.json','artifacts/readiness/v3_6_0_closure.json'))
    raise SystemExit('unknown script name')
