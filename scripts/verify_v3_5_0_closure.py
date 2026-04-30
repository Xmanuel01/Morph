#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--contract', default='enkai/contracts/v3_5_0_closure.json')
    p.add_argument('--output', default='artifacts/readiness/v3_5_0_closure.json')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    contract=load_json((root/args.contract).resolve())
    out=(root/args.output).resolve()
    failures=[]
    for rel in contract.get('required_files', []):
        path=root/rel
        if not path.is_file():
            failures.append(f'missing required file {rel}')
            continue
        if load_json(path).get('all_passed') is not True:
            failures.append(f'{rel} is not green')
    for rel, snippets in contract.get('required_text_snippets', {}).items():
        path=root/rel
        if not path.is_file():
            failures.append(f'missing required text file {rel}')
            continue
        text=path.read_text(encoding='utf-8')
        for s in snippets:
            if s not in text:
                failures.append(f'{rel} missing required snippet: {s}')
    report={'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'contract':str((root/args.contract).resolve()),'all_passed':not failures,'failures':failures,'verified_scope':contract.get('scope'),'verified_contract_version':contract.get('contract_version')}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'status':'ok' if report['all_passed'] else 'failed','output':str(out),'all_passed':report['all_passed']}, separators=(',',':')))
    return 0 if report['all_passed'] else 1
if __name__ == '__main__':
    raise SystemExit(main())
