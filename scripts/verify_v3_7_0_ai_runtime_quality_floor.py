#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the bounded AI runtime QA floor score.")
    parser.add_argument("--contract", default="enkai/contracts/v3_7_0_ai_runtime_quality_floor.json")
    parser.add_argument("--output", default="artifacts/readiness/v3_7_0_ai_runtime_quality_floor.json")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = (repo_root / args.contract).resolve()
    contract = read_json(contract_path)
    weights = contract["weights"]

    component_scores: dict[str, dict[str, Any]] = {}
    total_score = 0.0
    failures: list[str] = []

    for name, rel_path in contract["required_artifacts"].items():
        artifact = read_json((repo_root / rel_path).resolve())
        passed = bool(artifact.get("all_passed"))
        score = float(weights[name]) if passed else 0.0
        component_scores[name] = {"passed": passed, "score": score, "artifact": rel_path}
        total_score += score
        if not passed:
            failures.append(f"artifact {name} did not pass")

    doc_passed = True
    for rel_path, snippets in contract["required_text_snippets"].items():
        text = (repo_root / rel_path).read_text(encoding="utf-8-sig")
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"missing snippet in {rel_path}: {snippet}")
                doc_passed = False
    doc_score = float(weights["doc_evidence"]) if doc_passed else 0.0
    total_score += doc_score
    component_scores["doc_evidence"] = {"passed": doc_passed, "score": doc_score}

    output = {
        "schema_version": 1,
        "contract": str(contract_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": total_score >= float(contract["score_threshold"]) and not failures,
        "score": round(total_score, 3),
        "score_threshold": float(contract["score_threshold"]),
        "component_scores": component_scores,
        "failures": failures,
    }
    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
