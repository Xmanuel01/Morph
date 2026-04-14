#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a strict self-host dependency inventory from the frozen board."
    )
    parser.add_argument(
        "--board",
        required=True,
        help="Path to the strict self-host dependency board JSON",
    )
    parser.add_argument(
        "--workspace-cargo",
        default="Cargo.toml",
        help="Path to the root workspace Cargo.toml",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the generated inventory",
    )
    return parser.parse_args()


def read_json(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def parse_workspace_members(path: pathlib.Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    members_match = re.search(r"members\s*=\s*\[(.*?)\]", content, flags=re.DOTALL)
    if not members_match:
        raise RuntimeError(f"failed to parse workspace members from {path}")
    raw = members_match.group(1)
    return sorted(
        {
            match.group(1).strip()
            for match in re.finditer(r'"([^"]+)"', raw)
            if match.group(1).strip()
        }
    )


def ensure_component_schema(component: dict, workspace_members: set[str]) -> None:
    required_fields = {
        "id",
        "subsystem",
        "shipped_blocker",
        "status",
        "rust_workspace_members",
        "native_components",
        "enkai_replacements",
        "notes",
    }
    missing = sorted(required_fields.difference(component.keys()))
    if missing:
        raise RuntimeError(
            f"component {component.get('id', '<unknown>')} missing fields: {', '.join(missing)}"
        )
    status = component["status"]
    if status not in {"blocked", "partial", "done"}:
        raise RuntimeError(
            f"component {component['id']} has invalid status {status!r}; expected blocked|partial|done"
        )
    for member in component["rust_workspace_members"]:
        if member not in workspace_members:
            raise RuntimeError(
                f"component {component['id']} references unknown workspace member {member!r}"
            )


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    board_path = (root / args.board).resolve() if not pathlib.Path(args.board).is_absolute() else pathlib.Path(args.board)
    cargo_path = (
        (root / args.workspace_cargo).resolve()
        if not pathlib.Path(args.workspace_cargo).is_absolute()
        else pathlib.Path(args.workspace_cargo)
    )
    output_path = (
        (root / args.output).resolve()
        if not pathlib.Path(args.output).is_absolute()
        else pathlib.Path(args.output)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    board = read_json(board_path)
    workspace_members = parse_workspace_members(cargo_path)
    workspace_member_set = set(workspace_members)
    components = board.get("components")
    if not isinstance(components, list) or not components:
        raise RuntimeError("dependency board must define a non-empty components list")

    normalized_components = []
    remaining_rust_dependencies: set[str] = set()
    remaining_native_dependencies: set[str] = set()
    blocking_subsystems: list[str] = []

    for item in components:
        if not isinstance(item, dict):
            raise RuntimeError("dependency board components must be objects")
        ensure_component_schema(item, workspace_member_set)
        rust_members = sorted({str(value) for value in item["rust_workspace_members"]})
        native_components = sorted({str(value) for value in item["native_components"]})
        enkai_replacements = sorted({str(value) for value in item["enkai_replacements"]})
        component = {
            "id": str(item["id"]),
            "subsystem": str(item["subsystem"]),
            "shipped_blocker": bool(item["shipped_blocker"]),
            "status": str(item["status"]),
            "rust_workspace_members": rust_members,
            "native_components": native_components,
            "enkai_replacements": enkai_replacements,
            "notes": str(item["notes"]),
        }
        normalized_components.append(component)
        if component["shipped_blocker"] and component["status"] != "done":
            blocking_subsystems.append(component["id"])
            remaining_rust_dependencies.update(rust_members)
            remaining_native_dependencies.update(native_components)

    summary = {
        "total_components": len(normalized_components),
        "done_components": sum(1 for item in normalized_components if item["status"] == "done"),
        "partial_components": sum(1 for item in normalized_components if item["status"] == "partial"),
        "blocked_components": sum(1 for item in normalized_components if item["status"] == "blocked"),
        "strict_selfhost_cpu_complete": len(blocking_subsystems) == 0,
        "remaining_rust_dependencies": sorted(remaining_rust_dependencies),
        "remaining_native_dependencies": sorted(remaining_native_dependencies),
        "blocking_subsystems": sorted(blocking_subsystems),
    }

    report = {
        "schema_version": 1,
        "profile": "strict_selfhost",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "board": str(board_path.relative_to(root)),
        "workspace_cargo": str(cargo_path.relative_to(root)),
        "workspace_members": workspace_members,
        "target_state": board.get("target_state"),
        "policy": board.get("policy", {}),
        "summary": summary,
        "components": normalized_components,
    }

    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "profile": "strict_selfhost",
                "output": str(output_path.relative_to(root)),
                "strict_selfhost_cpu_complete": summary["strict_selfhost_cpu_complete"],
                "remaining_rust_dependencies": summary["remaining_rust_dependencies"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:  # pragma: no cover - script entrypoint
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
