#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def cargo_version() -> str:
    cargo = read("enkai/Cargo.toml")
    m = re.search(r'^version\s*=\s*"([^"]+)"', cargo, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("failed to parse version from enkai/Cargo.toml")
    return m.group(1)


def main() -> int:
    version = cargo_version()
    expected_tag = f"v{version}"
    failures: list[str] = []

    main_rs = read("enkai/src/main.rs")
    if 'const LANG_VERSION: &' in main_rs and 'env!("ENKAI_LANG_VERSION")' not in main_rs:
        failures.append("enkai/src/main.rs still hardcodes LANG_VERSION")

    readme = read("README.md")
    if f"Status ({expected_tag})" not in readme:
        failures.append(f"README.md missing Status ({expected_tag})")
    if "Distributed stubs:" in readme:
        failures.append("README.md contains outdated distributed stubs claim")

    docs_readme = read("docs/README.md")
    if expected_tag not in docs_readme:
        failures.append(f"docs/README.md missing release tag {expected_tag}")

    spec = read("docs/Enkai.spec")
    if f"v0.1 -> {expected_tag}" not in spec:
        failures.append("docs/Enkai.spec title is out of sync with crate version")
    if f"Known Limits in {expected_tag}" not in spec:
        failures.append("docs/Enkai.spec known limits header is out of sync")
    if "compile to stub functions" in spec:
        failures.append("docs/Enkai.spec still claims tool declarations compile to stubs")

    validation = read("VALIDATION.md")
    if "Validation Matrix" not in validation:
        failures.append("VALIDATION.md title should use release-line validation matrix wording")

    frontend_docs = read("docs/27_frontend_stack.md")
    if "backend_api.snapshot.json" not in frontend_docs:
        failures.append("docs/27_frontend_stack.md missing backend snapshot reference")
    if "sdk_api.snapshot.json" not in frontend_docs:
        failures.append("docs/27_frontend_stack.md missing SDK snapshot reference")

    required_snapshots = [
        ROOT / "enkai/contracts/backend_api_v1.snapshot.json",
        ROOT / "enkai/contracts/sdk_api_v1.snapshot.json",
        ROOT / "enkai/contracts/conversation_state_v1.schema.json",
    ]
    for snapshot in required_snapshots:
        if not snapshot.is_file():
            failures.append(f"missing contract snapshot file: {snapshot.relative_to(ROOT)}")

    if failures:
        print("docs consistency check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"docs consistency check passed for {expected_tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
