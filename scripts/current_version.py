#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    cargo = (root / "enkai" / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', cargo, flags=re.MULTILINE)
    if not match:
        print("failed to parse version from enkai/Cargo.toml", file=sys.stderr)
        return 1
    print(match.group(1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
