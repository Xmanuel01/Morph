#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenizer baseline fixture")
    parser.add_argument("dataset_path")
    parser.add_argument("tokenizer_path")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = Path(args.dataset_path).read_text(encoding="utf-8", errors="ignore")
    tokens = [tok for tok in text.split() if tok]
    counts = Counter(tokens)

    vocab = [
        token
        for token, freq in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if freq >= args.min_freq
    ][: args.vocab_size]

    out = {
        "format": "py-tokenizer-v1",
        "seed": args.seed,
        "vocab_size": len(vocab),
        "vocab": vocab,
        "unk": "<unk>",
    }
    path = Path(args.tokenizer_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
