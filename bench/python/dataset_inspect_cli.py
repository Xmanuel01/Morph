#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset inspect baseline fixture")
    parser.add_argument("dataset_path")
    parser.add_argument("tokenizer_path")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_text = Path(args.dataset_path).read_text(encoding="utf-8", errors="ignore")
    tokenizer = json.loads(Path(args.tokenizer_path).read_text(encoding="utf-8"))

    vocab = tokenizer.get("vocab", [])
    vocab_index = {token: idx + 1 for idx, token in enumerate(vocab)}

    tokens = [tok for tok in dataset_text.split() if tok]
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(tokens)

    encoded = [vocab_index.get(tok, 0) for tok in tokens]
    seq_len = max(1, args.seq_len)
    batch_size = max(1, args.batch_size)
    max_tokens = seq_len * batch_size
    token_count = min(len(encoded), max_tokens)
    packing_efficiency = (float(token_count) / float(max_tokens)) if max_tokens > 0 else 0.0

    out = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "token_count": token_count,
        "packing_efficiency": packing_efficiency,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
