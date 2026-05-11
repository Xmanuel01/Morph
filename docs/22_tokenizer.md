# Tokenizer

The `tokenizer` module provides a deterministic whitespace tokenizer with a
stable, fingerprinted file format. Saved tokenizers include format version,
vocabulary SHA-1, special token IDs, and provenance metadata so training
pipelines can reject tokenizer drift before replay or checkpoint resume.

The current runtime exposes `tokenizer.*` as a training runtime namespace.
Package-stable data-pipeline wrappers are also available through `std::data`
where supported.

## API

```enkai
import std::json

let cfg := json.parse("{\"path\":\"data.txt\",\"vocab_size\":32000}")
let tok := tokenizer.train(cfg)
let ids := tok.encode("hello world")
let text := tok.decode(ids)
```

Load a saved tokenizer:

```enkai
let tok := tokenizer.load("tokenizer.json")
```

## Notes

- `tok.encode` returns a Buffer of token IDs (little-endian u32).
- `tok.decode` accepts Buffer or List of Int.
- `tokenizer.train` supports optional fields: `lowercase`, `min_freq`, `save_path`.
- Saved tokenizer JSON includes `format_version`, `vocab_sha1`, and
  deterministic provenance. Loading a tampered tokenizer fails with a
  fingerprint mismatch.

## Common errors

- Missing `path` in the training config.
- Non-string path values.
- Tokenizer fingerprint mismatch when the saved vocabulary or special IDs have
  been modified after serialization.
- Using `json.parse` without `import std::json`.

