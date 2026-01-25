# Tokenizer

The `tokenizer` module provides a simple whitespace tokenizer with a stable file format.

## API

```
let cfg := json.parse("{\"path\":\"data.txt\",\"vocab_size\":32000}")
let tok := tokenizer.train(cfg)
let ids := tok.encode("hello world")
let text := tok.decode(ids)
```

Load a saved tokenizer:

```
let tok := tokenizer.load("tokenizer.json")
```

## Notes

- `tok.encode` returns a Buffer of token IDs (little-endian u32).
- `tok.decode` accepts Buffer or List of Int.
- `tokenizer.train` supports optional fields: `lowercase`, `min_freq`, `save_path`.

## Common errors

- Missing `path` in the training config.
- Non-string path values.
