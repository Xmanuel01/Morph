# Checkpoints

The `checkpoint` module persists training state safely and atomically.

## API

```
checkpoint.save("ckpt", state)
let path := checkpoint.latest("ckpt")
let loaded := checkpoint.load(path)
checkpoint.rotate("ckpt", 3)
```

`state` is a record with:

- `weights` (Buffer, f32 bytes)
- `optimizer` (Buffer, optional)
- `format_version` (Int, optional, defaults to 1)
- `step` (Int)
- `tokens` (Int, optional)
- `loss` (Float, optional)
- `config_hash` (String, optional)
- `model_sig` (String, optional)
- `dtype` (String, optional)
- `device` (String, optional)

## Notes

- Checkpoints are stored in `step_XXXXXXXX` subdirectories.
- Save is atomic: write to temp, then rename.
- `checkpoint.latest` returns `null` if none exist.
- v1 checkpoint metadata includes `format_version: 1` and is backward compatible
  with legacy checkpoints that omit it.

## Common errors

- Missing `weights` or `step`.
- Invalid buffer sizes (must be multiple of 4 bytes for f32).

