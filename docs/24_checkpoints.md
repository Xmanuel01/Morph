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
- `step` (Int)
- `tokens` (Int, optional)
- `loss` (Float, optional)
- `config_hash` (String, optional)

## Notes

- Checkpoints are stored in `step_XXXXXXXX` subdirectories.
- Save is atomic: write to temp, then rename.
- `checkpoint.latest` returns `null` if none exist.

## Common errors

- Missing `weights` or `step`.
- Invalid buffer sizes (must be multiple of 4 bytes for f32).

