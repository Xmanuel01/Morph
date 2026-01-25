# Sharded Checkpoints (v0.9)

The GPU backend supports a future-proof sharded checkpoint format.

## Format

```
checkpoint/step_00001/
  model_rank0.bin
  optim_rank0.bin
  optim_meta.json
  meta.json
```

`meta.json` includes:

- `step`
- `tokens`
- `loss`
- `config_hash`
- `world_size` (future)

`optim_meta.json` includes:

- `step`
- `slots`: list of `{ param, m, v }` entries pointing to tensors stored in `optim_rank0.bin`

## API

```
tensor.save_sharded(dir, param, opt_state, meta)
let bundle := tensor.load_sharded(dir)
```

Multi-parameter:

```
tensor.save_sharded_multi(dir, params, opt_state, meta)
let bundle := tensor.load_sharded_multi(dir)
```

`bundle` is a record:
- `param`
- `opt_state`
- `meta`

`bundle` for multi-parameter is a record:
- `params`
- `opt_state`
- `meta`

## Notes

- v0.9 uses `world_size=1`.
- v0.10+ will add multi-GPU sharding and tensor parallel layouts.
