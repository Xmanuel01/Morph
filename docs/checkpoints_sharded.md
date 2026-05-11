# Sharded Checkpoints

The tensor/checkpoint runtime supports sharded checkpoint helpers for bounded
training and compatibility with future multi-rank layouts.

## Format

```text
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

```enkai
import std::tensor

tensor.save_sharded(dir, param, opt_state, meta)
let bundle := tensor.load_sharded(dir)
```

Multi-parameter:

```enkai
import std::tensor

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

- Single-rank sharded helpers are available for bounded proof workloads.
- Broader multi-rank checkpoint merge/replay requires green distributed
  verifier evidence before production claims.

