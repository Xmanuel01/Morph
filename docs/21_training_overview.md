# Training Overview (v0.8)

Enkai v0.8 provides an end-to-end training pipeline:

1) Tokenizer (train/load, encode/decode)
2) Streaming dataset loader (shards + packing + batching)
3) Checkpoint manager (save/load/rotate/resume)
4) `Enkai train` / `Enkai eval`

The training CLI expects a `config.enk` file that returns a config record.
v1 configs must include `config_version: 1` and an explicit `backend`.
Because Enkai doesn't yet have record literals, use `json.parse` to build the config.

## Minimal example

```
fn main() ::
    return json.parse("{\"config_version\":1,\"backend\":\"cpu\",\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8}}")
::
```

Determinism:
- `seed` (Int >= 0) controls model initialization and is also used to seed tokenizer
  training tie-breaks and dataset file shuffling for repeatable runs.

Run:

```
Enkai train config.enk
Enkai eval config.enk
```

## Common errors

- Missing required config fields (e.g., `seq_len`, `batch_size`, `dataset_path`).
- Tokenizer config missing (`tokenizer_path` or `tokenizer_train`).
- Invalid `backend`, `dtype`, or `device` settings.
