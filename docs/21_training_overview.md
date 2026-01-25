# Training Overview (v0.8)

Morph v0.8 provides an end-to-end training pipeline:

1) Tokenizer (train/load, encode/decode)
2) Streaming dataset loader (shards + packing + batching)
3) Checkpoint manager (save/load/rotate/resume)
4) `morph train` / `morph eval`

The training CLI expects a `config.morph` file that returns a config record.
Because Morph doesn’t yet have record literals, use `json.parse` to build the config.

## Minimal example

```
fn main() ::
    return json.parse("{\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8}}")
::
```

Run:

```
morph train config.morph
morph eval config.morph
```

## Common errors

- Missing required config fields (e.g., `seq_len`, `batch_size`, `dataset_path`).
- Tokenizer config missing (`tokenizer_path` or `tokenizer_train`).
