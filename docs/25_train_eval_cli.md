# Train/Eval CLI

Enkai v0.8 adds:

```
Enkai train config.enk
Enkai eval config.enk
```

## Config format

`config.enk` must return a record. v1 configs must include `config_version: 1`
and an explicit `backend` ("cpu" or "native"). The simplest approach:

```
fn main() ::
    return json.parse("{\"config_version\":1,\"backend\":\"cpu\",\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8}}")
::
```

## Evaluation

```
Enkai eval config.enk
```

Uses the latest checkpoint and computes average loss/perplexity.

## Common errors

- `tokenizer_path` or `tokenizer_train` missing.
- `checkpoint_dir` not writable.
- `config_version` missing or not `1` for v1 configs.
- `backend`, `dtype`, or `device` invalid for the selected backend.


