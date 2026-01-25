# Train/Eval CLI

Morph v0.8 adds:

```
morph train config.morph
morph eval config.morph
```

## Config format

`config.morph` must return a record. The simplest approach:

```
fn main() ::
    return json.parse("{\"vocab_size\":8,\"hidden_size\":4,\"seq_len\":4,\"batch_size\":2,\"lr\":0.1,\"dataset_path\":\"data.txt\",\"checkpoint_dir\":\"ckpt\",\"max_steps\":2,\"save_every\":1,\"log_every\":1,\"tokenizer_train\":{\"path\":\"data.txt\",\"vocab_size\":8}}")
::
```

## Evaluation

```
morph eval config.morph
```

Uses the latest checkpoint and computes average loss/perplexity.

## Common errors

- `tokenizer_path` or `tokenizer_train` missing.
- `checkpoint_dir` not writable.
