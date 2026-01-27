# Dataset Streaming

The `dataset` module streams text shards, packs tokens into fixed sequences, and returns batches.

## API

```
let tok := tokenizer.load("tokenizer.json")
let cfg := json.parse("{\"seq_len\":128,\"batch_size\":8,\"add_eos\":true,\"drop_remainder\":true}")
let stream := dataset.open("data/*.txt", tok, cfg)
let batch := stream.next_batch()
```

`batch` is a record with:

- `input_ids` (Buffer)
- `target_ids` (Buffer)
- `attention_mask` (Buffer)
- `batch_size` (Int)
- `seq_len` (Int)

## Notes

- `dataset.open` supports file paths, directories, and glob patterns.
- `drop_remainder=false` pads the last batch and emits `attention_mask=0` for padded tokens.

## Common errors

- Missing `seq_len` or `batch_size` in config.
- Dataset path not found / empty.

