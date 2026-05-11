# Dataset Streaming

The `dataset` module streams text shards, packs tokens into fixed sequences, and
returns batches. The runtime now exposes deterministic cursor snapshots and a
pipeline manifest for production training replay evidence.

The current runtime exposes `dataset.*` as a training runtime namespace.
Use `std::json` for config records.

## API

```enkai
import std::json

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
- `DatasetStream::cursor` snapshots the current file index, line offset,
  buffered token IDs, exhaustion state, and emitted batch count.
- `DatasetStream::restore_cursor` restores a snapshot and replays the same next
  batch when `prefetch_batches = 0`.
- `dataset_pipeline_manifest` archives file SHA-1 values, dataset SHA-1,
  tokenizer SHA-1, batching config, shuffle seed, and replay capability.

## Common errors

- Missing `seq_len` or `batch_size` in config.
- Dataset path not found / empty.
- Cursor snapshot/restore is rejected when prefetching is enabled, because the
  background reader makes exact replay ownership ambiguous.
- Using `json.parse` requires `import std::json`.

