# Train/Eval CLI

Enkai v0.8 adds:

```
Enkai train config.enk
Enkai pretrain config.enk
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

Optional determinism:
- `seed` (Int >= 0) controls model initialization and is also used to seed tokenizer
  training tie-breaks and dataset file shuffling for repeatable runs.

Distributed orchestration (additive, backward compatible):
- `world_size` (Int >= 1, default `1`)
- `rank` (Int >= 0, default `0`)
- `dist.topology` (`standalone|single-node|multi-node`)
- `dist.rendezvous` (String; use `tcp://host:port` for `multi-node`)
- `dist.retry_budget` (Int >= 0)
- `dist.device_map` (String CSV like `"0,1"` or list of Ints)
- For `world_size > 1`, set `ENKAI_ENABLE_DIST=1`.

Optional model architecture fields (additive, backward compatible):
- `model.preset`: `tinylm`, `gpt2-small`, `gpt2-medium`, `llama-7b`
- `model.d_model`, `model.n_layers`, `model.n_heads`
- `model.ff_mult`, `model.activation` (`gelu|relu|silu`), `model.norm` (`layernorm|rmsnorm`)
- `model.tie_embeddings`, `model.dropout`

Runtime safety controls (additive):
- `ema_decay` (default `0.95`)
- `divergence_factor` (default `4.0`)
- `divergence_patience` (default `3`)
- `divergence_warmup_steps` (default `25`)

Pretraining run metadata (additive):
- `run_id` (String, optional)
- `parent_run_id` (String, optional)
- `run_name` (String, optional)
- `checkpoint_policy` (record, optional):
  - `validate_on_save` (Bool, default `true`)
  - `validate_on_resume` (Bool, default `true`)
  - `retention_recent` (Int > 0, default `keep_last`)
  - `retention_milestone_every` (Int > 0, default `save_every * 10`)
  - `retention_milestone_keep` (Int > 0, default `8`)

Runtime outputs for train/pretrain:
- `run_state.json` in `checkpoint_dir` with run status/identity/lineage.
- `runs/index.jsonl` append-only event log (`start|resume|checkpoint|completed|failed`).
- `checkpoint_lifecycle.json` with checkpoint integrity digests and retention metadata.

## Evaluation

```
Enkai eval config.enk
```

Uses the latest checkpoint and computes average loss/perplexity.

## Cluster orchestration helpers

```
enkai cluster validate config.enk [--json]
enkai cluster plan config.enk [--json]
enkai cluster run config.enk [--dry-run] [--json]
```

- `validate` checks config contract + distributed orchestration fields.
- `plan` emits deterministic per-rank launch env/command plans.
- `run` currently emits launch plans (operator launcher remains explicit).

## Common errors

- `tokenizer_path` or `tokenizer_train` missing.
- `checkpoint_dir` not writable.
- `config_version` missing or not `1` for v1 configs.
- `backend`, `dtype`, or `device` invalid for the selected backend.


