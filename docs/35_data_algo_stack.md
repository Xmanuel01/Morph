# 35. Data + Algorithm Stack (v2.1.7)

Enkai ships additive std modules for data analysis and algorithm development.

## std::analysis

Primary APIs:
- `read_csv(path, delimiter, has_header)`
- `read_jsonl(path)`
- `infer_schema(rows)`
- `infer_schema_typed(rows)`
- `validate_schema(rows, schema)`
- `filter_eq(rows, field, value)`
- `project(rows, columns)`
- `group_sum(rows, key, field)`
- `group_agg(rows, key, field, agg)` (`sum|count|mean|min|max`)
- `join(left_rows, right_rows, left_key, right_key, how)` (`inner|left|right|outer`)
- `describe(values)`
- `histogram(values, bins)`
- `quantiles(values, quantiles)`
- `rolling_mean(values, window)`
- `run_pipeline(rows, pipeline)`

Behavior notes:
- JSON interchange is deterministic.
- Invalid parse/input returns empty/default JSON values (module contract).
- `validate_schema` returns deterministic machine-readable reports:
  - `ok`, `row_count`, `field_count`, `errors`, `rows`
- `run_pipeline` returns:
  - `rows`: transformed output rows
  - `manifest`: reproducibility metadata (`schema_version`, stage stats, hashes)

## std::algo

Primary APIs:
- `sort_ints(values)`
- `binary_search_ints(values, target)`
- `top_k_ints(values, k)`
- `merge_sorted_ints(left, right)`
- `shortest_path(edges, start, goal)`
- `count_frequencies(values)`
- `merge_count_maps(left, right)`
- `window_sum(values, window)`
- `cumulative_sum(values)`
- `window_mean(values, window)`
- `accuracy(pred, target)`
- `mse(pred, target)`
- `mae(pred, target)`
- `rmse(pred, target)`
- `precision_recall_f1(pred, target, positive_label)`
- `split_indices(total, test_ratio, seed, shuffle)`
- `scheduler_linear_warmup(step, total_steps, warmup_steps, base_lr, min_lr)`

Behavior notes:
- `top_k_ints` uses deterministic priority-queue selection semantics.
- `merge_count_maps` returns deterministically ordered keys.
- `split_indices` is deterministic by `(seed, total, ratio, shuffle)` and returns `{train,test}` index sets.
- metric helpers return deterministic numeric outputs and machine-readable objects for PR/F1.

Validation and evidence:
- Runtime integration coverage in `enkairt/tests/ffi_modules.rs` for both baseline and golden corpus scenarios.
- Native correctness coverage in `enkai_native/src/lib.rs` unit tests.
- Complexity/perf baseline suite: `bench/suites/algorithm_kernels.json`.
