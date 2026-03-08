# 35. Data + Algorithm Stack (v2.1.4)

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
- `shortest_path(edges, start, goal)`
- `count_frequencies(values)`
- `window_sum(values, window)`
- `accuracy(pred, target)`
- `mse(pred, target)`

These APIs are native-backed and intended as baseline building blocks for software and ML utility layers.
