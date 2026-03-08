# 35. Data + Algorithm Stack (v2.1 foundation)

Enkai ships additive std modules for data analysis and algorithm development.

## std::analysis

Primary APIs:
- `read_csv(path, delimiter, has_header)`
- `read_jsonl(path)`
- `infer_schema(rows)`
- `filter_eq(rows, field, value)`
- `project(rows, columns)`
- `group_sum(rows, key, field)`
- `describe(values)`
- `histogram(values, bins)`

Behavior notes:
- JSON interchange is deterministic.
- Invalid parse/input returns empty/default JSON values (module contract).

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
