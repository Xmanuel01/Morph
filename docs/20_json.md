# JSON

Enkai includes JSON parsing and serialization via `json`.

## Usage

```
let v := json.parse("{\"a\":1,\"b\":[true,null]}")
let out := json.stringify(v)
print(out)
```

## Notes

- `json.parse` returns Records or Lists for objects/arrays.
- `json.stringify` accepts primitives, JSON values, and Records (mapped to JSON objects).

## Common errors

- Invalid JSON strings return a runtime error.
- `json.stringify` fails on unsupported values.

