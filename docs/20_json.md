# JSON

Enkai includes JSON parsing and serialization via `std::json`.
JSON is not a core implicit namespace; source files must import it explicitly.

## Usage

```enkai
import std::json

let v := json.parse("{\"a\":1,\"b\":[true,null]}")
let out := json.stringify(v)
let native := json.enkai(v)
```

## Notes

- `json.*` requires `import std::json`.
- `json.parse` returns Records or Lists for objects/arrays.
- `json.stringify` accepts primitives, JSON values, and Records (mapped to JSON objects).
- `json.enkai` is the Enkai-preferred alias for `json.stringify`.
- `json.stringify` remains supported for compatibility with existing Enkai code.

## Common errors

- Invalid JSON strings return a runtime error.
- `json.stringify` fails on unsupported values.
- Missing `import std::json` is a check-time import error:
  `ImportError: json.parse requires import std::json`.

