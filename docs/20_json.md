# JSON

`std::json` provides JSON parsing and encoding. Import it explicitly before using
`json.*`.

```enkai
import std::json

let value := json.parse("{\"ok\":true}")
let text := json.enkai(value)
```

## Encoding

`json.enkai(value)` is the preferred Enkai spelling for converting a value to
JSON text:

```enkai
let payload := json.enkai({"county": "Nairobi", "score": 0.92})
```

`json.stringify(value)` remains available as a compatibility alias for older
code, but new code should use `json.enkai(value)`.

## Parsing

```enkai
let parsed := json.parse("[1, 2, 3]")
```

Invalid JSON produces a deterministic runtime error.

## Import Rule

Using JSON without importing the module is an error:

```enkai
let text := json.enkai("hello")
```

```text
ImportError: `json.enkai` requires `import std::json`
```

## Supported Values

JSON encoding supports primitives, arrays, JSON values, and records/objects that
map cleanly to JSON objects. Unsupported values fail with a clear error instead
of silently producing invalid JSON.
