# HTTP Client

Enkai provides basic HTTP client helpers.

## Usage

```
let resp := http.get("http://127.0.0.1:8080/")
print(resp.status)
```

```
let body := "payload"
let resp := http.post("http://127.0.0.1:8080/echo", body)
```

## Response shape

`Response` includes:

- `status` (Int)
- `headers` (Record)
- `body` (Buffer)

## Common errors

- Only `http://` URLs are supported (no TLS yet).
- Invalid URLs or unreachable hosts return runtime errors.


