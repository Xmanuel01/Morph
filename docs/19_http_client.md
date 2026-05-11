# HTTP Client

Enkai provides basic HTTP client helpers.

## Usage

```enkai
import std::http
import std::io
import std::json

policy default ::
    allow net.http
    allow io.write
::policy

let resp := http.get("http://127.0.0.1:8080/")
let _ := io.stdout_write_text(json.enkai(resp.status) + "\n")
```

```enkai
import std::http

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
- Missing `net.http` permission is a policy error when static policy checking
  can prove the call.


