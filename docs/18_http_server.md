# HTTP Server

Morph includes a minimal HTTP server API.

## Usage

```
fn handler(req: Request) -> Response ::
    return http.ok("hello")
::

http.serve("127.0.0.1", 8080, handler)
```

## Request/Response

`Request` fields:

- `method` (String)
- `path` (String)
- `query` (String)
- `headers` (Record)
- `body` (Buffer)

`Response` helpers:

- `http.ok(body)`
- `http.bad_request(body)`
- `http.not_found(body)`
- `http.response(status, body)`

## Common errors

- Handler must be `fn(Request) -> Response`.
- Unsupported return types from handler result in a 500 response.

