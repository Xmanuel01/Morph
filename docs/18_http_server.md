# HTTP Server

Enkai includes a production-oriented HTTP runtime with routing, middleware, and streaming primitives.

## Usage

```
fn handler(req: Request) -> Response ::
    return http.ok("hello")
::

http.serve("127.0.0.1", 8080, handler)
```

Routed server:

```
fn get_item(req: Request) -> Response ::
    return http.ok(req.params.id)
::

let routes := [http.route("GET", "/items/:id", get_item)]
http.serve_with("127.0.0.1", 8080, routes, none)
```

Middleware server:

```
let auth := json.parse("{\"tokens\":[{\"token\":\"secret\",\"tenant\":\"acme\"}]}")
let middlewares := [http.middleware("auth", auth)]
http.serve_with("127.0.0.1", 8080, routes, middlewares)
```

Backpressure middleware:

```
let middlewares := [http.middleware("backpressure", json.parse("{\"max_inflight\":64}"))]
http.serve_with("127.0.0.1", 8080, routes, middlewares)
```

## Request/Response

`Request` fields:

- `method` (String)
- `path` (String)
- `query` (String)
- `headers` (Record)
- `body` (Buffer)
- `params` (Record, populated from `:param` route segments)
- `remote_addr` (String)

`Response` helpers:

- `http.ok(body)`
- `http.bad_request(body)`
- `http.not_found(body)`
- `http.response(status, body)`

Request helpers:

- `http.header(req, name)`
- `http.query(req, name)`

Streaming helpers:

- `http.stream_open(status, headers)`
- `http.stream_send(stream, chunk)`
- `http.stream_close(stream)`

WebSocket helpers (server-side):

- `http.ws_open(req)` upgrades an incoming HTTP request to WebSocket.
- `http.ws_send(ws, message)` sends text/binary frames (String/Buffer).
- `http.ws_recv(ws, timeout_ms)` receives inbound text/binary frames; returns `none` on timeout or closed session.
- `http.ws_close(ws)` sends close frame and terminates connection.

## Common errors

- Handler must be `fn(Request) -> Response`.
- Unsupported return types from handler result in a 500 response.
- Missing/invalid auth token returns 401 JSON error.
- Rate-limit violations return 429 JSON error.
- Backpressure limits return `503` with `error.code = "backpressure_overloaded"`.

## Middleware notes

- `rate_limit.key` supports:
  - `ip` (default)
  - `token`
  - `tenant`
  - `model`
  - `tenant_model`
- `backpressure` middleware uses `max_inflight` (`Int >= 0`), where `0` disables the cap.

## Observability headers

Every response includes `x-enkai-request-id`. When available, runtime also attaches:

- `x-enkai-correlation-id`
- `x-enkai-queue-ms`
- `x-enkai-latency-ms`
- `x-enkai-inflight`
- `x-enkai-tenant`
- `x-enkai-model-name`
- `x-enkai-model-version`
- `x-enkai-model-registry`
- `x-enkai-error-code` (for deterministic machine parsing)


