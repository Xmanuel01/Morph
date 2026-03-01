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

## Common errors

- Handler must be `fn(Request) -> Response`.
- Unsupported return types from handler result in a 500 response.
- Missing/invalid auth token returns 401 JSON error.
- Rate-limit violations return 429 JSON error.


