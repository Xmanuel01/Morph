# Serve CLI

`enkai serve` runs an Enkai program as a serving entrypoint and injects serving env vars.

## Command

```text
enkai serve [--host <host>] [--port <port>] \
  [--registry <dir> --model <name> [--model-version <v> | --latest] | --checkpoint <path>] \
  [--trace-vm] [--disasm] [--trace-task] [--trace-net] [file|dir]
```

## Model Resolution

You must choose one model source:

- Direct checkpoint path:
  - `--checkpoint <path>`
- Registry lookup:
  - `--registry <dir> --model <name> --model-version <v>`
  - `--registry <dir> --model <name> --latest`

For registry mode, Enkai resolves:

- `<registry>/<model>/<version>/checkpoint` if it exists.
- Otherwise `<registry>/<model>/<version>`.

## Injected Env Vars

- `ENKAI_SERVE_HOST`
- `ENKAI_SERVE_PORT`
- `ENKAI_SERVE_MODEL_PATH`
- `ENKAI_SERVE_MODEL_NAME`
- `ENKAI_SERVE_MODEL_VERSION`
- `ENKAI_SERVE_MODEL_REGISTRY`
- `ENKAI_REQUIRE_MODEL_VERSION_HEADER` (`1|true|yes|on` enables strict request header enforcement)
- `ENKAI_HTTP_MAX_INFLIGHT` (`Int >= 0`; `0` disables global inflight cap)

Programs can read these through `std::env` or `std::model_registry`.

## Runtime Protocol Notes

- HTTP routed handlers and middleware are available through `std::http`.
- Streaming responses can use:
  - chunked HTTP stream APIs (`http.stream_open/http.stream_send/http.stream_close`)
  - SSE helpers (`std::http.sse_*`)
  - server-side WebSocket helpers (`std::http.ws_open/ws_send/ws_close`)
- Model pin enforcement:
  - if `ENKAI_REQUIRE_MODEL_VERSION_HEADER=1`, requests missing `x-enkai-model-version` return `400 missing_model_version`
  - mismatched model version returns `409 model_version_mismatch`
  - mismatched model name (when provided) returns `409 model_name_mismatch`
- Deterministic serving metadata headers are attached for observability:
  - `x-enkai-request-id`, `x-enkai-correlation-id`, `x-enkai-queue-ms`, `x-enkai-latency-ms`,
    `x-enkai-inflight`, and model tags when configured.
