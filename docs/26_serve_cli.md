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

Programs can read these through `std::env` or `std::model_registry`.
