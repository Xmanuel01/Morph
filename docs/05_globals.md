# Globals

Top-level bindings are module globals. They are initialized before top-level code
or `main` uses them.

## Immutable Globals

Prefer immutable globals:

```enkai
const APP_NAME := "county-score"
let DEFAULT_REGION := "Kenya"
```

## Mutable Globals

Avoid mutable global state unless the runtime surface explicitly requires it.
Prefer passing values through functions or using local `mut` bindings.

## Imported Globals

Only public/exported symbols from another module can be accessed after import.
See `docs/09_modules.md` and `docs/10_visibility.md`.
