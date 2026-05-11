# Modules And Imports

Enkai supports explicit imports for standard-library modules and local modules.

## Standard Library Imports

```enkai
import std::io
import std::json
import std::tensor
```

After importing `std::json`, use the namespace as `json`:

```enkai
let text := json.enkai(value)
```

Everything outside the core language should be explicitly imported. Missing
imports are reported by `enkai check` where possible.

## Local Modules

```enkai
import app::utils
import app::math as math
```

`import app::utils` resolves to `./app/utils.enk` relative to the entry file or
project root.

## Accessing Exports

Only exported public symbols are available across modules:

```enkai
import app::utils as utils

fn main() -> Int ::
    return utils.add(1, 2)
::fn
```

See `docs/10_visibility.md` for export rules.

## Common Errors

```text
Module not found: app::utils
Circular import detected: main -> app::utils -> main
ImportError: `json.enkai` requires `import std::json`
```

## Style

Place imports at the top of the file, before policies and functions:

```enkai
import std::io
import std::json

policy default ::
    allow io.write
::policy
```
