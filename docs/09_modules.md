# Modules (v0.5)

Morph supports multi-file modules via `import`.

## Syntax

```
import app::utils
import app::math as math

fn main() -> Int ::
    return utils.add(1, 2) + math.mul(2, 3)
::
```

## Resolution

- `import app::utils` resolves to `./app/utils.morph`
- The module root is the directory that contains the entry file passed to `morph run` or `morph check`.

## Accessing exports

Only exported (public) symbols are accessible:

```
import app::utils as utils

fn main() -> Int ::
    return utils.add(1, 2)
::
```

## Common errors

- `Module not found: app::utils` — the file `app/utils.morph` is missing.
- `Circular import detected: main -> app::utils -> main`
