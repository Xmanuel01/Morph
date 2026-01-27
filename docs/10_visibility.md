# Visibility (public/private)

Symbols are private by default. Use `pub` to export.

## Functions

```
// app/utils.enk
pub fn add(a: Int, b: Int) -> Int ::
    return a + b
::

fn secret(x: Int) -> Int ::
    return x
::
```

```
// main.enk
import app::utils as utils

fn main() -> Int ::
    return utils.add(1, 2)
::
```

## Common error

Accessing a private symbol:

```
return utils.secret(1)
```

Produces:

```
error: Symbol 'secret' is private to module app::utils
```


