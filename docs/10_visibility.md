# Visibility

Visibility controls which module symbols can be used by other files.

## Public Exports

Use the module's public/export form for functions, constants, and types that are
part of the module API. Keep helper functions private unless another module needs
them.

```enkai
export fn add(a: Int, b: Int) -> Int ::
    return a + b
::fn
```

## Private Helpers

A function without an export marker is private to its module:

```enkai
fn normalize(value: Float) -> Float ::
    return value
::fn
```

## API Design Rule

Export the smallest stable surface. This keeps packages easier to evolve and
makes `enkai check` diagnostics clearer for users.
