# Scopes

A scope is the region where a name can be used. Blocks introduce nested scopes.

## Function Scope

Bindings declared inside a function are local to that function:

```enkai
fn example() ::
    let name := "Nairobi"
    line(name)
::fn
```

`name` is not available outside `example`.

## Block Scope

Bindings declared inside `if`, `while`, and other blocks are local to those
blocks:

```enkai
if ready ::
    let message := "ready"
    line(message)
::if
```

## Shadowing

Prefer clear names over shadowing. If shadowing is used, the inner binding only
applies to the inner scope.

## Mutation Across A Scope

Use `mut` in the scope where reassignment is needed:

```enkai
mut total := 0
while total < 10 ::
    total := total + 1
::while
```

`let` bindings cannot be reassigned from an inner or outer scope.
