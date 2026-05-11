# Functions

Functions are declared with `fn` and use `::` blocks.

## Basic Function

```enkai
fn add(a: Int, b: Int) -> Int ::
    return a + b
::fn
```

Call it with normal call syntax:

```enkai
let result := add(1, 2)
```

## Void Functions

If a return type is omitted, the function returns `Void`:

```enkai
fn log_ready() ::
    line("ready")
::fn
```

## Parameters

Parameters are immutable by default:

```enkai
fn double(value: Int) -> Int ::
    return value * 2
::fn
```

If a local value must change, copy it into a `mut` binding:

```enkai
fn count_to(limit: Int) ::
    mut index := 0
    while index < limit ::
        index := index + 1
    ::while
::fn
```

## Return Rules

The returned expression must match the declared return type:

```enkai
fn ok() -> Bool ::
    return true
::fn
```

A mismatch is a type error caught by `enkai check`.

## Style

Use tagged `::fn` closers for functions in committed code. Plain `::` remains
valid for short scripts and older code.
