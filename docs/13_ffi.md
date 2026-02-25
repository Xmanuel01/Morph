# FFI (native::import)

Enkai can call native C ABI functions via `native::import` blocks.

## Syntax

```
native::import "enkai_native" ::
    fn add_i64(a: Int, b: Int) -> Int
    fn echo_string(data: String) -> String
::

let x := add_i64(2, 3)
print(x)
```

`native::import` is a top-level declaration and must appear before other items.

## Supported types (v0.6)

- `Int` -> `i64`
- `Float` -> `f64`
- `Bool` -> `u8` (`0` or `1`)
- `String` -> `(ptr, len)` bytes (UTF-8)
- `Buffer` -> `(ptr, len)` bytes
- `Void` return only
- `Optional[T]` for `String`/`Buffer` (null pointer + len 0)

## Ownership rules

If a native function returns `String` or `Buffer`, it must return a `{ptr,len}`
allocated with a compatible allocator and expose a `enkai_free(ptr, len)`
function in the same library. Enkai will call `enkai_free` to release it
after copying the bytes into VM-owned memory.

## Common errors

- Library not found
- Symbol missing
- Signature mismatch

The VM returns a clean runtime error instead of panicking.
