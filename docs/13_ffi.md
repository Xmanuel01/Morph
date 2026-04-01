# FFI (native::import)

Enkai can call native C ABI functions via `native::import` blocks.

## Syntax

```
native::import "enkai_native" ::
    fn add_i64(a: Int, b: Int) -> Int
    fn echo_string(data: String) -> String
    fn handle_new(seed: Int) -> Handle
    fn handle_read(handle: Handle) -> Int
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
- `Handle` -> opaque native pointer managed by Enkai
- `Void` return only
- `Optional[T]` for `String`/`Buffer`/`Handle`

## Ownership rules

If a native function returns `String` or `Buffer`, it must return a `{ptr,len}`
allocated with a compatible allocator and expose a `enkai_free(ptr, len)`
function in the same library. Enkai will call `enkai_free` to release it
after copying the bytes into VM-owned memory.

If a native function returns `Handle`, it must return an opaque pointer and expose
`enkai_handle_free(ptr)` in the same library. Enkai treats the value as opaque and
will call `enkai_handle_free` automatically when the last VM reference is dropped.

## ABI policy

Official native extensions should expose:

- `enkai_abi_version() -> Int`
- `enkai_symbol_table() -> String`

`enkai_symbol_table` must return JSON:

```json
{"abi_version":1,"exports":["symbol_a","symbol_b"]}
```

`["*"]` is accepted for broad libraries, but exact symbol lists are preferred for
new simulation-grade native modules. If a library exports either ABI symbol, it
must export both.

## Profiling visibility

VM benchmark profiles record:

- native call count
- marshaled bytes in/out
- marshal/copy operation count
- native handle object count
- time spent in native calls vs VM execution

## Common errors

- Library not found
- Symbol missing
- Signature mismatch
- ABI version mismatch
- Missing `enkai_free` / `enkai_handle_free`

The VM returns a clean runtime error instead of panicking.
