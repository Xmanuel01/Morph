# FFI (`native::import`)

Enkai can call native C ABI functions via `native::import` blocks.
FFI is an advanced systems feature. Prefer standard-library modules first; use
FFI only when a native boundary is required and audited.

## Syntax

```enkai
import std::io
import std::json

native::import "enkai_native" ::
    fn add_i64(a: Int, b: Int) -> Int
    fn echo_string(data: String) -> String
    fn handle_new(seed: Int) -> Handle
    fn handle_read(handle: Handle) -> Int
::native

policy default ::
    allow io.write
::policy

fn main() ::
    let x := add_i64(2, 3)
    let _ := io.stdout_write_text("result=" + json.enkai(x) + "\n")
::fn
```

`native::import` is a top-level declaration and must appear before other items.
Tagged `::native` closers are preferred when supported by the parser; plain
`::` remains valid for compatibility.

## Supported Types

- `Int` maps to `i64`.
- `Float` maps to `f64`.
- `Bool` maps to `u8` (`0` or `1`).
- `String` maps to `(ptr, len)` UTF-8 bytes.
- `Buffer` maps to `(ptr, len)` bytes.
- `Handle` maps to an opaque native pointer managed by Enkai.
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

## Safety Guidance

- Do not use FFI to bypass Enkai policy checks for normal application IO.
- Keep native APIs small and versioned.
- Return deterministic error codes/messages from native code.
- Treat unaudited native libraries as outside the strict self-host proof surface.
