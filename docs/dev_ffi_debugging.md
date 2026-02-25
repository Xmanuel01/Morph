# FFI Debugging

## Trace library loading

- Ensure the native library file is in the current directory, the executable
  directory, or set `ENKAI_STD` for std modules.
- On Windows, the loader will also try `enkai_native.dll`.

## Typical failures

- `Failed to load library 'enkai_native'` -> file not found or wrong name.
- `Failed to resolve symbol 'foo'` -> missing export or name mismatch.

## ABI checks

- `String` and `Buffer` must be `(ptr, len)`.
- Returning `String`/`Buffer` requires `enkai_free(ptr, len)` in the library.
- Optional `String`/`Buffer` can be null pointer with length 0.

## Debugging tips

- Start with a trivial function like `add_i64`.
- Verify signature order and arity.
- Use `native::import` in a minimal file to isolate issues.
