# Native Modules

Enkai v0.6 ships native-backed modules that wrap FFI for common tasks.

## std::fsx

Fast file IO helpers.

```
import std::fsx

let data := fsx.read_bytes("data.bin")
fsx.write_bytes("copy.bin", data)
```

## std::zstd

Compression helpers.

```
import std::zstd

let compressed := zstd.compress(data, 3)
let decompressed := zstd.decompress(compressed)
```

## std::hash

Hash helpers.

```
import std::hash

let digest := hash.sha256_from_string("hello")
```

## Common errors

- `import std::...` fails if `std/` is missing from your project root
  (or `ENKAI_STD` is not set).
- Native library `enkai_native` not found.

## CLI usage

```
enkai run main.enk
enkai check main.enk
```


