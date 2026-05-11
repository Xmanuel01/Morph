# Native Modules

Native-backed standard modules wrap audited FFI for common tasks. They are
normal Enkai modules from the user's point of view: import them explicitly and
declare the required policy permissions before side effects.

## std::fsx

Fast file IO helpers.

```enkai
import std::fsx

policy default ::
    allow fs.read
    allow fs.write
::policy

let data := fsx.read_bytes("data.bin")
let _ := fsx.write_bytes("copy.bin", data)
```

## std::zstd

Compression helpers.

```enkai
import std::zstd

let compressed := zstd.compress(data, 3)
let decompressed := zstd.decompress(compressed)
```

## std::hash

Hash helpers.

```enkai
import std::hash

let digest := hash.sha256_from_string("hello")
```

## Common errors

- `import std::...` fails if `std/` is missing from your project root
  (or `ENKAI_STD` is not set).
- Native library `enkai_native` not found.

## CLI usage

```powershell
enkai run main.enk
enkai check main.enk
```

## Learner Rule

Use `std::io`, `std::json`, `std::array`, `std::vector`, and `std::tensor`
before reaching for native-backed modules. Native modules are useful for
performance and platform integration, but they are also where portability and
policy auditing matter most.


