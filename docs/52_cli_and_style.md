# Enkai CLI and Style

## Official CLI

Use `enkai run` to execute Enkai source files:

```powershell
enkai run main.enkai
enkai run county_score.enkai
```

Core commands:

```text
enkai run <file.enkai>      Execute an Enkai source file.
enkai check <file.enkai>    Check syntax, types, imports, and policies without executing.
enkai build <file.enkai>    Build an Enkai source file or project.
enkai test [dir]            Run project tests.
enkai fmt <file.enkai>      Format Enkai source.
enkai version               Print the Enkai version.
enkai help                  Show CLI help.
```

`enkai safari` is reserved for a future interactive Enkai workspace. It is not the normal execution command.

## Block Style

Enkai keeps `::` as its visual block identity.

Short blocks may use anonymous closers:

```enkai
if ready ::
    line("done")
::
```

Longer or nested blocks should use tagged closers:

```enkai
fn example() ::
    while condition ::
        if ready ::
            line("done")
        ::if
    ::while
::fn
```

Supported tagged closers include:

```text
::fn
::if
::else
::while
::for
::policy
::match
::struct
::enum
::impl
```

Plain `::` remains valid for backward compatibility.
`enkai fmt` upgrades recognized anonymous closers to tagged closers in the official style.

## Imports

Prefer explicit imports for standard-library namespaces:

```enkai
import std::io
import std::sparse
import std::json
```

`json.enkai(value)` is the Enkai-preferred JSON text encoder. `json.stringify(value)` remains supported for compatibility.
Using `json.*` without `import std::json` is an import error.

## Mutability

Enkai bindings are immutable by default:

```enkai
let name := "Nairobi"
mut score := 0.0
```

Use `mut` when reassignment is required:

```enkai
mut epoch := 0
while epoch < 3 ::
    epoch := epoch + 1
::while
```

Reassigning a `let` binding is an error:

```text
TypeError: cannot assign to immutable variable `epoch`; declare it with `mut` if mutation is required.
```

Function parameters are immutable by default.

## Collections And AI Types

Use `Array[T]` for normal ordered collections:

```enkai
let counties: Array[String] := ["Nairobi", "Mombasa", "Kisumu"]
let scores: Array[Float] := [0.75, 0.68, 1]
let empty: Array[String] := []
```

Use explicit AI-native annotations when the value is intended for numeric or tensor work:

```enkai
let vector: Vector[Float] := [0.75, 0.68, 0.62]
let weights: SparseVector[Float] := sparse.vector()
let matrix: Tensor[Float, 2] := tensor.zeros([3, 3], "f32", 0)
```

Empty arrays require an explicit type. Mixed-type array bindings require an explicit dynamic type such as `Array[Any]`.
