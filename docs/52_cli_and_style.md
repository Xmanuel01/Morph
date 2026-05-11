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
const PI := 3.14159
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

Use `const` for immutable compile-time constants. A `const` initializer may use literals,
constant identifiers, unary operators, binary operators, and literal arrays. Runtime calls are
rejected:

```enkai
const BASE := 40
const ANSWER := BASE + 2
```

```text
TypeError: const binding `VALUE` requires a compile-time constant expression.
```

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

Runtime helpers live in explicit std modules:

```enkai
import std::array
import std::vector
import std::tensor

let values := [1.0, 2.0, 3.0]
let values_kind := array.element_type(values)
let v := vector.from_array(values)
let scaled := vector.scale(v, 0.5)
let score := vector.dot(v, scaled)

let t := tensor.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])
let out := tensor.matmul(t, t)
let shifted := tensor.add(out, tensor.from_array([1.0, 2.0], [1, 2]))
let probs := tensor.softmax(out, 1)
let reduced := tensor.mean(probs, 1, false)
let selected := tensor.topk(out, 1, 1)
let out_shape := tensor.shape(out)

let tracked := tensor.requires_grad(t)
let loss := tensor.sum(tensor.mul(tracked, tracked), 0, false)
tensor.backward(loss)
let grad := tensor.grad(tracked)
```

`Array[T]`, `Vector[Float]`, and `Tensor` are therefore checked and executable on the bounded first-party runtime path. Tensor operations in this path include construction, conversion, shape access, NumPy-style broadcasting for elementwise arithmetic, matrix multiply, reshape, transpose, slice, concat, reductions, softmax, core activations, deterministic dropout, linear, embedding, layernorm, cross-entropy, gather/scatter, masking, sorting/top-k, limited einsum, conv2d, pooling, batchnorm, attention, and tape-backed autodiff controls, first-party SGD/AdamW single- and multi-parameter steps plus gradient clipping, live/peak tensor memory accounting, bounded allocation limits, and deterministic CPU-safe behavior. Larger accelerated kernels and multi-parameter native optimizer paths remain tracked by the tensor backend roadmap.

## Static Policy Checking

Policy blocks are checked before execution for known standard-library side effects:

```enkai
import std::io

policy default ::
    allow io.write
::policy

let _ := io.stdout_write_text("hello")
```

`enkai check` rejects calls that are denied or not explicitly allowed by `policy default`.
Known checks include `io.write`, `fs.read`, `fs.write`, `env.read`, `env.write`,
`process.spawn`, `process.control`, `db.read`, `db.write`, `net.tls`, `net.serve`,
`net.http`, `io.log`, and `time.sleep`.

Explicit `deny` rules override matching `allow` rules. Supported static filters are
`path_prefix` for filesystem paths and `domain` for network/TLS domains when the checked
argument is a string literal.
