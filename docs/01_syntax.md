# Enkai Syntax

This page teaches the day-to-day syntax used in normal Enkai programs.
For the complete normative reference, see `docs/Enkai.spec`.

## Comments

```enkai
// This is a line comment.
```

## Imports

Everything outside the core language must be imported explicitly:

```enkai
import std::io
import std::json
import std::tensor
```

A standard-library namespace is normally used without the `std::` prefix after
importing it:

```enkai
let text := json.enkai({"ok": true})
```

Using `json.*` without `import std::json` is an import error.

## Blocks

Enkai uses `::` as its visual block marker:

```enkai
if ready ::
    line("ready")
::
```

Plain `::` closers are valid for short code. Tagged closers are preferred for
functions, policies, loops, nested blocks, and blocks longer than a few lines:

```enkai
policy default ::
    allow io.write
::policy

fn clamp01(value: Float) -> Float ::
    if value < 0.0 ::
        return 0.0
    ::if

    if value > 1.0 ::
        return 1.0
    ::if

    return value
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
::module
```

A mismatched tagged closer is a syntax error:

```text
SyntaxError: expected ::while to close while block, found ::if
```

## Variables

`let` creates an immutable binding:

```enkai
let name := "Nairobi"
let age: Int := 22
```

`mut` creates a mutable binding:

```enkai
mut score := 0.0
score := score + 1.0
```

`const` creates a compile-time constant:

```enkai
const PI := 3.14159
const MAX_BATCH := 128
```

Reassigning a `let` or `const` binding is rejected:

```enkai
let count := 6
count := 7      // error
```

```text
TypeError: cannot assign to immutable variable `count`; declare it with `mut` if mutation is required.
```

## Functions

```enkai
fn add(a: Int, b: Int) -> Int ::
    return a + b
::fn
```

If no return type is written, the function returns `Void`:

```enkai
fn greet(name: String) ::
    line("hello " + name)
::fn
```

Function parameters are immutable by default.

## Conditionals

```enkai
if score >= 0.70 ::
    return "strong"
::if else ::
    return "needs_review"
::else
```

Plain closers remain valid:

```enkai
if score >= 0.70 ::
    return "strong"
::
```

## Loops

Use `mut` for loop counters because counters are reassigned:

```enkai
mut index := 0
while index < 3 ::
    line("tick")
    index := index + 1
::while
```

## Arrays And AI-Native Collections

Array literals infer homogeneous element types:

```enkai
let names := ["Nairobi", "Mombasa"]      // Array[String]
let scores := [0.7, 0.8, 1]              // Array[Float]
let empty: Array[String] := []           // explicit type required
```

Mixed-type arrays require an explicit dynamic type if the program intends that:

```enkai
let values: Array[Any] := ["Nairobi", 42, true]
```

Use explicit AI-native annotations for mathematical structures:

```enkai
let vector: Vector[Float] := [0.1, 0.2]
let sparse_values: SparseVector[Float] := sparse.vector()
let matrix: Tensor[Float, 2] := tensor.zeros([3, 3], "f32", 0)
```

## Policies

Policy blocks declare allowed and denied effects:

```enkai
import std::io

policy default ::
    allow io.write
    deny net.request
::policy
```

A call to a known side-effecting standard-library function is checked against
`policy default` before execution when possible.

## Complete Example

```enkai
import std::io
import std::json

policy default ::
    allow io.write
::policy

fn line(text: String) ::
    let _ := io.stdout_write_text(text)
    let _n := io.stdout_write_text("\n")
::fn

fn main() ::
    const COUNTRY := "Kenya"
    let counties: Array[String] := ["Nairobi", "Mombasa", "Kisumu"]
    mut index := 0

    while index < counties.length ::
        line(COUNTRY + " county=" + json.enkai(counties[index]))
        index := index + 1
    ::while
::fn
```
