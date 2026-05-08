# Enkai v0.4 Syntax (quick)

- Blocks use `::` to open. A plain `::` closes any block, and tagged closers such as `::fn`, `::if`, `::while`, and `::policy` can close larger or nested blocks.
- Immutable bindings: `let name := expr`
- Mutable bindings: `mut name := expr` or `let mut name := expr`
- Functions: `fn name(params) -> Type :: ... ::`
- If/else: `if cond :: ... :: else :: ... ::`
- While: `while cond :: ... ::`
- Arrays: `let names: Array[String] := ["Nairobi", "Mombasa"]`
- AI-native collection annotations: `Vector[Float]`, `SparseVector[Float]`, `Tensor[Float, 2]`
- Comments: `// line`

## Tagged Closers

Plain closers remain valid:

```enkai
if score >= 0.70 ::
    return "strong"
::
```

Tagged closers are preferred for functions, policies, loops, nested conditionals, and longer blocks:

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

If a tagged closer is mismatched, Enkai reports a syntax error, for example:

```text
SyntaxError: expected ::while to close while block, found ::if
```

`enkai fmt` upgrades recognized anonymous closers to tagged closers in the official style, for example `::fn`, `::if`, `::while`, and `::policy`.

## Collections

Array literals infer homogeneous element types:

```enkai
let names := ["Nairobi", "Mombasa"]      // Array[String]
let scores := [0.7, 0.8, 1]              // Array[Float]
let empty: Array[String] := []           // explicit type required
let vector: Vector[Float] := [0.1, 0.2]
```

Empty arrays require an explicit type. Mixed-type array bindings require an explicit dynamic type such as `Array[Any]`.

