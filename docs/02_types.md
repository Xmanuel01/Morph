# Types

Enkai is statically checked where the compiler has enough information, while the
runtime still reports deterministic errors for invalid dynamic values.

## Primitive Types

```text
Int       signed integer values
Float     floating-point values
Bool      true or false
String    UTF-8 text
Void      no useful return value
Any       explicit dynamic value escape hatch
```

## Optional Types

Use `T?` when a value may be absent:

```enkai
let maybe_name: String? := null
```

## Function Types

Function types use parameter types and a return type:

```text
fn(Int, Int) -> Int
fn(String) -> Void
```

## Arrays

Use `Array[T]` for ordered collections:

```enkai
let names: Array[String] := ["Nairobi", "Mombasa"]
let scores := [0.75, 0.80, 1]      // inferred as Array[Float]
let empty: Array[String] := []     // empty arrays need explicit type
```

Rules:

- homogeneous string arrays infer `Array[String]`
- homogeneous integer arrays infer `Array[Int]`
- mixed integer/float arrays infer `Array[Float]`
- empty arrays require an explicit type
- mixed unrelated element types require an explicit type such as `Array[Any]`

## AI-Native Numeric Types

The language reserves clear names for AI and numerical programming:

```enkai
let dense: Vector[Float] := [0.1, 0.2, 0.3]
let sparse_values: SparseVector[Float] := sparse.vector()
let matrix: Tensor[Float, 2] := tensor.zeros([3, 3], "f32", 0)
let image_batch: Tensor[Float, 4] := tensor.zeros([2, 3, 224, 224], "f32", 0)
```

`Array[T]` is a general collection. `Vector[Float]` is for mathematical dense
vectors. `SparseVector[Float]` is for sparse numerical data. `Tensor[T, N]` is
for shaped numerical data.

## Constants

`const` initializers must be compile-time constant expressions:

```enkai
const BASE := 40
const ANSWER := BASE + 2
```

Runtime calls are rejected in `const` initializers.

## Assignment Compatibility

`let` and function parameters are immutable. Use `mut` when a value must be
reassigned:

```enkai
mut epoch := 0
epoch := epoch + 1
```

The checker rejects reassignment to immutable bindings and rejects return values
that do not match the declared function return type.
