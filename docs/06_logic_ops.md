# Logic And Operators

## Arithmetic

```enkai
let a := 10 + 2
let b := 10 - 2
let c := 10 * 2
let d := 10 / 2
```

Arithmetic supports `Int` and `Float`. Mixed integer/float arithmetic promotes
toward `Float` where supported.

## Comparison

```enkai
let high := score >= 0.70
let same := name == "Nairobi"
let different := name != "Mombasa"
```

Comparisons return `Bool`.

## Boolean Logic

```enkai
if ready and allowed ::
    line("go")
::if

if not denied ::
    line("allowed")
::if
```

Use parentheses when precedence could be unclear.

## String Concatenation

```enkai
let label := "county=" + name
```

Convert structured data to text with explicit modules such as `std::json`.
