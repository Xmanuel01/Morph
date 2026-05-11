# Errors And Diagnostics

Use `enkai check` before `enkai run` when you want fast feedback without
executing side effects:

```powershell
enkai check main.enkai
```

## Common Error Classes

```text
SyntaxError     invalid tokens, block structure, or tagged closer mismatch
TypeError       invalid assignment, return type, call arity, or inference result
ImportError     missing module or missing required import
PolicyError     denied or unallowed side effect under policy default
RuntimeError    deterministic runtime failure during execution
```

## Examples

Mismatched block closer:

```text
SyntaxError: expected ::while to close while block, found ::if
```

Immutable reassignment:

```text
TypeError: cannot assign to immutable variable `count`; declare it with `mut` if mutation is required.
```

Missing JSON import:

```text
ImportError: `json.enkai` requires `import std::json`
```

Denied policy:

```text
PolicyError: io.write is denied by policy `default`.
```

## Debugging Flow

1. Run `enkai fmt --check <file>` to catch formatting drift.
2. Run `enkai check <file>` to catch syntax, type, import, and policy errors.
3. Run `enkai run <file>` only after checks pass.
4. Use `enkai run --trace-vm <file>` or `enkai run --disasm <file>` for VM-level debugging.
