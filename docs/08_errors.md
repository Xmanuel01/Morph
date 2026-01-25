# Errors

- Parse errors: line/col with snippet.
- Type errors: show location, expected vs found.
- Runtime errors: safe messages (stack underflow, undefined global, etc.).

Tips:
- Run `morph check file.morph` to catch type issues.
- Use `--trace-vm` to see execution flow.

## Diagnostic format (v0.5)

Compiler/type errors now include structured diagnostics with source snippets:

```
error: Symbol 'secret' is private to module app::utils
--> path/to/main.morph:5:18
  |
5 |     return utils.secret(x)
  |                  ^
```

Runtime errors include a lightweight stack trace (function + source + line):

```
Runtime error: Division by zero
  at main (path/to/main.morph:3)
  at <bootstrap> (path/to/main.morph:1)
```
