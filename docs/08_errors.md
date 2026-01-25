# Errors

- Parse errors: line/col with snippet.
- Type errors: show location, expected vs found.
- Runtime errors: safe messages (stack underflow, undefined global, etc.).

Tips:
- Run `morph check file.morph` to catch type issues.
- Use `--trace-vm` to see execution flow.
