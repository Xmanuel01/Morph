# Formatter (Enkai fmt)

Enkai includes a deterministic formatter.

## Usage

```
Enkai fmt path/to/file.enk
Enkai fmt ./project
```

Check-only mode:

```
Enkai fmt --check path/to/file.enk
```

## What it formats

- indentation inside `::` blocks
- spacing around operators
- consistent import and `let` layout

## Common error

```
Enkai fmt --check src/main.enk
```

If the file is unformatted, the command exits with a non-zero code.


