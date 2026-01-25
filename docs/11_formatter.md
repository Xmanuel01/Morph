# Formatter (morph fmt)

Morph includes a deterministic formatter.

## Usage

```
morph fmt path/to/file.morph
morph fmt ./project
```

Check-only mode:

```
morph fmt --check path/to/file.morph
```

## What it formats

- indentation inside `::` blocks
- spacing around operators
- consistent import and `let` layout

## Common error

```
morph fmt --check src/main.morph
```

If the file is unformatted, the command exits with a non-zero code.
