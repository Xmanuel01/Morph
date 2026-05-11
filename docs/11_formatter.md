# Formatter

`enkai fmt` applies the official source style.

## Usage

```powershell
enkai fmt path\to\file.enk
enkai fmt .\project
```

Check-only mode:

```powershell
enkai fmt --check path\to\file.enk
```

## What It Formats

- indentation inside `::` blocks
- spacing around common operators
- consistent import and binding layout
- recognized anonymous closers upgraded to tagged closers where the style prefers it

Example output style:

```enkai
fn example() ::
    while condition ::
        if ready ::
            line("done")
        ::if
    ::while
::fn
```

## CI Usage

Use check-only mode in scripts and CI:

```powershell
enkai fmt --check src\main.enk
```

If a file is unformatted, the command exits with a non-zero status.
