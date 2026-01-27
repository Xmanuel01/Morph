# Testing (Enkai test)

Enkai can run test programs in `/tests`.

## Usage

```
Enkai test
```

Optionally pass a project root:

```
Enkai test ./my_project
```

## Conventions

- Tests are `.enk` files in `./tests/`.
- Each test runs: `check -> compile -> VM`.
- A test passes if it exits without errors.
- A test fails if type-checking or runtime fails.

## Example

```
// tests/smoke.enk
fn main() -> Int ::
    return 0
::
```


