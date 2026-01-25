# Testing (morph test)

Morph can run test programs in `/tests`.

## Usage

```
morph test
```

Optionally pass a project root:

```
morph test ./my_project
```

## Conventions

- Tests are `.morph` files in `./tests/`.
- Each test runs: `check -> compile -> VM`.
- A test passes if it exits without errors.
- A test fails if type-checking or runtime fails.

## Example

```
// tests/smoke.morph
fn main() -> Int ::
    return 0
::
```
