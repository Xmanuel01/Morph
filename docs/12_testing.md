# Testing

Use `enkai test` to run project tests.

## Run Tests

```powershell
enkai test
```

Run tests for a specific project directory:

```powershell
enkai test .\my_project
```

## Recommended Local Loop

```powershell
enkai fmt --check .
enkai check .
enkai test .
```

Then run the program:

```powershell
enkai run .\my_project\main.enkai
```

## What To Test

- public functions
- policy-denied paths
- import errors for missing modules
- tensor/data/checkpoint deterministic behavior when using AI-native modules
- CLI entry points for real projects
