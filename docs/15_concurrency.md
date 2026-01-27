# Concurrency (Tasks)

Enkai provides lightweight tasks (green threads) for cooperative concurrency.

## Usage

```
let handle := task.spawn(work)
let result := task.join(handle)
task.sleep(50)
```

## Example

```
fn work() -> Int ::
    task.sleep(10)
    return 42
::

let h := task.spawn(work)
print(task.join(h)) // expected: 42
```

## Common errors

- `task.spawn` expects a function with arity 0.
- `task.join` expects a TaskHandle.


