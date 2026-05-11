# Concurrency (Tasks)

Enkai provides lightweight tasks (green threads) for cooperative concurrency.
The task namespace is a runtime namespace. Use it when work can progress
cooperatively without OS-thread ownership in user code.

## Usage

```enkai
let handle := task.spawn(work)
let result := task.join(handle)
task.sleep(50)
```

## Example

```enkai
import std::io
import std::json

policy default ::
    allow io.write
    allow time.sleep
::policy

fn work() -> Int ::
    task.sleep(10)
    return 42
::fn

fn main() ::
    let h := task.spawn(work)
    let result := task.join(h)
    let _ := io.stdout_write_text(json.enkai(result) + "\n")
::fn
```

## Common errors

- `task.spawn` expects a function with arity 0.
- `task.join` expects a TaskHandle.
- Long-running tasks must yield or sleep when cooperative scheduling fairness is
  required.


