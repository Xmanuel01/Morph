# Channels

Channels provide message passing between tasks.

## Usage

```enkai
let ch := chan.make()
chan.send(ch, 123)
let value := chan.recv(ch)
```

## Example

```enkai
import std::io
import std::json

policy default ::
    allow io.write
::policy

let ch := chan.make()

fn sender() -> Int ::
    chan.send(ch, 7)
    return 0
::fn

fn main() ::
    let h := task.spawn(sender)
    let value := chan.recv(ch)
    let _ := io.stdout_write_text(json.enkai(value) + "\n")
    let _joined := task.join(h)
::fn
```

## Common errors

- `chan.send` expects a Channel as the first argument.
- `chan.recv` blocks until a value is available.
- Use bounded protocols or timeouts at higher levels when a sender may fail.


