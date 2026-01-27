# Channels

Channels provide message passing between tasks.

## Usage

```
let ch := chan.make()
chan.send(ch, 123)
let value := chan.recv(ch)
```

## Example

```
let ch := chan.make()

fn sender() -> Int ::
    chan.send(ch, 7)
    return 0
::

let h := task.spawn(sender)
print(chan.recv(ch)) // expected: 7
task.join(h)
```

## Common errors

- `chan.send` expects a Channel as the first argument.
- `chan.recv` blocks until a value is available.


