# TCP Networking

The `net` module exposes basic TCP networking.

## Server

```
let listener := net.bind("127.0.0.1", 8080)
let conn := listener.accept()
let data := conn.read(5)
conn.write(data)
conn.close()
```

## Methods

- `net.bind(host, port) -> TcpListener`
- `listener.accept() -> TcpConnection`
- `conn.read(n) -> Buffer`
- `conn.read_all() -> Buffer`
- `conn.write(buf) -> Int`
- `conn.close() -> Void`

## Common errors

- `net.bind` expects a string host and integer port.
- `conn.read` expects an integer byte count.

