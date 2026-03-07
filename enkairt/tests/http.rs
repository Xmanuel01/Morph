use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Mutex, OnceLock};

use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
use enkairt::error::RuntimeError;
use enkairt::{Value, VM};

const POLICY_ALLOW_NET: &str = "policy default ::\n    allow net\n::\n\n";

fn inject_policy(source: &str) -> String {
    if source.contains("policy ") {
        return source.to_string();
    }
    let mut insert_at = 0usize;
    let mut in_native = false;
    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if in_native {
            if trimmed == "::" {
                in_native = false;
                insert_at = idx + 1;
            }
            continue;
        }
        if trimmed.starts_with("import ") {
            insert_at = idx + 1;
            continue;
        }
        if trimmed.starts_with("native::import ") {
            in_native = true;
            insert_at = idx + 1;
            continue;
        }
        break;
    }
    let mut out = String::new();
    let mut inserted = false;
    for (idx, line) in source.lines().enumerate() {
        if !inserted && idx == insert_at {
            out.push_str(POLICY_ALLOW_NET);
            inserted = true;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !inserted {
        out.push_str(POLICY_ALLOW_NET);
    }
    out
}

fn run_value(source: &str) -> Value {
    let source = inject_policy(source);
    let module = parse_module(&source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program).expect("run")
}

fn run_result_raw(source: &str) -> Result<Value, RuntimeError> {
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program)
}

fn http_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

fn response_body(value: &Value) -> Option<Vec<u8>> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            enkairt::object::Obj::Record(map) => {
                let map = map.borrow();
                match map.get("body") {
                    Some(Value::Obj(obj)) => match obj.as_obj() {
                        enkairt::object::Obj::Buffer(bytes) => Some(bytes.clone()),
                        enkairt::object::Obj::String(s) => Some(s.as_bytes().to_vec()),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        },
        _ => None,
    }
}

fn send_raw_request(port: u16, request: &[u8]) -> Vec<u8> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream.write_all(request).expect("write");
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(err) if err.kind() == std::io::ErrorKind::ConnectionReset => break,
            Err(err) => panic!("read: {}", err),
        }
    }
    buf
}

fn response_status(buf: &[u8]) -> u16 {
    let text = String::from_utf8_lossy(buf);
    let line = text.lines().next().unwrap_or_default();
    let mut parts = line.split_whitespace();
    let _ = parts.next();
    parts
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(0)
}

fn websocket_header_value(headers: &str, key: &str) -> Option<String> {
    for line in headers.lines() {
        let mut parts = line.splitn(2, ':');
        let name = parts.next()?.trim();
        if !name.eq_ignore_ascii_case(key) {
            continue;
        }
        return Some(parts.next()?.trim().to_string());
    }
    None
}

fn decode_first_ws_frame(bytes: &[u8]) -> Option<(u8, Vec<u8>)> {
    if bytes.len() < 2 {
        return None;
    }
    let opcode = bytes[0] & 0x0f;
    let masked = (bytes[1] & 0x80) != 0;
    if masked {
        return None;
    }
    let mut idx = 2usize;
    let mut len = (bytes[1] & 0x7f) as usize;
    if len == 126 {
        if bytes.len() < idx + 2 {
            return None;
        }
        len = u16::from_be_bytes([bytes[idx], bytes[idx + 1]]) as usize;
        idx += 2;
    } else if len == 127 {
        if bytes.len() < idx + 8 {
            return None;
        }
        len = u64::from_be_bytes([
            bytes[idx],
            bytes[idx + 1],
            bytes[idx + 2],
            bytes[idx + 3],
            bytes[idx + 4],
            bytes[idx + 5],
            bytes[idx + 6],
            bytes[idx + 7],
        ]) as usize;
        idx += 8;
    }
    if bytes.len() < idx + len {
        return None;
    }
    Some((opcode, bytes[idx..idx + len].to_vec()))
}

fn encode_client_ws_text(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x81);
    let len = payload.len();
    if len <= 125 {
        out.push(0x80 | (len as u8));
    } else if len <= u16::MAX as usize {
        out.push(0x80 | 126);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0x80 | 127);
        out.extend_from_slice(&(len as u64).to_be_bytes());
    }
    let mask = [0x12, 0x34, 0x56, 0x78];
    out.extend_from_slice(&mask);
    for (idx, byte) in payload.iter().enumerate() {
        out.push(byte ^ mask[idx % 4]);
    }
    out
}

fn response_status_value(value: &Value) -> Option<i64> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            enkairt::object::Obj::Record(map) => {
                let map = map.borrow();
                match map.get("status") {
                    Some(Value::Int(status)) => Some(*status),
                    _ => None,
                }
            }
            _ => None,
        },
        _ => None,
    }
}

#[test]
fn http_get_roundtrip() {
    let _guard = http_test_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let source = format!(
        "fn handler(req: Request) -> Response ::\n    return http.ok(\"pong\")\n::\n\
         http.serve(\"127.0.0.1\", {port}, handler)\n\
         let resp := http.get(\"http://127.0.0.1:{port}/\")\n\
         return resp\n"
    );
    let result = run_value(&source);
    let body = response_body(&result).expect("body");
    assert_eq!(body, b"pong");
}

#[test]
fn http_server_handles_concurrent_requests() {
    let _guard = http_test_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let source = format!(
        "fn handler(req: Request) -> Response ::\n    return http.ok(\"ok\")\n::\n\
         http.serve(\"127.0.0.1\", {port}, handler)\n\
         task.sleep(3000)\n"
    );
    let server = std::thread::spawn(move || {
        let _ = run_value(&source);
    });

    std::thread::sleep(std::time::Duration::from_millis(50));

    let mut handles = Vec::new();
    for _ in 0..50 {
        handles.push(std::thread::spawn(move || {
            let request = b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
            let buf = send_raw_request(port, request);
            assert!(buf.starts_with(b"HTTP/1.1 200"));
            assert!(buf.ends_with(b"ok"));
        }));
    }
    for handle in handles {
        handle.join().expect("client");
    }

    server.join().expect("server");
}

#[test]
fn http_routes_with_params() {
    let _guard = http_test_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let source = format!(
        "fn handler(req: Request) -> Response ::\n    return http.ok(req.params.id)\n::\n\
         let routes := [http.route(\"GET\", \"/items/:id\", handler)]\n\
         http.serve_with(\"127.0.0.1\", {port}, routes, none)\n\
         let resp := http.get(\"http://127.0.0.1:{port}/items/42\")\n\
         return resp\n"
    );
    let result = run_value(&source);
    let body = response_body(&result).expect("body");
    assert_eq!(body, b"42");
}

#[test]
fn http_stream_sends_chunks() {
    let _guard = http_test_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let source = format!(
        "fn handler(req: Request) -> Response ::\n    let headers := json.parse(\"{{\\\"content-type\\\":\\\"text/event-stream\\\"}}\")\n    let s := http.stream_open(200, headers)\n    http.stream_send(s, \"data: one\\n\\n\")\n    http.stream_send(s, \"data: two\\n\\n\")\n    http.stream_close(s)\n    return none\n::\n\
         http.serve(\"127.0.0.1\", {port}, handler)\n\
         task.sleep(100)\n"
    );
    let server = std::thread::spawn(move || {
        let _ = run_value(&source);
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    let request = b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    let buf = send_raw_request(port, request);
    let body = String::from_utf8_lossy(&buf);
    assert!(body.contains("data: one"));
    assert!(body.contains("data: two"));

    server.join().expect("server");
}

#[test]
fn http_request_config_headers() {
    let _guard = http_test_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let source = format!(
        "fn handler(req: Request) -> Response ::\n    let value := http.header(req, \"x-test\")?\n    return http.ok(value)\n::\n\
         http.serve(\"127.0.0.1\", {port}, handler)\n\
         let cfg := json.parse(\"{{\\\"method\\\":\\\"GET\\\",\\\"url\\\":\\\"http://127.0.0.1:{port}/\\\",\\\"headers\\\":{{\\\"x-test\\\":\\\"ok\\\"}},\\\"timeout_ms\\\":1000,\\\"retries\\\":1}}\")\n\
         let resp := http.request(cfg)\n\
         return resp\n"
    );
    let result = run_value(&source);
    let body = response_body(&result).expect("body");
    assert_eq!(body, b"ok");
}

#[test]
fn http_auth_middleware_rejects_missing_token() {
    let _guard = http_test_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let source = format!(
        "fn handler(req: Request) -> Response ::\n    return http.ok(\"ok\")\n::\n\
         let routes := [http.route(\"GET\", \"/\", handler)]\n\
         let auth := json.parse(\"{{\\\"tokens\\\":[{{\\\"token\\\":\\\"secret\\\",\\\"tenant\\\":\\\"acme\\\"}}],\\\"allow_anonymous\\\":false}}\")\n\
         let middlewares := [http.middleware(\"auth\", auth)]\n\
         http.serve_with(\"127.0.0.1\", {port}, routes, middlewares)\n\
         task.sleep(150)\n"
    );
    let server = std::thread::spawn(move || {
        let _ = run_value(&source);
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    let response = send_raw_request(
        port,
        b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    let text = String::from_utf8_lossy(&response);
    assert_eq!(response_status(&response), 401);
    assert!(text.contains("\"code\":\"unauthorized\""));
    assert!(text.contains("X-Enkai-Request-Id"));

    server.join().expect("server");
}

#[test]
fn http_rate_limit_middleware_enforces_capacity() {
    let _guard = http_test_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let source = format!(
        "fn handler(req: Request) -> Response ::\n    return http.ok(\"ok\")\n::\n\
         let routes := [http.route(\"GET\", \"/\", handler)]\n\
         let rate := json.parse(\"{{\\\"capacity\\\":1,\\\"refill_per_sec\\\":0.1,\\\"key\\\":\\\"ip\\\"}}\")\n\
         let middlewares := [http.middleware(\"rate_limit\", rate)]\n\
         http.serve_with(\"127.0.0.1\", {port}, routes, middlewares)\n\
         let first := http.get(\"http://127.0.0.1:{port}/\")\n\
         let second := http.get(\"http://127.0.0.1:{port}/\")\n\
         return [first, second]\n"
    );
    let result = run_value(&source);
    let items = match result {
        Value::Obj(obj) => match obj.as_obj() {
            enkairt::object::Obj::List(items) => items.borrow().clone(),
            _ => panic!("expected list"),
        },
        _ => panic!("expected list"),
    };
    assert_eq!(items.len(), 2);
    let first_status = response_status_value(&items[0]).expect("first status");
    let second_status = response_status_value(&items[1]).expect("second status");
    assert_eq!(first_status, 200);
    assert_eq!(second_status, 429);
    let second_body = response_body(&items[1]).expect("second body");
    let second_body = String::from_utf8_lossy(&second_body);
    assert!(second_body.contains("\"code\":\"rate_limited\""));
}

#[test]
fn http_requires_policy_capability() {
    let result = run_result_raw("http.get(\"http://127.0.0.1:1/\")\n");
    assert!(result.is_err());
    let message = result.err().unwrap().to_string();
    assert!(message.contains("[E_POLICY_DENIED]"));
}

#[test]
fn http_websocket_upgrade_and_send_text() {
    let _guard = http_test_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let source = format!(
        "fn handler(req: Request) -> Response ::\n    let ws := http.ws_open(req)\n    http.ws_send(ws, \"hello\")\n    http.ws_close(ws)\n    return none\n::\n\
         http.serve(\"127.0.0.1\", {port}, handler)\n\
         task.sleep(200)\n"
    );
    let server = std::thread::spawn(move || {
        let _ = run_value(&source);
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let request = b"GET /chat HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
    stream.write_all(request).expect("write");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(1)))
        .expect("timeout");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read");
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("header end");
    let header_text = String::from_utf8_lossy(&raw[..split + 4]);
    assert!(header_text.starts_with("HTTP/1.1 101"));
    let accept = websocket_header_value(&header_text, "Sec-WebSocket-Accept").expect("accept");
    assert_eq!(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    let (opcode, payload) = decode_first_ws_frame(&raw[split + 4..]).expect("ws frame");
    assert_eq!(opcode, 0x1);
    assert_eq!(payload, b"hello");

    server.join().expect("server");
}

#[test]
fn http_websocket_recv_and_echo() {
    let _guard = http_test_guard();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let source = format!(
        "fn handler(req: Request) -> Response ::\n    let ws := http.ws_open(req)\n    let msg := http.ws_recv(ws, 1000)\n    if msg != none ::\n        http.ws_send(ws, msg)\n    ::\n    http.ws_close(ws)\n    return none\n::\n\
         http.serve(\"127.0.0.1\", {port}, handler)\n\
         task.sleep(200)\n"
    );
    let server = std::thread::spawn(move || {
        let _ = run_value(&source);
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let request = b"GET /chat HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
    stream.write_all(request).expect("write");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .expect("timeout");
    let mut header = vec![0u8; 512];
    let header_n = stream.read(&mut header).expect("handshake read");
    let header = &header[..header_n];
    assert!(response_status(header) == 101);

    let frame = encode_client_ws_text(b"ping");
    stream.write_all(&frame).expect("ws write");
    let mut ws_buf = Vec::new();
    let mut chunk = [0u8; 256];
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let (opcode, payload) = loop {
        let n = stream.read(&mut chunk).expect("ws read");
        if n == 0 {
            break decode_first_ws_frame(&ws_buf).expect("ws response frame");
        }
        ws_buf.extend_from_slice(&chunk[..n]);
        if let Some(frame) = decode_first_ws_frame(&ws_buf) {
            break frame;
        }
        if std::time::Instant::now() >= deadline {
            panic!("timeout waiting for websocket frame: {:?}", ws_buf);
        }
    };
    assert_eq!(opcode, 0x1);
    assert_eq!(payload, b"ping");

    server.join().expect("server");
}
