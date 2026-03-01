use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Mutex, OnceLock};

use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
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
        let port = port;
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
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let request = b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    stream.write_all(request).expect("write");
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).expect("read");
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
