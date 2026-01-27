use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
use enkairt::{Value, VM};

fn run_value(source: &str) -> Value {
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program).expect("run")
}

fn response_body(value: &Value) -> Option<Vec<u8>> {
    match value {
        Value::Obj(obj) => match obj.as_obj() {
            enkairt::object::Obj::Record(map) => match map.get("body") {
                Some(Value::Obj(obj)) => match obj.as_obj() {
                    enkairt::object::Obj::Buffer(bytes) => Some(bytes.clone()),
                    enkairt::object::Obj::String(s) => Some(s.as_bytes().to_vec()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

#[test]
fn http_get_roundtrip() {
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
            let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
            let request = b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
            stream.write_all(request).expect("write");
            let mut buf = Vec::new();
            stream.read_to_end(&mut buf).expect("read");
            assert!(buf.starts_with(b"HTTP/1.1 200"));
            assert!(buf.ends_with(b"ok"));
        }));
    }
    for handle in handles {
        handle.join().expect("client");
    }

    server.join().expect("server");
}
