use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

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

#[test]
fn tcp_accept_read_write_roundtrip() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let client = std::thread::spawn(move || {
        let mut stream = loop {
            match TcpStream::connect(("127.0.0.1", port)) {
                Ok(stream) => break stream,
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        };
        stream.write_all(b"hello").expect("write");
        let mut buf = [0u8; 5];
        stream.read_exact(&mut buf).expect("read");
        assert_eq!(&buf, b"hello");
    });

    let source = format!(
        "let listener := net.bind(\"127.0.0.1\", {})\n\
         fn serve() -> Int ::\n    let conn := listener.accept()\n    let buf := conn.read(5)\n    conn.write(buf)\n    return 1\n::\n\
         let h := task.spawn(serve)\n\
         task.join(h)\n",
        port
    );
    let result = run_value(&source);
    assert_eq!(result, Value::Int(1));
    client.join().expect("client");
}
