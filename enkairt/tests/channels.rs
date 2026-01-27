use enkaic::compiler::compile_module;
use enkaic::parser::parse_module;
use enkairt::{Value, VM};

fn run_value(source: &str) -> Value {
    let module = parse_module(source).expect("parse");
    let program = compile_module(&module).expect("compile");
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program).expect("run")
}

#[test]
fn channel_send_recv() {
    let value = run_value("let c := chan.make()\nchan.send(c, 42)\nchan.recv(c)\n");
    assert_eq!(value, Value::Int(42));
}

#[test]
fn channel_recv_blocks_until_send() {
    let value = run_value(
        "let c := chan.make()\n\
        fn sender() -> Int ::\n    chan.send(c, 7)\n    return 0\n::\n\
        let h := task.spawn(sender)\n\
        let v := chan.recv(c)\n\
        task.join(h)\n\
        v\n",
    );
    assert_eq!(value, Value::Int(7));
}
