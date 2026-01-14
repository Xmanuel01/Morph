use std::env;
use std::fs;
use std::process;

use morphc::parser::parse_module;
use morphrt::{Interpreter, Value};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "run" => {
            if args.len() < 3 {
                eprintln!("morph run requires a file path");
                process::exit(1);
            }
            let path = &args[2];
            let source = match fs::read_to_string(path) {
                Ok(src) => src,
                Err(err) => {
                    eprintln!("Failed to read {}: {}", path, err);
                    process::exit(1);
                }
            };
            let module = match parse_module(&source) {
                Ok(module) => module,
                Err(err) => {
                    eprintln!("Parse error: {}", err);
                    process::exit(1);
                }
            };
            let mut interpreter = Interpreter::new();
            if let Err(err) = interpreter.eval_module(&module) {
                eprintln!("Runtime error: {}", err);
                process::exit(1);
            }
            match interpreter.call_main() {
                Ok(Some(Value::Int(code))) => {
                    if code != 0 {
                        process::exit(code as i32);
                    }
                }
                Ok(Some(_)) => {}
                Ok(None) => {}
                Err(err) => {
                    eprintln!("Runtime error: {}", err);
                    process::exit(1);
                }
            }
        }
        "fmt" => {
            eprintln!("morph fmt is not implemented yet");
            process::exit(2);
        }
        "test" => {
            eprintln!("morph test is not implemented yet. Use cargo test.");
            process::exit(2);
        }
        _ => {
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Morph CLI");
    eprintln!("Usage:");
    eprintln!("  morph run <file>");
    eprintln!("  morph fmt");
    eprintln!("  morph test");
}
