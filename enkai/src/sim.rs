use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::json;

use enkai_compiler::compiler::compile_module;
use enkai_compiler::parser::parse_module_named;
use enkai_compiler::TypeChecker;
use enkai_runtime::object::Obj;
use enkai_runtime::{Value, VM};

#[derive(Debug, Clone, PartialEq, Eq)]
struct SimRunOptions {
    target: PathBuf,
    trace_vm: bool,
    disasm: bool,
    trace_task: bool,
    trace_net: bool,
    emit_json_stdout: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SimProfileOptions {
    run: SimRunOptions,
    profile_output: PathBuf,
    case_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SimReplayOptions {
    snapshot: PathBuf,
    steps: usize,
    emit_json_stdout: bool,
    output: Option<PathBuf>,
}

pub fn sim_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_sim_usage();
        return 1;
    }
    match args[0].as_str() {
        "run" => sim_run_command(&args[1..]),
        "profile" => sim_profile_command(&args[1..]),
        "replay" => sim_replay_command(&args[1..]),
        _ => {
            print_sim_usage();
            1
        }
    }
}

pub fn print_sim_usage() {
    eprintln!("  enkai sim run [--trace-vm] [--disasm] [--trace-task] [--trace-net] [--json] [--output <file>] <file|dir>");
    eprintln!("  enkai sim profile [--trace-vm] [--disasm] [--trace-task] [--trace-net] [--case <id>] --output <file> <file|dir>");
    eprintln!("  enkai sim replay --snapshot <file> --steps <n> [--json] [--output <file>]");
}

fn sim_run_command(args: &[String]) -> i32 {
    let options = match parse_sim_run_args(args) {
        Ok(options) => options,
        Err(err) => {
            eprintln!("enkai sim run: {}", err);
            return 1;
        }
    };
    match execute_sim_run(&options, None, None) {
        Ok(exit_code) => exit_code,
        Err(err) => {
            eprintln!("{}", err);
            1
        }
    }
}

fn sim_profile_command(args: &[String]) -> i32 {
    let options = match parse_sim_profile_args(args) {
        Ok(options) => options,
        Err(err) => {
            eprintln!("enkai sim profile: {}", err);
            return 1;
        }
    };
    match execute_sim_run(
        &options.run,
        Some(options.profile_output.as_path()),
        Some(options.case_id.as_str()),
    ) {
        Ok(exit_code) => exit_code,
        Err(err) => {
            eprintln!("{}", err);
            1
        }
    }
}

fn sim_replay_command(args: &[String]) -> i32 {
    let options = match parse_sim_replay_args(args) {
        Ok(options) => options,
        Err(err) => {
            eprintln!("enkai sim replay: {}", err);
            return 1;
        }
    };
    let snapshot_text = match fs::read_to_string(&options.snapshot) {
        Ok(text) => text,
        Err(err) => {
            eprintln!(
                "enkai sim replay: failed to read {}: {}",
                options.snapshot.display(),
                err
            );
            return 1;
        }
    };
    let snapshot_minified = match serde_json::from_str::<serde_json::Value>(&snapshot_text) {
        Ok(value) => match serde_json::to_string(&value) {
            Ok(compact) => compact,
            Err(err) => {
                eprintln!(
                    "enkai sim replay: failed to compact snapshot {}: {}",
                    options.snapshot.display(),
                    err
                );
                return 1;
            }
        },
        Err(err) => {
            eprintln!(
                "enkai sim replay: snapshot {} is not valid JSON: {}",
                options.snapshot.display(),
                err
            );
            return 1;
        }
    };
    let escaped_snapshot = snapshot_minified
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\r', "\\r")
        .replace('\n', "\\n");
    let source = format!(
        "import json\nimport std::sim\nfn main() ::\n    let snapshot := json.parse(\"{escaped_snapshot}\")\n    let world := sim.restore(snapshot)\n    sim.run(world, {steps})\n    return sim.snapshot(world)\n::\nmain()\n",
        escaped_snapshot = escaped_snapshot,
        steps = options.steps
    );
    let started = Instant::now();
    let (value, exit_code) = match run_inline_module(&source, "<sim-replay>") {
        Ok(value) => {
            let code = match value {
                Value::Int(code) => code as i32,
                _ => 0,
            };
            (value, code)
        }
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let payload = json!({
        "command": "sim.replay",
        "snapshot": options.snapshot.display().to_string(),
        "steps": options.steps,
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
        "exit_code": exit_code,
        "result": value_to_json(&value),
    });
    if let Err(err) = write_json_outputs(
        &payload,
        options.emit_json_stdout,
        options.output.as_deref(),
    ) {
        eprintln!("enkai sim replay: {}", err);
        return 1;
    }
    exit_code
}

fn execute_sim_run(
    options: &SimRunOptions,
    profile_output: Option<&Path>,
    case_id: Option<&str>,
) -> Result<i32, String> {
    let started = Instant::now();
    let program = load_program_from_target(&options.target)?;
    let _guard = crate::env_guard();
    let previous_profile_out = env::var("ENKAI_BENCH_PROFILE_OUT").ok();
    let previous_profile_case = env::var("ENKAI_BENCH_PROFILE_CASE").ok();
    if let Some(path) = profile_output {
        env::set_var("ENKAI_BENCH_PROFILE_OUT", path);
    } else {
        env::remove_var("ENKAI_BENCH_PROFILE_OUT");
    }
    if let Some(case_id) = case_id {
        env::set_var("ENKAI_BENCH_PROFILE_CASE", case_id);
    } else {
        env::remove_var("ENKAI_BENCH_PROFILE_CASE");
    }
    let mut vm = VM::new(
        options.trace_vm,
        options.disasm,
        options.trace_task,
        options.trace_net,
    );
    let run_result = vm.run(&program);
    match previous_profile_out {
        Some(value) => env::set_var("ENKAI_BENCH_PROFILE_OUT", value),
        None => env::remove_var("ENKAI_BENCH_PROFILE_OUT"),
    }
    match previous_profile_case {
        Some(value) => env::set_var("ENKAI_BENCH_PROFILE_CASE", value),
        None => env::remove_var("ENKAI_BENCH_PROFILE_CASE"),
    }
    match run_result {
        Ok(value) => {
            let exit_code = match value {
                Value::Int(code) => code as i32,
                _ => 0,
            };
            let payload = json!({
                "command": "sim.run",
                "target": options.target.display().to_string(),
                "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
                "exit_code": exit_code,
                "result": value_to_json(&value),
            });
            write_json_outputs(
                &payload,
                options.emit_json_stdout,
                options.output.as_deref(),
            )?;
            Ok(exit_code)
        }
        Err(err) => Err(format!("Runtime error: {}", err)),
    }
}

fn load_program_from_target(target: &Path) -> Result<enkai_compiler::bytecode::Program, String> {
    if target.is_file() {
        let parent = target
            .parent()
            .ok_or_else(|| "Invalid file path".to_string())?;
        if super::find_project_root(parent).is_none() {
            let source = fs::read_to_string(target)
                .map_err(|err| format!("Failed to read {}: {}", target.display(), err))?;
            return compile_source_module(&source, &target.display().to_string());
        }
    }
    let (root, entry) = super::resolve_entry(target)?;
    match super::load_cached_program(&root, &entry) {
        Ok(Some(program)) => Ok(program),
        Ok(None) | Err(_) => {
            let package =
                enkai_compiler::modules::load_package(&entry).map_err(|err| err.to_string())?;
            enkai_compiler::TypeChecker::check_package(&package)
                .map_err(|err| err.message.clone())?;
            enkai_compiler::compiler::compile_package(&package).map_err(|err| err.message)
        }
    }
}

fn run_inline_module(source: &str, source_name: &str) -> Result<Value, String> {
    let program = compile_source_module(source, source_name)?;
    let mut vm = VM::new(false, false, false, false);
    vm.run(&program)
        .map_err(|err| format!("Runtime error: {}", err))
}

fn compile_source_module(
    source: &str,
    source_name: &str,
) -> Result<enkai_compiler::bytecode::Program, String> {
    let module = parse_module_named(source, Some(source_name)).map_err(|err| err.to_string())?;
    let mut checker = TypeChecker::new();
    checker
        .check_module(&module)
        .map_err(|err| err.message.clone())?;
    compile_module(&module).map_err(|err| err.message)
}

fn write_json_outputs(
    payload: &serde_json::Value,
    emit_stdout: bool,
    output: Option<&Path>,
) -> Result<(), String> {
    if emit_stdout {
        println!(
            "{}",
            serde_json::to_string_pretty(payload).map_err(|err| err.to_string())?
        );
    }
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
        }
        fs::write(
            path,
            serde_json::to_vec_pretty(payload).map_err(|err| err.to_string())?,
        )
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(value) => serde_json::Value::Bool(*value),
        Value::Int(value) => json!(value),
        Value::Float(value) => json!(value),
        Value::Obj(obj) => match obj.as_obj() {
            Obj::String(text) => json!(text),
            Obj::Buffer(bytes) => serde_json::Value::Array(
                bytes.iter().copied().map(serde_json::Value::from).collect(),
            ),
            Obj::List(values) => serde_json::Value::Array(
                values
                    .borrow()
                    .iter()
                    .map(value_to_json)
                    .collect::<Vec<_>>(),
            ),
            Obj::Record(map) => {
                let map = map.borrow();
                let mut out = serde_json::Map::with_capacity(map.len());
                for (key, value) in map.iter() {
                    out.insert(key.clone(), value_to_json(value));
                }
                serde_json::Value::Object(out)
            }
            Obj::Json(json) => json.clone(),
            other => json!({
                "__enkai_type": other.type_name(),
                "__display": format!("<{}>", other.type_name())
            }),
        },
    }
}

fn parse_sim_run_args(args: &[String]) -> Result<SimRunOptions, String> {
    let mut target: Option<PathBuf> = None;
    let mut trace_vm = false;
    let mut disasm = false;
    let mut trace_task = false;
    let mut trace_net = false;
    let mut emit_json_stdout = false;
    let mut output: Option<PathBuf> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--trace-vm" => trace_vm = true,
            "--disasm" => disasm = true,
            "--trace-task" => trace_task = true,
            "--trace-net" => trace_net = true,
            "--json" => emit_json_stdout = true,
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                output = Some(PathBuf::from(&args[idx]));
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option '{}'", flag)),
            path => {
                if target.is_some() {
                    return Err("sim run accepts only one target".to_string());
                }
                target = Some(PathBuf::from(path));
            }
        }
        idx += 1;
    }
    let target = target.ok_or_else(|| "sim run requires <file|dir>".to_string())?;
    Ok(SimRunOptions {
        target,
        trace_vm,
        disasm,
        trace_task,
        trace_net,
        emit_json_stdout,
        output,
    })
}

fn parse_sim_profile_args(args: &[String]) -> Result<SimProfileOptions, String> {
    let mut output: Option<PathBuf> = None;
    let mut case_id = "sim_cli".to_string();
    let mut passthrough = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                output = Some(PathBuf::from(&args[idx]));
            }
            "--case" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--case requires a value".to_string());
                }
                case_id = args[idx].clone();
            }
            value => passthrough.push(value.to_string()),
        }
        idx += 1;
    }
    let profile_output =
        output.ok_or_else(|| "sim profile requires --output <file>".to_string())?;
    let run = parse_sim_run_args(&passthrough)?;
    Ok(SimProfileOptions {
        run,
        profile_output,
        case_id,
    })
}

fn parse_sim_replay_args(args: &[String]) -> Result<SimReplayOptions, String> {
    let mut snapshot: Option<PathBuf> = None;
    let mut steps: Option<usize> = None;
    let mut emit_json_stdout = false;
    let mut output: Option<PathBuf> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--snapshot" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--snapshot requires a value".to_string());
                }
                snapshot = Some(PathBuf::from(&args[idx]));
            }
            "--steps" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--steps requires a value".to_string());
                }
                steps = Some(
                    args[idx]
                        .parse::<usize>()
                        .map_err(|_| "--steps expects a non-negative integer".to_string())?,
                );
            }
            "--json" => emit_json_stdout = true,
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                output = Some(PathBuf::from(&args[idx]));
            }
            flag => return Err(format!("unknown option '{}'", flag)),
        }
        idx += 1;
    }
    Ok(SimReplayOptions {
        snapshot: snapshot.ok_or_else(|| "sim replay requires --snapshot <file>".to_string())?,
        steps: steps.ok_or_else(|| "sim replay requires --steps <n>".to_string())?,
        emit_json_stdout,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_sim_run_args_supports_json_and_output() {
        let options = parse_sim_run_args(&[
            "--json".to_string(),
            "--output".to_string(),
            "artifacts/sim/run.json".to_string(),
            "examples/nn_sanity.enk".to_string(),
        ])
        .expect("parse");
        assert!(options.emit_json_stdout);
        assert_eq!(
            options.output,
            Some(PathBuf::from("artifacts/sim/run.json"))
        );
        assert_eq!(options.target, PathBuf::from("examples/nn_sanity.enk"));
    }

    #[test]
    fn parse_sim_profile_args_requires_output() {
        let err = parse_sim_profile_args(&["examples/nn_sanity.enk".to_string()]).unwrap_err();
        assert!(err.contains("--output"));
    }

    #[test]
    fn sim_run_writes_json_report() {
        let dir = tempdir().expect("tempdir");
        let script = dir.path().join("sim.enk");
        fs::write(
            &script,
            "import std::sim\nfn main() ::\n    let w := sim.make_seeded(8, 7)\n    sim.schedule(w, 1.0, 9)\n    sim.run(w, 1)\n    return sim.snapshot(w)\n::\nmain()\n",
        )
        .expect("write");
        let out = dir.path().join("run.json");
        let options = SimRunOptions {
            target: script,
            trace_vm: false,
            disasm: false,
            trace_task: false,
            trace_net: false,
            emit_json_stdout: false,
            output: Some(out.clone()),
        };
        let exit = execute_sim_run(&options, None, None).expect("run");
        assert_eq!(exit, 0);
        let text = fs::read_to_string(out).expect("report");
        assert!(text.contains("\"command\": \"sim.run\""));
        assert!(text.contains("\"result\""));
    }

    #[test]
    fn sim_profile_writes_vm_profile_artifact() {
        let dir = tempdir().expect("tempdir");
        let script = dir.path().join("sim.enk");
        fs::write(
            &script,
            "import std::sim\nfn main() ::\n    let w := sim.make_seeded(8, 7)\n    sim.schedule(w, 1.0, 9)\n    sim.run(w, 1)\n    return 0\n::\nmain()\n",
        )
        .expect("write");
        let profile = dir.path().join("profile.json");
        let options = SimRunOptions {
            target: script,
            trace_vm: false,
            disasm: false,
            trace_task: false,
            trace_net: false,
            emit_json_stdout: false,
            output: None,
        };
        let exit = execute_sim_run(&options, Some(&profile), Some("sim_cli_test")).expect("run");
        assert_eq!(exit, 0);
        let text = fs::read_to_string(profile).expect("profile");
        assert!(text.contains("\"case\": \"sim_cli_test\""));
    }

    #[test]
    fn sim_replay_parses_args() {
        let options = parse_sim_replay_args(&[
            "--snapshot".to_string(),
            "artifacts/sim/snapshot.json".to_string(),
            "--steps".to_string(),
            "4".to_string(),
            "--json".to_string(),
        ])
        .expect("parse");
        assert_eq!(
            options.snapshot,
            PathBuf::from("artifacts/sim/snapshot.json")
        );
        assert_eq!(options.steps, 4);
        assert!(options.emit_json_stdout);
    }
}
