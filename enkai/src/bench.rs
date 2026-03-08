use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
struct BenchRunOptions {
    suite: String,
    baseline: String,
    output: PathBuf,
    machine_profile: Option<PathBuf>,
    iterations: usize,
    warmup: usize,
    target_speedup_pct: f64,
    target_memory_pct: f64,
    enforce_target: bool,
    python_command: Option<String>,
    enkai_bin: Option<PathBuf>,
}

impl Default for BenchRunOptions {
    fn default() -> Self {
        Self {
            suite: "core".to_string(),
            baseline: "python".to_string(),
            output: PathBuf::from("bench").join("results").join("latest.json"),
            machine_profile: None,
            iterations: 5,
            warmup: 1,
            target_speedup_pct: 5.0,
            target_memory_pct: 5.0,
            enforce_target: false,
            python_command: None,
            enkai_bin: None,
        }
    }
}

pub fn bench_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_bench_usage();
        return 1;
    }
    match args[0].as_str() {
        "run" => run_suite_command(&args[1..]),
        _ => {
            eprintln!("enkai bench: unknown subcommand '{}'", args[0]);
            print_bench_usage();
            1
        }
    }
}

pub fn print_bench_usage() {
    eprintln!("  enkai bench run [--suite <name>] [--baseline <python|none>] [--output <file>] [--machine-profile <file>] [--iterations <n>] [--warmup <n>] [--target-speedup <pct>] [--target-memory <pct>] [--enforce-target] [--python <command>] [--enkai-bin <path>]");
}

fn run_suite_command(args: &[String]) -> i32 {
    let options = match parse_bench_run_options(args) {
        Ok(options) => options,
        Err(err) => {
            eprintln!("enkai bench run: {}", err);
            print_bench_usage();
            return 1;
        }
    };
    let script = bench_script_path();
    if !script.is_file() {
        eprintln!(
            "enkai bench run: benchmark harness not found at {}",
            script.display()
        );
        return 1;
    }
    if let Some(parent) = options.output.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(err) = fs::create_dir_all(parent) {
                eprintln!(
                    "enkai bench run: failed to create output directory {}: {}",
                    parent.display(),
                    err
                );
                return 1;
            }
        }
    }
    let python_cmd = match resolve_python_command(options.python_command.as_deref()) {
        Ok(cmd) => cmd,
        Err(err) => {
            eprintln!("enkai bench run: {}", err);
            return 1;
        }
    };
    let mut command = Command::new(&python_cmd[0]);
    if python_cmd.len() > 1 {
        command.args(&python_cmd[1..]);
    }
    command.arg(script.as_os_str());
    command.arg("--suite").arg(&options.suite);
    command.arg("--baseline").arg(&options.baseline);
    command
        .arg("--output")
        .arg(options.output.as_os_str())
        .arg("--iterations")
        .arg(options.iterations.to_string())
        .arg("--warmup")
        .arg(options.warmup.to_string())
        .arg("--target-speedup")
        .arg(options.target_speedup_pct.to_string())
        .arg("--target-memory")
        .arg(options.target_memory_pct.to_string());
    if options.enforce_target {
        command.arg("--enforce-target");
    }
    if let Some(machine_profile) = options.machine_profile {
        command
            .arg("--machine-profile")
            .arg(machine_profile.as_os_str());
    }
    if let Some(enkai_bin) = options.enkai_bin {
        command.arg("--enkai-bin").arg(enkai_bin.as_os_str());
    }
    let status = match command.status() {
        Ok(status) => status,
        Err(err) => {
            eprintln!(
                "enkai bench run: failed to launch benchmark harness: {}",
                err
            );
            return 1;
        }
    };
    status.code().unwrap_or(1)
}

fn parse_bench_run_options(args: &[String]) -> Result<BenchRunOptions, String> {
    let mut options = BenchRunOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let arg = &args[idx];
        match arg.as_str() {
            "--suite" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--suite requires a value".to_string());
                }
                options.suite = args[idx].clone();
            }
            "--baseline" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--baseline requires a value".to_string());
                }
                let value = args[idx].trim().to_ascii_lowercase();
                if value != "python" && value != "none" {
                    return Err("--baseline expects 'python' or 'none'".to_string());
                }
                options.baseline = value;
            }
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                options.output = PathBuf::from(&args[idx]);
            }
            "--machine-profile" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--machine-profile requires a value".to_string());
                }
                options.machine_profile = Some(PathBuf::from(&args[idx]));
            }
            "--iterations" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--iterations requires a value".to_string());
                }
                options.iterations = parse_positive_usize("--iterations", &args[idx])?;
            }
            "--warmup" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--warmup requires a value".to_string());
                }
                options.warmup = parse_nonnegative_usize("--warmup", &args[idx])?;
            }
            "--target-speedup" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--target-speedup requires a value".to_string());
                }
                options.target_speedup_pct = parse_nonnegative_f64("--target-speedup", &args[idx])?;
            }
            "--target-memory" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--target-memory requires a value".to_string());
                }
                options.target_memory_pct = parse_nonnegative_f64("--target-memory", &args[idx])?;
            }
            "--enforce-target" => {
                options.enforce_target = true;
            }
            "--python" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--python requires a value".to_string());
                }
                options.python_command = Some(args[idx].clone());
            }
            "--enkai-bin" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--enkai-bin requires a value".to_string());
                }
                options.enkai_bin = Some(PathBuf::from(&args[idx]));
            }
            _ => {
                return Err(format!("unknown option '{}'", arg));
            }
        }
        idx += 1;
    }
    Ok(options)
}

fn parse_positive_usize(name: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{} expects a positive integer", name))?;
    if parsed == 0 {
        return Err(format!("{} expects a value > 0", name));
    }
    Ok(parsed)
}

fn parse_nonnegative_usize(name: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{} expects a non-negative integer", name))
}

fn parse_nonnegative_f64(name: &str, value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("{} expects a numeric value", name))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(format!("{} expects a non-negative finite value", name));
    }
    Ok(parsed)
}

fn bench_script_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("run_suite.py")
}

fn resolve_python_command(explicit: Option<&str>) -> Result<Vec<String>, String> {
    if let Some(explicit) = explicit {
        let trimmed = explicit.trim();
        if trimmed.is_empty() {
            return Err("--python command cannot be empty".to_string());
        }
        return Ok(vec![trimmed.to_string()]);
    }
    for candidate in default_python_candidates() {
        if command_available(&candidate) {
            return Ok(candidate);
        }
    }
    Err("Python interpreter not found (tried: python3, python, py -3)".to_string())
}

fn default_python_candidates() -> Vec<Vec<String>> {
    if cfg!(windows) {
        vec![
            vec!["py".to_string(), "-3".to_string()],
            vec!["python".to_string()],
            vec!["python3".to_string()],
        ]
    } else {
        vec![
            vec!["python3".to_string()],
            vec!["python".to_string()],
            vec!["py".to_string(), "-3".to_string()],
        ]
    }
}

fn command_available(command: &[String]) -> bool {
    if command.is_empty() {
        return false;
    }
    let mut probe = Command::new(&command[0]);
    if command.len() > 1 {
        probe.args(&command[1..]);
    }
    match probe
        .arg("-c")
        .arg("import sys; sys.stdout.write(sys.version)")
        .output()
    {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults() {
        let options = parse_bench_run_options(&[]).expect("parse");
        assert_eq!(options.suite, "core");
        assert_eq!(options.baseline, "python");
        assert_eq!(options.iterations, 5);
        assert_eq!(options.warmup, 1);
        assert!(!options.enforce_target);
    }

    #[test]
    fn parse_custom_values() {
        let options = parse_bench_run_options(&[
            "--suite".to_string(),
            "llm_core".to_string(),
            "--baseline".to_string(),
            "none".to_string(),
            "--output".to_string(),
            "bench/results/custom.json".to_string(),
            "--iterations".to_string(),
            "3".to_string(),
            "--warmup".to_string(),
            "0".to_string(),
            "--target-speedup".to_string(),
            "7.5".to_string(),
            "--target-memory".to_string(),
            "8.5".to_string(),
            "--enforce-target".to_string(),
        ])
        .expect("parse");
        assert_eq!(options.suite, "llm_core");
        assert_eq!(options.baseline, "none");
        assert_eq!(options.output, PathBuf::from("bench/results/custom.json"));
        assert_eq!(options.iterations, 3);
        assert_eq!(options.warmup, 0);
        assert_eq!(options.target_speedup_pct, 7.5);
        assert_eq!(options.target_memory_pct, 8.5);
        assert!(options.enforce_target);
    }

    #[test]
    fn parse_rejects_invalid_baseline() {
        let err = parse_bench_run_options(&["--baseline".to_string(), "unknown".to_string()])
            .expect_err("must reject");
        assert!(err.contains("baseline"));
    }
}
