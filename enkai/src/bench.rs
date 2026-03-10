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
    enforce_all_cases: bool,
    enforce_class_targets: bool,
    class_targets: Option<PathBuf>,
    fairness_check_only: bool,
    equivalence_contract: Option<PathBuf>,
    profile_case: Option<String>,
    profile_output: Option<PathBuf>,
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
            enforce_all_cases: false,
            enforce_class_targets: false,
            class_targets: None,
            fairness_check_only: false,
            equivalence_contract: None,
            profile_case: None,
            profile_output: None,
            python_command: None,
            enkai_bin: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BenchProfileOptions {
    case_id: String,
    output: PathBuf,
    suite: String,
    machine_profile: Option<PathBuf>,
    iterations: usize,
    warmup: usize,
    python_command: Option<String>,
    enkai_bin: Option<PathBuf>,
}

pub fn bench_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_bench_usage();
        return 1;
    }
    match args[0].as_str() {
        "run" => run_suite_command(&args[1..]),
        "profile" => profile_case_command(&args[1..]),
        _ => {
            eprintln!("enkai bench: unknown subcommand '{}'", args[0]);
            print_bench_usage();
            1
        }
    }
}

pub fn print_bench_usage() {
    eprintln!(
        "  enkai bench run [--suite <name>] [--baseline <python|none>] [--output <file>] [--machine-profile <file>] [--iterations <n>] [--warmup <n>] [--target-speedup <pct>] [--target-memory <pct>] [--enforce-target] [--enforce-all-cases] [--enforce-class-targets --class-targets <file>] [--fairness-check-only] [--equivalence-contract <file>] [--profile-case <id> --profile-output <file>] [--python <command>] [--enkai-bin <path>]"
    );
    eprintln!(
        "  enkai bench profile --case <id> --output <file> [--suite <name>] [--machine-profile <file>] [--iterations <n>] [--warmup <n>] [--python <command>] [--enkai-bin <path>]"
    );
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
    invoke_bench_harness("enkai bench run", &options)
}

fn profile_case_command(args: &[String]) -> i32 {
    let options = match parse_bench_profile_options(args) {
        Ok(options) => options,
        Err(err) => {
            eprintln!("enkai bench profile: {}", err);
            print_bench_usage();
            return 1;
        }
    };

    let mut run_options = BenchRunOptions {
        suite: options.suite,
        baseline: "none".to_string(),
        output: derive_profile_bench_output(&options.output),
        machine_profile: options.machine_profile,
        iterations: options.iterations,
        warmup: options.warmup,
        target_speedup_pct: 0.0,
        target_memory_pct: 0.0,
        enforce_target: false,
        enforce_all_cases: false,
        enforce_class_targets: false,
        class_targets: None,
        fairness_check_only: false,
        equivalence_contract: None,
        profile_case: Some(options.case_id),
        profile_output: Some(options.output),
        python_command: options.python_command,
        enkai_bin: options.enkai_bin,
    };

    // Profiling a single case should be deterministic and cheap by default.
    if run_options.iterations == 0 {
        run_options.iterations = 1;
    }

    invoke_bench_harness("enkai bench profile", &run_options)
}

fn derive_profile_bench_output(profile_output: &Path) -> PathBuf {
    let parent = profile_output
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("bench").join("results").join("profiles"));
    let stem = profile_output
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("profile");
    parent.join(format!("{}.bench.json", stem))
}

fn invoke_bench_harness(context: &str, options: &BenchRunOptions) -> i32 {
    let script = bench_script_path();
    if !script.is_file() {
        eprintln!(
            "{}: benchmark harness not found at {}",
            context,
            script.display()
        );
        return 1;
    }
    if let Err(err) = ensure_output_parent(&options.output) {
        eprintln!("{}: {}", context, err);
        return 1;
    }
    if let Some(path) = options.profile_output.as_deref() {
        if let Err(err) = ensure_output_parent(path) {
            eprintln!("{}: {}", context, err);
            return 1;
        }
    }
    let python_cmd = match resolve_python_command(options.python_command.as_deref()) {
        Ok(cmd) => cmd,
        Err(err) => {
            eprintln!("{}: {}", context, err);
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
    if options.enforce_all_cases {
        command.arg("--enforce-all-cases");
    }
    if options.enforce_class_targets {
        command.arg("--enforce-class-targets");
    }
    if let Some(class_targets) = options.class_targets.as_ref() {
        command
            .arg("--class-targets")
            .arg(class_targets.as_os_str());
    }
    if options.fairness_check_only {
        command.arg("--fairness-check-only");
    }
    if let Some(contract) = options.equivalence_contract.as_ref() {
        command
            .arg("--equivalence-contract")
            .arg(contract.as_os_str());
    }
    if let Some(case_id) = options.profile_case.as_ref() {
        command.arg("--profile-case").arg(case_id);
    }
    if let Some(profile_output) = options.profile_output.as_ref() {
        command
            .arg("--profile-output")
            .arg(profile_output.as_os_str());
    }
    if let Some(machine_profile) = options.machine_profile.as_ref() {
        command
            .arg("--machine-profile")
            .arg(machine_profile.as_os_str());
    }
    if let Some(enkai_bin) = options.enkai_bin.as_ref() {
        command.arg("--enkai-bin").arg(enkai_bin.as_os_str());
    }
    let status = match command.status() {
        Ok(status) => status,
        Err(err) => {
            eprintln!("{}: failed to launch benchmark harness: {}", context, err);
            return 1;
        }
    };
    status.code().unwrap_or(1)
}

fn ensure_output_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create output directory {}: {}",
                    parent.display(),
                    err
                )
            })?;
        }
    }
    Ok(())
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
            "--enforce-all-cases" => {
                options.enforce_all_cases = true;
            }
            "--enforce-class-targets" => {
                options.enforce_class_targets = true;
            }
            "--class-targets" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--class-targets requires a value".to_string());
                }
                options.class_targets = Some(PathBuf::from(&args[idx]));
            }
            "--fairness-check-only" => {
                options.fairness_check_only = true;
            }
            "--equivalence-contract" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--equivalence-contract requires a value".to_string());
                }
                options.equivalence_contract = Some(PathBuf::from(&args[idx]));
            }
            "--profile-case" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--profile-case requires a value".to_string());
                }
                options.profile_case = Some(args[idx].clone());
            }
            "--profile-output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--profile-output requires a value".to_string());
                }
                options.profile_output = Some(PathBuf::from(&args[idx]));
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

    if options.enforce_class_targets && options.class_targets.is_none() {
        return Err("--enforce-class-targets requires --class-targets <file>".to_string());
    }
    if options.profile_case.is_some() && options.profile_output.is_none() {
        return Err("--profile-case requires --profile-output <file>".to_string());
    }

    Ok(options)
}

fn parse_bench_profile_options(args: &[String]) -> Result<BenchProfileOptions, String> {
    let mut case_id: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut suite = "official_v2_3_0_matrix".to_string();
    let mut machine_profile: Option<PathBuf> = None;
    let mut iterations = 1usize;
    let mut warmup = 0usize;
    let mut python_command: Option<String> = None;
    let mut enkai_bin: Option<PathBuf> = None;

    let mut idx = 0usize;
    while idx < args.len() {
        let arg = &args[idx];
        match arg.as_str() {
            "--case" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--case requires a value".to_string());
                }
                case_id = Some(args[idx].clone());
            }
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                output = Some(PathBuf::from(&args[idx]));
            }
            "--suite" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--suite requires a value".to_string());
                }
                suite = args[idx].clone();
            }
            "--machine-profile" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--machine-profile requires a value".to_string());
                }
                machine_profile = Some(PathBuf::from(&args[idx]));
            }
            "--iterations" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--iterations requires a value".to_string());
                }
                iterations = parse_positive_usize("--iterations", &args[idx])?;
            }
            "--warmup" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--warmup requires a value".to_string());
                }
                warmup = parse_nonnegative_usize("--warmup", &args[idx])?;
            }
            "--python" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--python requires a value".to_string());
                }
                python_command = Some(args[idx].clone());
            }
            "--enkai-bin" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--enkai-bin requires a value".to_string());
                }
                enkai_bin = Some(PathBuf::from(&args[idx]));
            }
            _ => {
                return Err(format!("unknown option '{}'", arg));
            }
        }
        idx += 1;
    }

    let case_id = case_id.ok_or_else(|| "--case is required".to_string())?;
    let output = output.unwrap_or_else(|| {
        PathBuf::from("bench")
            .join("results")
            .join("profiles")
            .join(format!("{}.json", case_id))
    });

    Ok(BenchProfileOptions {
        case_id,
        output,
        suite,
        machine_profile,
        iterations,
        warmup,
        python_command,
        enkai_bin,
    })
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

#[cfg(all(test, not(windows)))]
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
        assert!(!options.enforce_class_targets);
        assert!(options.class_targets.is_none());
    }

    #[test]
    fn parse_custom_values() {
        let options = parse_bench_run_options(&[
            "--suite".to_string(),
            "official_v2_3_0_matrix".to_string(),
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
            "--enforce-class-targets".to_string(),
            "--class-targets".to_string(),
            "bench/suites/official_v2_3_0_targets.json".to_string(),
            "--fairness-check-only".to_string(),
        ])
        .expect("parse");
        assert_eq!(options.suite, "official_v2_3_0_matrix");
        assert_eq!(options.baseline, "none");
        assert_eq!(options.output, PathBuf::from("bench/results/custom.json"));
        assert_eq!(options.iterations, 3);
        assert_eq!(options.warmup, 0);
        assert_eq!(options.target_speedup_pct, 7.5);
        assert_eq!(options.target_memory_pct, 8.5);
        assert!(options.enforce_target);
        assert!(options.enforce_class_targets);
        assert!(options.fairness_check_only);
        assert!(!options.enforce_all_cases);
    }

    #[test]
    fn parse_rejects_invalid_baseline() {
        let err = parse_bench_run_options(&["--baseline".to_string(), "unknown".to_string()])
            .expect_err("must reject");
        assert!(err.contains("baseline"));
    }

    #[test]
    fn parse_requires_class_targets_when_enforced() {
        let err = parse_bench_run_options(&["--enforce-class-targets".to_string()])
            .expect_err("must reject");
        assert!(err.contains("class-targets"));
    }

    #[test]
    fn parse_profile_options() {
        let options = parse_bench_profile_options(&[
            "--case".to_string(),
            "numeric_kernel".to_string(),
            "--output".to_string(),
            "bench/results/profiles/numeric_kernel.json".to_string(),
        ])
        .expect("parse");
        assert_eq!(options.case_id, "numeric_kernel");
        assert_eq!(
            options.output,
            PathBuf::from("bench/results/profiles/numeric_kernel.json")
        );
        assert_eq!(options.suite, "official_v2_3_0_matrix");
    }
}
