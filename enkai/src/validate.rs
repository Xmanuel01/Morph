use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use enkai_compiler::bytecode::Program;
use enkai_compiler::compiler::{compile_module, compile_package};
use enkai_compiler::modules::load_package;
use enkai_compiler::parser::parse_module_named;
use enkai_compiler::TypeChecker;
use enkai_runtime::object::Obj;
use enkai_runtime::{Value, VM};

#[derive(Debug, Clone, Deserialize)]
struct ValidationManifest {
    schema_version: u32,
    version_line: String,
    machine_profiles: BTreeMap<String, String>,
    suites: Vec<ValidationSuite>,
}

#[derive(Debug, Clone, Deserialize)]
struct ValidationSuite {
    id: String,
    description: String,
    kind: String,
    target: String,
    env: Option<BTreeMap<String, String>>,
    default_runs: Option<usize>,
    profile_case: Option<String>,
    expected_result: Option<serde_json::Value>,
    expected_output_hash: Option<String>,
    perf_metric: Option<String>,
    perf_direction: Option<String>,
    regression_budget_pct: Option<f64>,
    min_ffi_calls: Option<u64>,
    min_native_function_calls: Option<u64>,
    max_marshal_copy_ratio: Option<f64>,
    require_native_dominant: Option<bool>,
    reference_only: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct PerfBaselineManifest {
    schema_version: u32,
    version_line: String,
    profiles: BTreeMap<String, PerfBaselineProfile>,
}

#[derive(Debug, Clone, Deserialize)]
struct PerfBaselineProfile {
    machine_profile: String,
    suites: BTreeMap<String, PerfBaselineEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct PerfBaselineEntry {
    metric: String,
    better: String,
    baseline: f64,
    regression_budget_pct: f64,
}

#[derive(Debug, Clone)]
struct JsonOutputArgs {
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct DeterminismArgs {
    suite: String,
    runs: usize,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct PerfBaselineArgs {
    suite: String,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct Adam0Args {
    scenario: String,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug)]
struct ValidationRun {
    suite: ValidationSuite,
    elapsed_ms: f64,
    result: serde_json::Value,
    output_hash: String,
    profile: Option<serde_json::Value>,
}

#[derive(Debug)]
struct BaselineAssessment {
    profile: String,
    metric: String,
    better: String,
    budget_pct: f64,
    baseline_value: f64,
    observed_value: f64,
    passed: bool,
}

#[derive(Debug)]
struct PerfSample {
    run: ValidationRun,
    metrics: serde_json::Value,
    observed_value: f64,
    profile_path: PathBuf,
}

pub fn validate_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_validate_usage();
        return 1;
    }
    match args[0].as_str() {
        "ffi-correctness" => ffi_correctness_command(&args[1..]),
        "determinism" => determinism_command(&args[1..]),
        "perf-baseline" => perf_baseline_command(&args[1..]),
        "pool-safety" => pool_safety_command(&args[1..]),
        "adam0-cpu" => adam0_cpu_command(&args[1..]),
        _ => {
            eprintln!("enkai validate: unknown subcommand '{}'", args[0]);
            print_validate_usage();
            1
        }
    }
}

pub fn print_validate_usage() {
    eprintln!("  enkai validate ffi-correctness [--json] [--output <file>]");
    eprintln!(
        "  enkai validate determinism --suite <event_queue|sim_replay|adam0_reference_100> [--runs <n>] [--json] [--output <file>]"
    );
    eprintln!(
        "  enkai validate perf-baseline --suite <ffi_noop|sparse_dot|adam0_reference_100> [--json] [--output <file>]"
    );
    eprintln!("  enkai validate pool-safety [--json] [--output <file>]");
    eprintln!(
        "  enkai validate adam0-cpu --scenario <fake10|ref100|stress1000|target10000> [--json] [--output <file>]"
    );
}

fn ffi_correctness_command(args: &[String]) -> i32 {
    let parsed = match parse_json_output_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai validate ffi-correctness: {}", err);
            return 1;
        }
    };
    let manifest = match load_validation_manifest() {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai validate ffi-correctness: {}", err);
            return 1;
        }
    };
    let correctness_suite = match suite_by_id(&manifest, "ffi_correctness") {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate ffi-correctness: {}", err);
            return 1;
        }
    };
    let handle_suite = match suite_by_id(&manifest, "ffi_handle_live_count") {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate ffi-correctness: {}", err);
            return 1;
        }
    };

    let run = match run_validation_suite(&correctness_suite, None) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate ffi-correctness: {}", err);
            return 1;
        }
    };
    let handle_check = match run_validation_suite(&handle_suite, None) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate ffi-correctness: {}", err);
            return 1;
        }
    };

    let expected = correctness_suite
        .expected_result
        .clone()
        .unwrap_or(serde_json::Value::Null);
    let passed = run.result == expected && handle_check.result == json!(0);
    let payload = json!({
        "schema_version": 1,
        "validation": "ffi_correctness",
        "description": correctness_suite.description,
        "version_line": manifest.version_line,
        "passed": passed,
        "reference_machine_profiles": manifest.machine_profiles,
        "target": run.suite.target,
        "elapsed_ms": run.elapsed_ms,
        "actual": run.result,
        "expected": expected,
        "handle_live_count_after_run": handle_check.result,
        "output_hash": run.output_hash,
    });
    emit_validation_payload(&payload, &parsed)
}

fn determinism_command(args: &[String]) -> i32 {
    let parsed = match parse_determinism_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai validate determinism: {}", err);
            return 1;
        }
    };
    let manifest = match load_validation_manifest() {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai validate determinism: {}", err);
            return 1;
        }
    };
    let suite = match suite_by_id(&manifest, &parsed.suite) {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate determinism: {}", err);
            return 1;
        }
    };

    let mut native_hashes = Vec::with_capacity(parsed.runs);
    let mut vm_hashes = Vec::with_capacity(parsed.runs);
    let mut first_native_result = None;
    let mut first_vm_result = None;
    for _ in 0..parsed.runs {
        let native = match run_validation_suite(&suite, None) {
            Ok(run) => run,
            Err(err) => {
                eprintln!("enkai validate determinism: {}", err);
                return 1;
            }
        };
        let vm_only = match run_validation_suite(
            &suite,
            Some(BTreeMap::from([(
                "ENKAI_SIM_ACCEL".to_string(),
                "0".to_string(),
            )])),
        ) {
            Ok(run) => run,
            Err(err) => {
                eprintln!("enkai validate determinism: {}", err);
                return 1;
            }
        };
        if first_native_result.is_none() {
            first_native_result = Some(native.result.clone());
        }
        if first_vm_result.is_none() {
            first_vm_result = Some(vm_only.result.clone());
        }
        native_hashes.push(native.output_hash);
        vm_hashes.push(vm_only.output_hash);
    }
    let native_unique = native_hashes.iter().cloned().collect::<BTreeSet<_>>();
    let vm_unique = vm_hashes.iter().cloned().collect::<BTreeSet<_>>();
    let native_hash = native_hashes.first().cloned().unwrap_or_default();
    let vm_hash = vm_hashes.first().cloned().unwrap_or_default();
    let passed = native_unique.len() == 1 && vm_unique.len() == 1 && native_hash == vm_hash;
    let payload = json!({
        "schema_version": 1,
        "validation": "determinism",
        "suite": parsed.suite,
        "description": suite.description,
        "default_runs": suite.default_runs,
        "runs": parsed.runs,
        "version_line": manifest.version_line,
        "passed": passed,
        "native_hashes": native_hashes,
        "vm_fallback_hashes": vm_hashes,
        "native_result": first_native_result.unwrap_or(serde_json::Value::Null),
        "vm_fallback_result": first_vm_result.unwrap_or(serde_json::Value::Null),
    });
    emit_validation_payload(
        &payload,
        &JsonOutputArgs {
            json: parsed.json,
            output: parsed.output,
        },
    )
}

fn perf_baseline_command(args: &[String]) -> i32 {
    let parsed = match parse_perf_baseline_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai validate perf-baseline: {}", err);
            return 1;
        }
    };
    let manifest = match load_validation_manifest() {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai validate perf-baseline: {}", err);
            return 1;
        }
    };
    let suite = match suite_by_id(&manifest, &parsed.suite) {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate perf-baseline: {}", err);
            return 1;
        }
    };
    let program = match load_suite_program(&suite) {
        Ok(program) => program,
        Err(err) => {
            eprintln!("enkai validate perf-baseline: {}", err);
            return 1;
        }
    };
    let sample_count = suite.default_runs.unwrap_or(5).max(1);
    let validation_dir = workspace_root().join("artifacts").join("validation");
    let mut samples = Vec::with_capacity(sample_count);
    for sample_idx in 0..sample_count {
        let profile_path = validation_dir.join(format!(
            "perf_{}_profile_sample_{}.json",
            parsed.suite, sample_idx
        ));
        let run = match execute_validation_program(&program, &suite, None, Some(&profile_path)) {
            Ok(run) => run,
            Err(err) => {
                eprintln!("enkai validate perf-baseline: {}", err);
                return 1;
            }
        };
        let metrics = match perf_metrics_for_suite(&parsed.suite, &run) {
            Ok(metrics) => metrics,
            Err(err) => {
                eprintln!("enkai validate perf-baseline: {}", err);
                return 1;
            }
        };
        let observed_value = match observed_perf_metric(&suite, &metrics) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("enkai validate perf-baseline: {}", err);
                return 1;
            }
        };
        samples.push(PerfSample {
            run,
            metrics,
            observed_value,
            profile_path,
        });
    }
    let metric_direction = suite
        .perf_direction
        .clone()
        .unwrap_or_else(|| "lower".to_string());
    samples.sort_by(|left, right| {
        left.observed_value
            .partial_cmp(&right.observed_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let median_index = samples.len() / 2;
    let selected = samples.swap_remove(median_index);
    for sample in samples {
        let _ = fs::remove_file(sample.profile_path);
    }
    let run = selected.run;
    let metrics = selected.metrics;
    let assessment = match assess_perf_baseline(&parsed.suite, &suite, &metrics) {
        Ok(assessment) => assessment,
        Err(err) => {
            eprintln!("enkai validate perf-baseline: {}", err);
            return 1;
        }
    };
    let passed = assessment.as_ref().map(|item| item.passed).unwrap_or(true);
    let payload = json!({
        "schema_version": 1,
        "validation": "perf_baseline",
        "suite": parsed.suite,
        "description": suite.description,
        "version_line": manifest.version_line,
        "passed": passed,
        "reference_machine_profiles": manifest.machine_profiles,
        "target": run.suite.target,
        "samples": sample_count,
        "sample_selection": {
            "strategy": "median",
            "metric_direction": metric_direction,
            "selected_metric": selected.observed_value
        },
        "elapsed_ms": run.elapsed_ms,
        "metrics": metrics,
        "baseline_assessment": assessment.as_ref().map(|item| json!({
            "profile": item.profile,
            "metric": item.metric,
            "better": item.better,
            "budget_pct": item.budget_pct,
            "baseline_value": item.baseline_value,
            "observed_value": item.observed_value,
            "passed": item.passed
        })),
        "profile": run.profile,
        "output_hash": run.output_hash,
    });
    emit_validation_payload(
        &payload,
        &JsonOutputArgs {
            json: parsed.json,
            output: parsed.output,
        },
    )
}

fn pool_safety_command(args: &[String]) -> i32 {
    let parsed = match parse_json_output_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai validate pool-safety: {}", err);
            return 1;
        }
    };
    let manifest = match load_validation_manifest() {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai validate pool-safety: {}", err);
            return 1;
        }
    };
    let suite = match suite_by_id(&manifest, "pool_safety") {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate pool-safety: {}", err);
            return 1;
        }
    };
    let handle_suite = match suite_by_id(&manifest, "ffi_handle_live_count") {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate pool-safety: {}", err);
            return 1;
        }
    };
    let run = match run_validation_suite(&suite, None) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate pool-safety: {}", err);
            return 1;
        }
    };
    let handle_check = match run_validation_suite(&handle_suite, None) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate pool-safety: {}", err);
            return 1;
        }
    };
    let expected = suite
        .expected_result
        .clone()
        .unwrap_or(serde_json::Value::Null);
    let passed = run.result == expected && handle_check.result == json!(0);
    let payload = json!({
        "schema_version": 1,
        "validation": "pool_safety",
        "description": suite.description,
        "version_line": manifest.version_line,
        "passed": passed,
        "target": run.suite.target,
        "elapsed_ms": run.elapsed_ms,
        "actual": run.result,
        "expected": expected,
        "handle_live_count_after_run": handle_check.result,
        "output_hash": run.output_hash,
    });
    emit_validation_payload(&payload, &parsed)
}

fn adam0_cpu_command(args: &[String]) -> i32 {
    let parsed = match parse_adam0_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai validate adam0-cpu: {}", err);
            return 1;
        }
    };
    let manifest = match load_validation_manifest() {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai validate adam0-cpu: {}", err);
            return 1;
        }
    };
    let suite_id = match parsed.scenario.as_str() {
        "fake10" => "adam0_fake_10",
        "ref100" => "adam0_reference_100",
        "stress1000" => "adam0_reference_1000",
        "target10000" => "adam0_reference_10000",
        other => {
            eprintln!(
                "enkai validate adam0-cpu: unsupported scenario '{}'; expected fake10|ref100|stress1000|target10000",
                other
            );
            return 1;
        }
    };
    let suite = match suite_by_id(&manifest, suite_id) {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate adam0-cpu: {}", err);
            return 1;
        }
    };
    let profile_path = workspace_root()
        .join("artifacts")
        .join("validation")
        .join(format!("adam0_{}_profile.json", parsed.scenario));
    let run = match run_validation_suite_with_profile(&suite, None, &profile_path) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate adam0-cpu: {}", err);
            return 1;
        }
    };
    let handle_suite = match suite_by_id(&manifest, "ffi_handle_live_count") {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate adam0-cpu: {}", err);
            return 1;
        }
    };
    let handle_check = match run_validation_suite(&handle_suite, None) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate adam0-cpu: {}", err);
            return 1;
        }
    };
    let profile = run.profile.clone().unwrap_or(serde_json::Value::Null);
    let counters = profile
        .get("counters")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let timing = profile
        .get("timing_ms")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let native_calls = counters
        .get("native_function_calls")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let ffi_calls = counters
        .get("ffi_calls")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let marshal_copy_ops = counters
        .get("marshal_copy_ops")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let native_ms = timing
        .get("native_calls")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    let vm_ms = timing
        .get("vm_exec")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    let marshal_copy_ratio = if ffi_calls > 0 {
        marshal_copy_ops as f64 / ffi_calls as f64
    } else {
        0.0
    };
    let hot_path_native_dominant = native_ms >= vm_ms;
    let output_hash_matches = suite
        .expected_output_hash
        .as_ref()
        .map(|expected| expected == &run.output_hash)
        .unwrap_or(true);
    let min_ffi_ok = suite
        .min_ffi_calls
        .map(|minimum| ffi_calls >= minimum)
        .unwrap_or(true);
    let min_native_ok = suite
        .min_native_function_calls
        .map(|minimum| native_calls >= minimum)
        .unwrap_or(true);
    let marshal_ratio_ok = suite
        .max_marshal_copy_ratio
        .map(|maximum| marshal_copy_ratio <= maximum)
        .unwrap_or(true);
    let native_dominant_ok = suite
        .require_native_dominant
        .map(|required| !required || hot_path_native_dominant)
        .unwrap_or(true);
    let handle_live_count_zero = handle_check.result == json!(0);
    let passed = output_hash_matches
        && min_ffi_ok
        && min_native_ok
        && marshal_ratio_ok
        && native_dominant_ok
        && handle_live_count_zero;
    let payload = json!({
        "schema_version": 1,
        "validation": "adam0_cpu",
        "scenario": parsed.scenario,
        "description": suite.description,
        "reference_only": suite.reference_only.unwrap_or(false),
        "version_line": manifest.version_line,
        "target": run.suite.target,
        "elapsed_ms": run.elapsed_ms,
        "passed": passed,
        "result": run.result,
        "profile": run.profile,
        "hot_path_assessment": {
            "ffi_calls": ffi_calls,
            "marshal_copy_ops": marshal_copy_ops,
            "marshal_copy_ratio": marshal_copy_ratio,
            "native_function_calls": native_calls,
            "native_time_ms": native_ms,
            "vm_exec_ms": vm_ms,
            "native_dominant": hot_path_native_dominant,
        },
        "proof_checks": {
            "expected_output_hash": suite.expected_output_hash,
            "output_hash_matches": output_hash_matches,
            "min_ffi_calls": suite.min_ffi_calls,
            "min_ffi_calls_passed": min_ffi_ok,
            "min_native_function_calls": suite.min_native_function_calls,
            "min_native_function_calls_passed": min_native_ok,
            "max_marshal_copy_ratio": suite.max_marshal_copy_ratio,
            "max_marshal_copy_ratio_passed": marshal_ratio_ok,
            "require_native_dominant": suite.require_native_dominant.unwrap_or(false),
            "require_native_dominant_passed": native_dominant_ok,
            "handle_live_count_after_run": handle_check.result,
            "handle_live_count_zero": handle_live_count_zero
        },
        "output_hash": run.output_hash,
    });
    emit_validation_payload(
        &payload,
        &JsonOutputArgs {
            json: parsed.json,
            output: parsed.output,
        },
    )
}

fn emit_validation_payload(payload: &serde_json::Value, options: &JsonOutputArgs) -> i32 {
    let text = match serde_json::to_string_pretty(payload) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("enkai validate: failed to serialize report: {}", err);
            return 1;
        }
    };
    if options.json {
        println!("{}", text);
    }
    if let Some(path) = options.output.as_ref() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    eprintln!(
                        "enkai validate: failed to create output directory {}: {}",
                        parent.display(),
                        err
                    );
                    return 1;
                }
            }
        }
        if let Err(err) = fs::write(path, text.as_bytes()) {
            eprintln!(
                "enkai validate: failed to write report {}: {}",
                path.display(),
                err
            );
            return 1;
        }
    }
    let passed = payload
        .get("passed")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    if passed {
        0
    } else {
        1
    }
}

fn parse_json_output_args(args: &[String]) -> Result<JsonOutputArgs, String> {
    let mut parsed = JsonOutputArgs {
        json: false,
        output: None,
    };
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--json" => parsed.json = true,
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                parsed.output = Some(PathBuf::from(&args[idx]));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    Ok(parsed)
}

fn parse_determinism_args(args: &[String]) -> Result<DeterminismArgs, String> {
    let mut suite = None;
    let mut runs = 10usize;
    let mut json = false;
    let mut output = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--suite" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--suite requires a value".to_string());
                }
                suite = Some(args[idx].clone());
            }
            "--runs" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--runs requires a value".to_string());
                }
                runs = args[idx]
                    .parse()
                    .map_err(|_| format!("invalid --runs value '{}'", args[idx]))?;
            }
            "--json" => json = true,
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                output = Some(PathBuf::from(&args[idx]));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    let suite = suite.ok_or_else(|| "--suite <name> is required".to_string())?;
    Ok(DeterminismArgs {
        suite,
        runs,
        json,
        output,
    })
}

fn parse_perf_baseline_args(args: &[String]) -> Result<PerfBaselineArgs, String> {
    let mut suite = None;
    let mut json = false;
    let mut output = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--suite" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--suite requires a value".to_string());
                }
                suite = Some(args[idx].clone());
            }
            "--json" => json = true,
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                output = Some(PathBuf::from(&args[idx]));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    Ok(PerfBaselineArgs {
        suite: suite.ok_or_else(|| "--suite <name> is required".to_string())?,
        json,
        output,
    })
}

fn parse_adam0_args(args: &[String]) -> Result<Adam0Args, String> {
    let mut scenario = None;
    let mut json = false;
    let mut output = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--scenario" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--scenario requires a value".to_string());
                }
                scenario = Some(args[idx].clone());
            }
            "--json" => json = true,
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                output = Some(PathBuf::from(&args[idx]));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    Ok(Adam0Args {
        scenario: scenario.ok_or_else(|| "--scenario <name> is required".to_string())?,
        json,
        output,
    })
}

fn load_validation_manifest() -> Result<ValidationManifest, String> {
    let manifest: ValidationManifest =
        serde_json::from_str(include_str!("../contracts/validation_cpu_v3_0_0.json"))
            .map_err(|err| format!("failed to parse validation manifest: {}", err))?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported validation manifest schema_version {}; expected 1",
            manifest.schema_version
        ));
    }
    Ok(manifest)
}

fn load_perf_baseline_manifest() -> Result<PerfBaselineManifest, String> {
    let path = workspace_root()
        .join("bench")
        .join("baselines")
        .join("validation_cpu_v3_0_0.json");
    let text = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    let manifest: PerfBaselineManifest = serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse {}: {}", path.display(), err))?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported perf baseline manifest schema_version {}; expected 1",
            manifest.schema_version
        ));
    }
    Ok(manifest)
}

fn active_machine_profile_id() -> String {
    if let Ok(profile) = env::var("ENKAI_VALIDATION_MACHINE_PROFILE") {
        if !profile.trim().is_empty() {
            return profile;
        }
    }
    if cfg!(target_os = "windows") {
        "windows_local".to_string()
    } else {
        "linux_local".to_string()
    }
}

fn assess_perf_baseline(
    suite_id: &str,
    suite: &ValidationSuite,
    metrics: &serde_json::Value,
) -> Result<Option<BaselineAssessment>, String> {
    let metric_name = if let Some(metric) = suite.perf_metric.as_ref() {
        metric.clone()
    } else {
        return Ok(None);
    };
    let observed_value = observed_perf_metric(suite, metrics)?;
    let baseline_manifest = load_perf_baseline_manifest()?;
    let _ = &baseline_manifest.version_line;
    let profile_id = active_machine_profile_id();
    let Some(profile) = baseline_manifest.profiles.get(&profile_id) else {
        return Ok(None);
    };
    let _ = &profile.machine_profile;
    let Some(entry) = profile.suites.get(suite_id) else {
        return Ok(None);
    };
    if entry.metric != metric_name {
        return Err(format!(
            "baseline metric mismatch for suite '{}': manifest expected '{}', baseline has '{}'",
            suite_id, metric_name, entry.metric
        ));
    }
    let better = suite
        .perf_direction
        .clone()
        .unwrap_or_else(|| entry.better.clone());
    let budget_pct = suite
        .regression_budget_pct
        .unwrap_or(entry.regression_budget_pct);
    let passed = match better.as_str() {
        "lower" => observed_value <= entry.baseline * (1.0 + budget_pct / 100.0),
        "higher" => observed_value >= entry.baseline * (1.0 - budget_pct / 100.0),
        other => {
            return Err(format!(
                "unsupported perf direction '{}' for suite '{}'",
                other, suite_id
            ))
        }
    };
    Ok(Some(BaselineAssessment {
        profile: profile_id,
        metric: metric_name,
        better,
        budget_pct,
        baseline_value: entry.baseline,
        observed_value,
        passed,
    }))
}

fn suite_by_id(manifest: &ValidationManifest, id: &str) -> Result<ValidationSuite, String> {
    manifest
        .suites
        .iter()
        .find(|suite| suite.id == id)
        .cloned()
        .ok_or_else(|| format!("unknown validation suite '{}'", id))
}

fn run_validation_suite(
    suite: &ValidationSuite,
    override_env: Option<BTreeMap<String, String>>,
) -> Result<ValidationRun, String> {
    run_validation_suite_impl(suite, override_env, None)
}

fn run_validation_suite_with_profile(
    suite: &ValidationSuite,
    override_env: Option<BTreeMap<String, String>>,
    profile_output: &Path,
) -> Result<ValidationRun, String> {
    run_validation_suite_impl(suite, override_env, Some(profile_output))
}

fn run_validation_suite_impl(
    suite: &ValidationSuite,
    override_env: Option<BTreeMap<String, String>>,
    profile_output: Option<&Path>,
) -> Result<ValidationRun, String> {
    if suite.kind != "run" {
        return Err(format!(
            "unsupported validation suite kind '{}' for {}",
            suite.kind, suite.id
        ));
    }
    let target = workspace_root().join(&suite.target);
    if !target.is_file() {
        return Err(format!(
            "validation target '{}' not found at {}",
            suite.id,
            target.display()
        ));
    }
    let program = load_program_from_target(&target)?;
    execute_validation_program(&program, suite, override_env, profile_output)
}

fn execute_validation_program(
    program: &Program,
    suite: &ValidationSuite,
    override_env: Option<BTreeMap<String, String>>,
    profile_output: Option<&Path>,
) -> Result<ValidationRun, String> {
    let env_guard = crate::env_guard();
    let mut restore = Vec::new();
    if let Some(profile) = profile_output {
        let profile_str = profile.display().to_string();
        restore.push((
            "ENKAI_BENCH_PROFILE_OUT".to_string(),
            env::var("ENKAI_BENCH_PROFILE_OUT").ok(),
        ));
        env::set_var("ENKAI_BENCH_PROFILE_OUT", profile_str);
        restore.push((
            "ENKAI_BENCH_PROFILE_CASE".to_string(),
            env::var("ENKAI_BENCH_PROFILE_CASE").ok(),
        ));
        env::set_var(
            "ENKAI_BENCH_PROFILE_CASE",
            suite.profile_case.as_deref().unwrap_or(suite.id.as_str()),
        );
    }
    let mut merged_env = suite.env.clone().unwrap_or_default();
    if let Some(override_env) = override_env {
        for (key, value) in override_env {
            merged_env.insert(key, value);
        }
    }
    for (key, value) in merged_env {
        restore.push((key.clone(), env::var(&key).ok()));
        env::set_var(key, value);
    }
    drop(env_guard);

    let started = Instant::now();
    let mut vm = VM::new(false, false, false, false);
    let value = vm
        .run(program)
        .map_err(|err| format!("validation suite '{}' runtime error: {}", suite.id, err));
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    let _guard = crate::env_guard();
    for (key, previous) in restore.into_iter().rev() {
        if let Some(previous) = previous {
            env::set_var(key, previous);
        } else {
            env::remove_var(key);
        }
    }
    drop(_guard);

    let value = value?;
    let result = canonicalize_json(&value_to_json(&value));
    let output_hash = hash_json(&result)?;
    let profile = if let Some(profile_output) = profile_output {
        Some(read_json(profile_output)?)
    } else {
        None
    };

    Ok(ValidationRun {
        suite: suite.clone(),
        elapsed_ms,
        result,
        output_hash,
        profile,
    })
}

fn perf_metrics_for_suite(
    suite_id: &str,
    run: &ValidationRun,
) -> Result<serde_json::Value, String> {
    let iterations = run
        .result
        .get("iterations")
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    let elapsed_s = run.elapsed_ms / 1000.0;
    match suite_id {
        "ffi_noop" => Ok(json!({
            "iterations": iterations,
            "ns_per_call": if iterations > 0.0 { (run.elapsed_ms * 1_000_000.0) / iterations } else { 0.0 },
        })),
        "sparse_dot" => Ok(json!({
            "iterations": iterations,
            "ops_per_sec": if elapsed_s > 0.0 { iterations / elapsed_s } else { 0.0 },
        })),
        "adam0_reference_100" => Ok(json!({
            "agent_count": run.result.get("agent_count").and_then(|value| value.as_i64()).unwrap_or_default(),
            "elapsed_ms": run.elapsed_ms,
            "ops_per_sec": if elapsed_s > 0.0 { 1.0 / elapsed_s } else { 0.0 },
        })),
        _ => Err(format!("unsupported perf baseline suite '{}'", suite_id)),
    }
}

fn observed_perf_metric(
    suite: &ValidationSuite,
    metrics: &serde_json::Value,
) -> Result<f64, String> {
    let metric_name = suite
        .perf_metric
        .as_ref()
        .ok_or_else(|| format!("suite '{}' is missing perf_metric", suite.id))?;
    metrics
        .get(metric_name)
        .and_then(|value| value.as_f64())
        .ok_or_else(|| format!("performance metric '{}' missing from report", metric_name))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn load_suite_program(suite: &ValidationSuite) -> Result<Program, String> {
    let target = workspace_root().join(&suite.target);
    if !target.is_file() {
        return Err(format!(
            "validation target '{}' not found at {}",
            suite.id,
            target.display()
        ));
    }
    load_program_from_target(&target)
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
    let (root, entry) = super::resolve_entry(target).map_err(|err| err.to_string())?;
    match super::load_cached_program(&root, &entry) {
        Ok(Some(program)) => Ok(program),
        Ok(None) | Err(_) => {
            let package = load_package(&entry).map_err(|err| err.to_string())?;
            TypeChecker::check_package(&package).map_err(|err| err.message.clone())?;
            compile_package(&package).map_err(|err| err.message)
        }
    }
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
                let mut out = serde_json::Map::new();
                let mut keys = map.keys().cloned().collect::<Vec<_>>();
                keys.sort();
                for key in keys {
                    if let Some(value) = map.get(&key) {
                        out.insert(key, value_to_json(value));
                    }
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

fn canonicalize_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.iter().map(canonicalize_json).collect::<Vec<_>>())
        }
        serde_json::Value::Object(map) => {
            let mut ordered = serde_json::Map::new();
            let mut keys = map.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                if let Some(value) = map.get(&key) {
                    ordered.insert(key, canonicalize_json(value));
                }
            }
            serde_json::Value::Object(ordered)
        }
        other => other.clone(),
    }
}

fn hash_json(value: &serde_json::Value) -> Result<String, String> {
    let canonical = serde_json::to_vec(value).map_err(|err| err.to_string())?;
    let digest = Sha256::digest(&canonical);
    Ok(format!("{:x}", digest))
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse {}: {}", path.display(), err))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_output_args_accepts_output() {
        let parsed = parse_json_output_args(&[
            "--json".to_string(),
            "--output".to_string(),
            "artifacts/validation/out.json".to_string(),
        ])
        .expect("parsed");
        assert!(parsed.json);
        assert_eq!(
            parsed.output,
            Some(PathBuf::from("artifacts/validation/out.json"))
        );
    }

    #[test]
    fn parse_determinism_args_defaults_runs() {
        let parsed = parse_determinism_args(&["--suite".to_string(), "event_queue".to_string()])
            .expect("parsed");
        assert_eq!(parsed.suite, "event_queue");
        assert_eq!(parsed.runs, 10);
    }

    #[test]
    fn parse_adam0_args_requires_scenario() {
        assert!(parse_adam0_args(&[]).is_err());
    }

    #[test]
    fn validation_manifest_loads() {
        let manifest = load_validation_manifest().expect("manifest");
        assert_eq!(manifest.schema_version, 1);
        assert!(manifest
            .suites
            .iter()
            .any(|suite| suite.id == "ffi_correctness"));
        assert!(manifest
            .suites
            .iter()
            .any(|suite| suite.id == "adam0_reference_10000"));
    }

    #[test]
    fn perf_baseline_manifest_loads() {
        let manifest = load_perf_baseline_manifest().expect("baseline manifest");
        assert_eq!(manifest.schema_version, 1);
        assert!(manifest.profiles.contains_key("windows_local"));
    }

    #[test]
    fn canonicalize_json_orders_object_keys() {
        let value = json!({
            "b": 2,
            "a": {
                "d": 4,
                "c": 3
            }
        });
        let canonical = canonicalize_json(&value);
        let keys = canonical
            .as_object()
            .expect("object")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(keys, vec!["a".to_string(), "b".to_string()]);
    }
}
