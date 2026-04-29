use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
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
    require_native_path: Option<bool>,
    require_coroutine_counters: Option<bool>,
    perf_metric: Option<String>,
    perf_direction: Option<String>,
    regression_budget_pct: Option<f64>,
    min_ffi_calls: Option<u64>,
    min_native_function_calls: Option<u64>,
    max_marshal_copy_ratio: Option<f64>,
    require_native_dominant: Option<bool>,
    required_native_counters: Option<BTreeMap<String, u64>>,
    require_kernel_native_dominance: Option<bool>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ValidationSuiteDispatchManifest {
    pub(crate) schema_version: u32,
    pub(crate) profile: String,
    pub(crate) emit_json: bool,
    pub(crate) result_output: Option<String>,
    pub(crate) command: ValidationSuiteDispatchCommand,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "subcommand", rename_all = "kebab-case")]
pub(crate) enum ValidationSuiteDispatchCommand {
    FfiCorrectness,
    FfiSafety,
    Determinism { suite: String, runs: usize },
    PerfBaseline { suite: String },
    PoolSafety,
    Adam0Cpu { scenario: String },
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

fn profile_counter_u64(profile: &serde_json::Value, key: &str) -> u64 {
    profile
        .get("counters")
        .and_then(|value| value.get(key))
        .and_then(|value| value.as_u64())
        .unwrap_or(0)
}

fn evaluate_required_native_counters(
    profile: &serde_json::Value,
    required: Option<&BTreeMap<String, u64>>,
) -> (serde_json::Value, bool, u64) {
    let mut checks = serde_json::Map::new();
    let mut all_passed = true;
    let mut total_observed = 0u64;
    if let Some(required) = required {
        for (counter, minimum) in required {
            let observed = profile_counter_u64(profile, counter);
            total_observed = total_observed.saturating_add(observed);
            let passed = observed >= *minimum;
            if !passed {
                all_passed = false;
            }
            checks.insert(
                counter.clone(),
                json!({
                    "minimum": minimum,
                    "observed": observed,
                    "passed": passed
                }),
            );
        }
    }
    (
        serde_json::Value::Object(checks),
        all_passed,
        total_observed,
    )
}

fn native_path_exercised(profile: &serde_json::Value) -> bool {
    profile_counter_u64(profile, "ffi_calls") > 0
        && profile_counter_u64(profile, "native_function_calls") > 0
}

fn simulation_audit_from_result(
    suite: &ValidationSuite,
    result: &serde_json::Value,
    profile: &serde_json::Value,
) -> Result<Option<serde_json::Value>, String> {
    let snapshot = result.get("snapshot");
    let replay_snapshot = result.get("replay_snapshot");
    let event_log = result.get("log");
    if snapshot.is_none() && replay_snapshot.is_none() && event_log.is_none() {
        return Ok(None);
    }
    let snapshot_hash = snapshot.map(hash_json).transpose()?.unwrap_or_default();
    let replay_hash = replay_snapshot
        .map(hash_json)
        .transpose()?
        .unwrap_or_default();
    let event_log_hash = event_log.map(hash_json).transpose()?.unwrap_or_default();
    let config_hash = hash_json(&canonicalize_json(&json!({
        "suite": suite.id,
        "target": suite.target,
        "env": suite.env,
    })))?;
    let seed = snapshot
        .and_then(|value| value.get("seed"))
        .cloned()
        .or_else(|| result.get("seed").cloned())
        .unwrap_or(serde_json::Value::Null);
    Ok(Some(json!({
        "seed": seed,
        "config_hash": config_hash,
        "event_log_hash": event_log_hash,
        "snapshot_hash": snapshot_hash,
        "replay_hash": replay_hash,
        "task_counters": {
            "sim_coroutines_spawned": profile_counter_u64(profile, "sim_coroutines_spawned"),
            "sim_coroutine_emits": profile_counter_u64(profile, "sim_coroutine_emits"),
            "sim_coroutine_next_waits": profile_counter_u64(profile, "sim_coroutine_next_waits")
        }
    })))
}

pub fn validate_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_validate_usage();
        return 1;
    }
    match args[0].as_str() {
        "suite-manifest" => suite_manifest_command(&args[1..]),
        "suite-exec" => suite_exec_command(&args[1..]),
        "ffi-correctness" => dispatch_validation_subcommand("ffi-correctness", &args[1..]),
        "ffi-safety" => dispatch_validation_subcommand("ffi-safety", &args[1..]),
        "determinism" => dispatch_validation_subcommand("determinism", &args[1..]),
        "perf-baseline" => dispatch_validation_subcommand("perf-baseline", &args[1..]),
        "pool-safety" => dispatch_validation_subcommand("pool-safety", &args[1..]),
        "adam0-cpu" => dispatch_validation_subcommand("adam0-cpu", &args[1..]),
        _ => {
            eprintln!("enkai validate: unknown subcommand '{}'", args[0]);
            print_validate_usage();
            1
        }
    }
}

pub fn print_validate_usage() {
    eprintln!(
        "  enkai validate suite-manifest <ffi-correctness|ffi-safety|determinism|perf-baseline|pool-safety|adam0-cpu> [suite args] [--manifest-output <file>]"
    );
    eprintln!("  enkai validate suite-exec --manifest <file>");
    eprintln!("  enkai validate ffi-correctness [--json] [--output <file>]");
    eprintln!("  enkai validate ffi-safety [--json] [--output <file>]");
    eprintln!(
        "  enkai validate determinism --suite <event_queue|sim_replay|sim_coroutines|adam0_reference_100> [--runs <n>] [--json] [--output <file>]"
    );
    eprintln!(
        "  enkai validate perf-baseline --suite <ffi_noop|sparse_dot|adam0_reference_100|adam0_reference_1000|adam0_reference_10000> [--json] [--output <file>]"
    );
    eprintln!("  enkai validate pool-safety [--json] [--output <file>]");
    eprintln!(
        "  enkai validate adam0-cpu --scenario <fake10|ref100|stress1000|target10000> [--json] [--output <file>]"
    );
}

fn dispatch_validation_subcommand(subcommand: &str, args: &[String]) -> i32 {
    let manifest = match build_validation_suite_manifest(subcommand, args) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai validate {}: {}", subcommand, err);
            return 1;
        }
    };
    execute_validation_suite_manifest(&manifest)
}

fn suite_manifest_command(args: &[String]) -> i32 {
    let (subcommand, suite_args, manifest_output) = match parse_suite_manifest_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai validate suite-manifest: {}", err);
            print_validate_usage();
            return 1;
        }
    };
    let manifest = match build_validation_suite_manifest(&subcommand, &suite_args) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai validate suite-manifest: {}", err);
            return 1;
        }
    };
    emit_suite_manifest(&manifest, manifest_output.as_deref())
}

fn suite_exec_command(args: &[String]) -> i32 {
    let manifest_path = match parse_manifest_flag("suite-exec", args) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{}", err);
            print_validate_usage();
            return 1;
        }
    };
    let manifest = match load_json_manifest::<ValidationSuiteDispatchManifest>(&manifest_path) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai validate suite-exec: {}", err);
            return 1;
        }
    };
    execute_validation_suite_manifest(&manifest)
}

fn build_validation_suite_manifest(
    subcommand: &str,
    args: &[String],
) -> Result<ValidationSuiteDispatchManifest, String> {
    match subcommand {
        "ffi-correctness" => {
            let parsed = parse_json_output_args(args)?;
            Ok(ValidationSuiteDispatchManifest {
                schema_version: 1,
                profile: "validation_suite_dispatch".to_string(),
                emit_json: parsed.json,
                result_output: parsed.output.map(|value| value.display().to_string()),
                command: ValidationSuiteDispatchCommand::FfiCorrectness,
            })
        }
        "ffi-safety" => {
            let parsed = parse_json_output_args(args)?;
            Ok(ValidationSuiteDispatchManifest {
                schema_version: 1,
                profile: "validation_suite_dispatch".to_string(),
                emit_json: parsed.json,
                result_output: parsed.output.map(|value| value.display().to_string()),
                command: ValidationSuiteDispatchCommand::FfiSafety,
            })
        }
        "determinism" => {
            let parsed = parse_determinism_args(args)?;
            Ok(ValidationSuiteDispatchManifest {
                schema_version: 1,
                profile: "validation_suite_dispatch".to_string(),
                emit_json: parsed.json,
                result_output: parsed.output.map(|value| value.display().to_string()),
                command: ValidationSuiteDispatchCommand::Determinism {
                    suite: parsed.suite,
                    runs: parsed.runs,
                },
            })
        }
        "perf-baseline" => {
            let parsed = parse_perf_baseline_args(args)?;
            Ok(ValidationSuiteDispatchManifest {
                schema_version: 1,
                profile: "validation_suite_dispatch".to_string(),
                emit_json: parsed.json,
                result_output: parsed.output.map(|value| value.display().to_string()),
                command: ValidationSuiteDispatchCommand::PerfBaseline {
                    suite: parsed.suite,
                },
            })
        }
        "pool-safety" => {
            let parsed = parse_json_output_args(args)?;
            Ok(ValidationSuiteDispatchManifest {
                schema_version: 1,
                profile: "validation_suite_dispatch".to_string(),
                emit_json: parsed.json,
                result_output: parsed.output.map(|value| value.display().to_string()),
                command: ValidationSuiteDispatchCommand::PoolSafety,
            })
        }
        "adam0-cpu" => {
            let parsed = parse_adam0_args(args)?;
            Ok(ValidationSuiteDispatchManifest {
                schema_version: 1,
                profile: "validation_suite_dispatch".to_string(),
                emit_json: parsed.json,
                result_output: parsed.output.map(|value| value.display().to_string()),
                command: ValidationSuiteDispatchCommand::Adam0Cpu {
                    scenario: parsed.scenario,
                },
            })
        }
        other => Err(format!(
            "unknown suite '{}'; expected ffi-correctness|ffi-safety|determinism|perf-baseline|pool-safety|adam0-cpu",
            other
        )),
    }
}

pub(crate) fn execute_validation_suite_manifest(manifest: &ValidationSuiteDispatchManifest) -> i32 {
    let args = validation_manifest_args(manifest);
    match &manifest.command {
        ValidationSuiteDispatchCommand::FfiCorrectness => ffi_correctness_command(&args),
        ValidationSuiteDispatchCommand::FfiSafety => ffi_safety_command(&args),
        ValidationSuiteDispatchCommand::Determinism { .. } => determinism_command(&args),
        ValidationSuiteDispatchCommand::PerfBaseline { .. } => perf_baseline_command(&args),
        ValidationSuiteDispatchCommand::PoolSafety => pool_safety_command(&args),
        ValidationSuiteDispatchCommand::Adam0Cpu { .. } => adam0_cpu_command(&args),
    }
}

fn validation_manifest_args(manifest: &ValidationSuiteDispatchManifest) -> Vec<String> {
    let mut args = Vec::new();
    match &manifest.command {
        ValidationSuiteDispatchCommand::FfiCorrectness => {}
        ValidationSuiteDispatchCommand::FfiSafety => {}
        ValidationSuiteDispatchCommand::Determinism { suite, runs } => {
            args.push("--suite".to_string());
            args.push(suite.clone());
            args.push("--runs".to_string());
            args.push(runs.to_string());
        }
        ValidationSuiteDispatchCommand::PerfBaseline { suite } => {
            args.push("--suite".to_string());
            args.push(suite.clone());
        }
        ValidationSuiteDispatchCommand::PoolSafety => {}
        ValidationSuiteDispatchCommand::Adam0Cpu { scenario } => {
            args.push("--scenario".to_string());
            args.push(scenario.clone());
        }
    }
    if manifest.emit_json {
        args.push("--json".to_string());
    }
    if let Some(output) = &manifest.result_output {
        args.push("--output".to_string());
        args.push(output.clone());
    }
    args
}

fn parse_suite_manifest_args(
    args: &[String],
) -> Result<(String, Vec<String>, Option<PathBuf>), String> {
    let Some(subcommand) = args.first().cloned() else {
        return Err("missing suite name".to_string());
    };
    let mut suite_args = Vec::new();
    let mut manifest_output = None;
    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--manifest-output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--manifest-output requires a value".to_string());
                }
                manifest_output = Some(PathBuf::from(&args[idx]));
            }
            other => suite_args.push(other.to_string()),
        }
        idx += 1;
    }
    Ok((subcommand, suite_args, manifest_output))
}

fn emit_suite_manifest(manifest: &ValidationSuiteDispatchManifest, output: Option<&Path>) -> i32 {
    let text = match serde_json::to_string_pretty(manifest) {
        Ok(text) => text,
        Err(err) => {
            eprintln!(
                "enkai validate suite-manifest: failed to serialize manifest: {}",
                err
            );
            return 1;
        }
    };
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    eprintln!(
                        "enkai validate suite-manifest: failed to create output directory {}: {}",
                        parent.display(),
                        err
                    );
                    return 1;
                }
            }
        }
        if let Err(err) = fs::write(path, text.as_bytes()) {
            eprintln!(
                "enkai validate suite-manifest: failed to write manifest {}: {}",
                path.display(),
                err
            );
            return 1;
        }
    } else {
        println!("{}", text);
    }
    0
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

    let profile_path = workspace_root()
        .join("artifacts")
        .join("validation")
        .join("ffi_correctness_profile.json");
    let run = match run_validation_suite_with_profile(&correctness_suite, None, &profile_path) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate ffi-correctness: {}", err);
            return 1;
        }
    };
    let fallback = match run_validation_suite(
        &correctness_suite,
        Some(BTreeMap::from([(
            "ENKAI_SIM_ACCEL".to_string(),
            "0".to_string(),
        )])),
    ) {
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
    let native_result = run.result.clone();
    let fallback_result = fallback.result.clone();
    let native_profile = run.profile.clone().unwrap_or(serde_json::Value::Null);
    let native_profile_exercised = native_path_exercised(&native_profile);
    let native_equals_expected = native_result == expected;
    let fallback_equals_expected = fallback_result == expected;
    let native_vm_equal = native_result == fallback_result;
    let handle_live_count_zero = handle_check.result == json!(0);
    let passed = native_equals_expected
        && fallback_equals_expected
        && native_vm_equal
        && native_profile_exercised
        && handle_live_count_zero;
    let payload = json!({
        "schema_version": 1,
        "validation": "ffi_correctness",
        "description": correctness_suite.description,
        "version_line": manifest.version_line,
        "passed": passed,
        "reference_machine_profiles": manifest.machine_profiles,
        "target": run.suite.target,
        "elapsed_ms": run.elapsed_ms,
        "native_result": native_result,
        "vm_fallback_result": fallback_result,
        "expected": expected,
        "native_profile": native_profile,
        "output_hash": run.output_hash,
        "vm_fallback_hash": fallback.output_hash,
        "proof_checks": {
            "native_equals_expected": native_equals_expected,
            "vm_fallback_equals_expected": fallback_equals_expected,
            "native_vm_equal": native_vm_equal,
            "native_path_exercised": native_profile_exercised,
            "handle_live_count_after_run": handle_check.result,
            "handle_live_count_zero": handle_live_count_zero
        }
    });
    emit_validation_payload(&payload, &parsed)
}

fn ffi_safety_command(args: &[String]) -> i32 {
    let parsed = match parse_json_output_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai validate ffi-safety: {}", err);
            return 1;
        }
    };
    let manifest = match load_validation_manifest() {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai validate ffi-safety: {}", err);
            return 1;
        }
    };
    let suite = match suite_by_id(&manifest, "ffi_safety") {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate ffi-safety: {}", err);
            return 1;
        }
    };
    let handle_suite = match suite_by_id(&manifest, "ffi_handle_live_count") {
        Ok(suite) => suite,
        Err(err) => {
            eprintln!("enkai validate ffi-safety: {}", err);
            return 1;
        }
    };

    let run = match run_validation_suite(&suite, None) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate ffi-safety: {}", err);
            return 1;
        }
    };
    let handle_check = match run_validation_suite(&handle_suite, None) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate ffi-safety: {}", err);
            return 1;
        }
    };

    let null_error = match run_inline_validation_error(
        "native::import \"enkai_native\" ::\n    fn fault_string_null() -> String\n::\nfault_string_null()\n",
    ) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("enkai validate ffi-safety: {}", err);
            return 1;
        }
    };
    let oversized_error = match run_inline_validation_error(
        "native::import \"enkai_native\" ::\n    fn fault_buffer_oversized() -> Buffer\n::\nfault_buffer_oversized()\n",
    ) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("enkai validate ffi-safety: {}", err);
            return 1;
        }
    };
    let utf8_error = match run_inline_validation_error(
        "native::import \"enkai_native\" ::\n    fn fault_string_invalid_utf8() -> String\n::\nfault_string_invalid_utf8()\n",
    ) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("enkai validate ffi-safety: {}", err);
            return 1;
        }
    };

    let expected = suite
        .expected_result
        .clone()
        .unwrap_or(serde_json::Value::Null);
    let result_matches = run.result == expected;
    let null_code_ok = null_error.as_deref() == Some("E_FFI_RETURN_NULL");
    let oversized_code_ok = oversized_error.as_deref() == Some("E_FFI_RETURN_OVERSIZED");
    let utf8_code_ok = utf8_error.as_deref() == Some("E_FFI_UTF8");
    let handle_live_count_zero = handle_check.result == json!(0);
    let passed = result_matches
        && null_code_ok
        && oversized_code_ok
        && utf8_code_ok
        && handle_live_count_zero;
    let payload = json!({
        "schema_version": 1,
        "validation": "ffi_safety",
        "description": suite.description,
        "version_line": manifest.version_line,
        "passed": passed,
        "target": run.suite.target,
        "elapsed_ms": run.elapsed_ms,
        "result": run.result,
        "expected": expected,
        "output_hash": run.output_hash,
        "fault_errors": {
            "null_return": null_error,
            "oversized_buffer": oversized_error,
            "invalid_utf8": utf8_error
        },
        "proof_checks": {
            "result_matches_expected": result_matches,
            "null_return_error_stable": null_code_ok,
            "oversized_buffer_error_stable": oversized_code_ok,
            "invalid_utf8_error_stable": utf8_code_ok,
            "handle_live_count_after_run": handle_check.result,
            "handle_live_count_zero": handle_live_count_zero
        }
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
    let mut first_native_profile = None;
    for _ in 0..parsed.runs {
        let native = match if first_native_profile.is_none() {
            let profile_path = workspace_root()
                .join("artifacts")
                .join("validation")
                .join(format!("determinism_{}_native_profile.json", parsed.suite));
            run_validation_suite_with_profile(&suite, None, &profile_path)
        } else {
            run_validation_suite(&suite, None)
        } {
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
            first_native_profile = native.profile.clone();
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
    let native_result = first_native_result
        .clone()
        .unwrap_or(serde_json::Value::Null);
    let vm_result = first_vm_result.clone().unwrap_or(serde_json::Value::Null);
    let native_profile = first_native_profile.unwrap_or(serde_json::Value::Null);
    let native_path_used = native_path_exercised(&native_profile);
    let native_path_required = suite.require_native_path.unwrap_or(false);
    let native_path_ok = !native_path_required || native_path_used;
    let coroutine_counters_required = suite.require_coroutine_counters.unwrap_or(false);
    let coroutine_counters_ok = !coroutine_counters_required
        || (profile_counter_u64(&native_profile, "sim_coroutines_spawned") > 0
            && profile_counter_u64(&native_profile, "sim_coroutine_emits") > 0
            && profile_counter_u64(&native_profile, "sim_coroutine_next_waits") > 0);
    let expected_matches = suite
        .expected_result
        .as_ref()
        .map(|expected| &native_result == expected && &vm_result == expected)
        .unwrap_or(true);
    let simulation_audit =
        match simulation_audit_from_result(&suite, &native_result, &native_profile) {
            Ok(audit) => audit,
            Err(err) => {
                eprintln!("enkai validate determinism: {}", err);
                return 1;
            }
        };
    let replay_hash_matches = simulation_audit
        .as_ref()
        .map(|audit| {
            let snapshot_hash = audit
                .get("snapshot_hash")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let replay_hash = audit
                .get("replay_hash")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            snapshot_hash.is_empty() || replay_hash.is_empty() || snapshot_hash == replay_hash
        })
        .unwrap_or(true);
    let passed = native_unique.len() == 1
        && vm_unique.len() == 1
        && native_hash == vm_hash
        && expected_matches
        && native_path_ok
        && coroutine_counters_ok
        && replay_hash_matches;
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
        "native_result": native_result,
        "vm_fallback_result": vm_result,
        "expected_result": suite.expected_result,
        "native_profile": native_profile,
        "simulation_audit": simulation_audit,
        "proof_checks": {
            "native_unique_hash_count": native_unique.len(),
            "vm_fallback_unique_hash_count": vm_unique.len(),
            "native_vm_hash_equal": native_hash == vm_hash,
            "native_vm_result_equal": first_native_result == first_vm_result,
            "expected_matches": expected_matches,
            "require_native_path": native_path_required,
            "native_path_exercised": native_path_used,
            "native_path_requirement_passed": native_path_ok,
            "require_coroutine_counters": coroutine_counters_required,
            "coroutine_counter_requirement_passed": coroutine_counters_ok,
            "replay_hash_matches": replay_hash_matches
        }
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
    let profile = run.profile.clone().unwrap_or(serde_json::Value::Null);
    let counters = profile
        .get("counters")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let ffi_calls = counters
        .get("ffi_calls")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let marshal_copy_ops = counters
        .get("marshal_copy_ops")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let marshal_copy_ratio = if ffi_calls > 0 {
        marshal_copy_ops as f64 / ffi_calls as f64
    } else {
        0.0
    };
    let (required_native_counter_checks, required_native_counters_ok, total_kernel_native_calls) =
        evaluate_required_native_counters(&profile, suite.required_native_counters.as_ref());
    let kernel_native_dominance = total_kernel_native_calls > marshal_copy_ops;
    let kernel_native_dominance_ok = suite
        .require_kernel_native_dominance
        .map(|required| !required || kernel_native_dominance)
        .unwrap_or(true);
    let marshal_ratio_ok = suite
        .max_marshal_copy_ratio
        .map(|maximum| marshal_copy_ratio <= maximum)
        .unwrap_or(true);
    let passed = assessment.as_ref().map(|item| item.passed).unwrap_or(true)
        && required_native_counters_ok
        && kernel_native_dominance_ok
        && marshal_ratio_ok;
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
        "profile": profile,
        "proof_checks": {
            "required_native_counters": suite.required_native_counters,
            "required_native_counters_passed": required_native_counters_ok,
            "required_native_counter_checks": required_native_counter_checks,
            "kernel_native_calls_total": total_kernel_native_calls,
            "require_kernel_native_dominance": suite.require_kernel_native_dominance.unwrap_or(false),
            "require_kernel_native_dominance_passed": kernel_native_dominance_ok,
            "max_marshal_copy_ratio": suite.max_marshal_copy_ratio,
            "max_marshal_copy_ratio_passed": marshal_ratio_ok,
            "marshal_copy_ratio": marshal_copy_ratio
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
    let profile_path = workspace_root()
        .join("artifacts")
        .join("validation")
        .join("pool_safety_profile.json");
    let run = match run_validation_suite_with_profile(&suite, None, &profile_path) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("enkai validate pool-safety: {}", err);
            return 1;
        }
    };
    let fallback = match run_validation_suite(
        &suite,
        Some(BTreeMap::from([(
            "ENKAI_SIM_ACCEL".to_string(),
            "0".to_string(),
        )])),
    ) {
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
    let native_result = run.result.clone();
    let fallback_result = fallback.result.clone();
    let native_profile = run.profile.clone().unwrap_or(serde_json::Value::Null);
    let native_profile_exercised = native_path_exercised(&native_profile);
    let native_equals_expected = native_result == expected;
    let fallback_equals_expected = fallback_result == expected;
    let native_vm_equal = native_result == fallback_result;
    let high_watermark_matches_capacity = native_result
        .get("before_reset")
        .and_then(|value| value.get("high_watermark"))
        .and_then(|value| value.as_i64())
        .zip(
            native_result
                .get("capacity")
                .and_then(|value| value.as_i64()),
        )
        .map(|(high_watermark, capacity)| high_watermark == capacity)
        .unwrap_or(false);
    let handle_live_count_zero = handle_check.result == json!(0);
    let passed = native_equals_expected
        && fallback_equals_expected
        && native_vm_equal
        && native_profile_exercised
        && high_watermark_matches_capacity
        && handle_live_count_zero;
    let payload = json!({
        "schema_version": 1,
        "validation": "pool_safety",
        "description": suite.description,
        "version_line": manifest.version_line,
        "passed": passed,
        "target": run.suite.target,
        "elapsed_ms": run.elapsed_ms,
        "native_result": native_result,
        "vm_fallback_result": fallback_result,
        "expected": expected,
        "native_profile": native_profile,
        "output_hash": run.output_hash,
        "vm_fallback_hash": fallback.output_hash,
        "proof_checks": {
            "native_equals_expected": native_equals_expected,
            "vm_fallback_equals_expected": fallback_equals_expected,
            "native_vm_equal": native_vm_equal,
            "native_path_exercised": native_profile_exercised,
            "high_watermark_matches_capacity": high_watermark_matches_capacity,
            "handle_live_count_after_run": handle_check.result,
            "handle_live_count_zero": handle_live_count_zero
        }
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
    let (required_native_counter_checks, required_native_counters_ok, total_kernel_native_calls) =
        evaluate_required_native_counters(&profile, suite.required_native_counters.as_ref());
    let kernel_native_dominance = total_kernel_native_calls > marshal_copy_ops;
    let kernel_native_dominance_ok = suite
        .require_kernel_native_dominance
        .map(|required| !required || kernel_native_dominance)
        .unwrap_or(true);
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
        && required_native_counters_ok
        && kernel_native_dominance_ok
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
            "kernel_native_calls_total": total_kernel_native_calls,
            "kernel_native_dominance": kernel_native_dominance,
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
            "required_native_counters": suite.required_native_counters,
            "required_native_counters_passed": required_native_counters_ok,
            "required_native_counter_checks": required_native_counter_checks,
            "require_kernel_native_dominance": suite.require_kernel_native_dominance.unwrap_or(false),
            "require_kernel_native_dominance_passed": kernel_native_dominance_ok,
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

fn parse_manifest_flag(command_name: &str, args: &[String]) -> Result<PathBuf, String> {
    let mut manifest_path = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--manifest" => {
                idx += 1;
                if idx >= args.len() {
                    return Err(format!(
                        "enkai validate {}: --manifest requires a value",
                        command_name
                    ));
                }
                manifest_path = Some(PathBuf::from(&args[idx]));
            }
            other => {
                return Err(format!(
                    "enkai validate {}: unknown option '{}'",
                    command_name, other
                ));
            }
        }
        idx += 1;
    }
    manifest_path.ok_or_else(|| format!("enkai validate {}: --manifest is required", command_name))
}

fn load_json_manifest<T>(path: &Path) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read manifest {}: {}", path.display(), err))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse manifest {}: {}", path.display(), err))
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

fn run_inline_validation_error(source: &str) -> Result<Option<String>, String> {
    let program = with_bundled_std(|| compile_source_module(source, "<validation-inline>"))?;
    let mut vm = VM::new(false, false, false, false);
    match vm.run(&program) {
        Ok(_) => Err("expected inline validation to fail".to_string()),
        Err(err) => Ok(err.code().map(str::to_string)),
    }
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
    let program = with_bundled_std(|| load_program_from_target(&target))?;
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
        "adam0_reference_100" | "adam0_reference_1000" | "adam0_reference_10000" => Ok(json!({
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
    with_bundled_std(|| load_program_from_target(&target))
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

fn with_bundled_std<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let _guard = crate::env_guard();
    let std_override = if env::var_os("ENKAI_STD").is_none() {
        let bundled_std = workspace_root().join("std");
        bundled_std.is_dir().then_some(bundled_std)
    } else {
        None
    };
    let previous_std = if std_override.is_some() {
        env::var_os("ENKAI_STD")
    } else {
        None
    };
    if let Some(std_path) = &std_override {
        unsafe {
            env::set_var("ENKAI_STD", std_path);
        }
    }
    let result = f();
    if std_override.is_some() {
        unsafe {
            restore_env_var("ENKAI_STD", previous_std);
        }
    }
    result
}

unsafe fn restore_env_var(key: &str, value: Option<OsString>) {
    if let Some(value) = value {
        env::set_var(key, value);
    } else {
        env::remove_var(key);
    }
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
    fn build_validation_suite_manifest_captures_determinism_dispatch() {
        let manifest = build_validation_suite_manifest(
            "determinism",
            &[
                "--suite".to_string(),
                "event_queue".to_string(),
                "--runs".to_string(),
                "3".to_string(),
                "--json".to_string(),
                "--output".to_string(),
                "artifacts/validation/determinism_event_queue.json".to_string(),
            ],
        )
        .expect("manifest");
        assert_eq!(manifest.profile, "validation_suite_dispatch");
        assert!(manifest.emit_json);
        assert_eq!(
            manifest.result_output.as_deref(),
            Some("artifacts/validation/determinism_event_queue.json")
        );
        assert_eq!(
            manifest.command,
            ValidationSuiteDispatchCommand::Determinism {
                suite: "event_queue".to_string(),
                runs: 3,
            }
        );
    }

    #[test]
    fn validation_manifest_args_round_trip_perf_suite() {
        let manifest = ValidationSuiteDispatchManifest {
            schema_version: 1,
            profile: "validation_suite_dispatch".to_string(),
            emit_json: true,
            result_output: Some("artifacts/validation/perf_ffi_noop.json".to_string()),
            command: ValidationSuiteDispatchCommand::PerfBaseline {
                suite: "ffi_noop".to_string(),
            },
        };
        assert_eq!(
            validation_manifest_args(&manifest),
            vec![
                "--suite".to_string(),
                "ffi_noop".to_string(),
                "--json".to_string(),
                "--output".to_string(),
                "artifacts/validation/perf_ffi_noop.json".to_string(),
            ]
        );
    }

    #[test]
    fn parse_suite_manifest_args_separates_manifest_output() {
        let (subcommand, suite_args, manifest_output) = parse_suite_manifest_args(&[
            "ffi-correctness".to_string(),
            "--json".to_string(),
            "--output".to_string(),
            "artifacts/validation/ffi_correctness.json".to_string(),
            "--manifest-output".to_string(),
            "artifacts/validation/ffi_manifest.json".to_string(),
        ])
        .expect("parsed");
        assert_eq!(subcommand, "ffi-correctness");
        assert_eq!(
            suite_args,
            vec![
                "--json".to_string(),
                "--output".to_string(),
                "artifacts/validation/ffi_correctness.json".to_string(),
            ]
        );
        assert_eq!(
            manifest_output,
            Some(PathBuf::from("artifacts/validation/ffi_manifest.json"))
        );
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
