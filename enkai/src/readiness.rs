use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct ReadinessManifest {
    schema_version: u32,
    profile: String,
    checks: Vec<ReadinessCheckSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReadinessCheckSpec {
    id: String,
    description: String,
    command: Vec<String>,
}

#[derive(Debug, Clone)]
struct ReadinessCheckArgs {
    profile: String,
    json: bool,
    output: Option<PathBuf>,
    skip_checks: Vec<String>,
}

#[derive(Debug, Clone)]
struct VerifyBlockersArgs {
    profile: String,
    report: PathBuf,
    json: bool,
    output: Option<PathBuf>,
    require_gpu_evidence: bool,
    skip_release_evidence: bool,
    allow_skipped_required_checks: Vec<String>,
    version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReleaseBlockerManifest {
    schema_version: u32,
    profile: String,
    version_line: String,
    release_blockers: ReleaseBlockerGroups,
}

#[derive(Debug, Clone, Deserialize)]
struct ReleaseBlockerGroups {
    non_hardware: ReleaseBlockerChecks,
    hardware_evidence: ReleaseBlockerArtifacts,
    release_evidence: ReleaseBlockerArtifacts,
}

#[derive(Debug, Clone, Deserialize)]
struct ReleaseBlockerChecks {
    required_checks: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReleaseBlockerArtifacts {
    required_artifacts: Vec<String>,
}

struct VerifyReleaseBlockersContext<'a> {
    workspace: &'a Path,
    manifest: &'a ReleaseBlockerManifest,
    readiness_report: &'a ReadinessReport,
    readiness_report_path: &'a Path,
    version: &'a str,
    require_gpu_evidence: bool,
    skip_release_evidence: bool,
    allow_skipped_required_checks: &'a [String],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReadinessReport {
    schema_version: u32,
    profile: String,
    language_version: String,
    cli_version: String,
    started_unix_ms: u128,
    finished_unix_ms: u128,
    all_passed: bool,
    skipped_checks: Vec<String>,
    checks: Vec<ReadinessCheckReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReadinessCheckReport {
    id: String,
    description: String,
    command: Vec<String>,
    success: bool,
    exit_code: i32,
    duration_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
struct ReleaseBlockerReport {
    schema_version: u32,
    profile: String,
    version_line: String,
    version: String,
    readiness_report: String,
    verified_unix_ms: u128,
    require_gpu_evidence: bool,
    skip_release_evidence: bool,
    all_passed: bool,
    required_checks: Vec<String>,
    missing_checks: Vec<String>,
    failed_checks: Vec<String>,
    skipped_required_checks: Vec<String>,
    waived_skipped_required_checks: Vec<String>,
    required_artifacts: Vec<String>,
    missing_artifacts: Vec<String>,
    required_gpu_artifacts: Vec<String>,
    missing_gpu_artifacts: Vec<String>,
}

pub fn print_readiness_usage() {
    eprintln!(
        "  enkai readiness check [--profile production|full_platform] [--json] [--output <file>] [--skip-check <id>]"
    );
    eprintln!(
        "  enkai readiness verify-blockers --profile full_platform --report <file> [--json] [--output <file>] [--require-gpu-evidence] [--skip-release-evidence] [--version <x.y.z>]"
    );
    eprintln!("    [--allow-skipped-required-check <id>]");
}

pub fn readiness_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_readiness_usage();
        return 1;
    }
    match args[0].as_str() {
        "check" => readiness_check_command(&args[1..]),
        "verify-blockers" => readiness_verify_blockers_command(&args[1..]),
        _ => {
            eprintln!("enkai readiness: unknown subcommand '{}'", args[0]);
            print_readiness_usage();
            1
        }
    }
}

fn readiness_verify_blockers_command(args: &[String]) -> i32 {
    let parsed = match parse_verify_blockers_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai readiness verify-blockers: {}", err);
            print_readiness_usage();
            return 1;
        }
    };
    let manifest = match load_release_blocker_manifest(&parsed.profile) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai readiness verify-blockers: {}", err);
            return 1;
        }
    };
    let readiness_report = match fs::read_to_string(&parsed.report) {
        Ok(raw) => match serde_json::from_str::<ReadinessReport>(&raw) {
            Ok(report) => report,
            Err(err) => {
                eprintln!(
                    "enkai readiness verify-blockers: failed to parse readiness report {}: {}",
                    parsed.report.display(),
                    err
                );
                return 1;
            }
        },
        Err(err) => {
            eprintln!(
                "enkai readiness verify-blockers: failed to read readiness report {}: {}",
                parsed.report.display(),
                err
            );
            return 1;
        }
    };

    let version = parsed
        .version
        .clone()
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
    let verify_context = VerifyReleaseBlockersContext {
        workspace: &workspace_root(),
        manifest: &manifest,
        readiness_report: &readiness_report,
        readiness_report_path: &parsed.report,
        version: &version,
        require_gpu_evidence: parsed.require_gpu_evidence,
        skip_release_evidence: parsed.skip_release_evidence,
        allow_skipped_required_checks: &parsed.allow_skipped_required_checks,
    };
    let report = verify_release_blockers(&verify_context);

    let json = match serde_json::to_string_pretty(&report) {
        Ok(json) => json,
        Err(err) => {
            eprintln!(
                "enkai readiness verify-blockers: failed to serialize blocker report: {}",
                err
            );
            return 1;
        }
    };

    if let Some(path) = parsed.output.as_ref() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    eprintln!(
                        "enkai readiness verify-blockers: failed to create output directory {}: {}",
                        parent.display(),
                        err
                    );
                    return 1;
                }
            }
        }
        if let Err(err) = fs::write(path, json.as_bytes()) {
            eprintln!(
                "enkai readiness verify-blockers: failed to write report {}: {}",
                path.display(),
                err
            );
            return 1;
        }
        println!("[readiness] blocker report written: {}", path.display());
    }

    if parsed.json {
        println!("{}", json);
    }

    if report.all_passed {
        println!(
            "[readiness] blocker verification passed for profile '{}'",
            parsed.profile
        );
        0
    } else {
        if !report.missing_checks.is_empty() {
            eprintln!(
                "[readiness] missing required checks: {}",
                report.missing_checks.join(", ")
            );
        }
        if !report.failed_checks.is_empty() {
            eprintln!(
                "[readiness] failed required checks: {}",
                report.failed_checks.join(", ")
            );
        }
        if !report.skipped_required_checks.is_empty() {
            eprintln!(
                "[readiness] skipped required checks: {}",
                report.skipped_required_checks.join(", ")
            );
        }
        if !report.missing_artifacts.is_empty() {
            eprintln!(
                "[readiness] missing required artifacts: {}",
                report.missing_artifacts.join(", ")
            );
        }
        if !report.missing_gpu_artifacts.is_empty() {
            eprintln!(
                "[readiness] missing required GPU artifacts: {}",
                report.missing_gpu_artifacts.join(", ")
            );
        }
        eprintln!(
            "[readiness] blocker verification failed for profile '{}'",
            parsed.profile
        );
        1
    }
}

fn readiness_check_command(args: &[String]) -> i32 {
    let parsed = match parse_check_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai readiness check: {}", err);
            print_readiness_usage();
            return 1;
        }
    };
    let manifest = match load_manifest(&parsed.profile) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("enkai readiness check: {}", err);
            return 1;
        }
    };
    let (checks, skipped_checks) = match filter_manifest_checks(&manifest, &parsed.skip_checks) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai readiness check: {}", err);
            return 1;
        }
    };

    let started_unix_ms = unix_millis();
    let workspace = workspace_root();
    let mut check_reports = Vec::with_capacity(checks.len());

    for check in &checks {
        println!(
            "[readiness] [{}] {}",
            check.id.trim(),
            check.description.trim()
        );
        let report = run_check(&workspace, check);
        let status = if report.success { "PASS" } else { "FAIL" };
        println!(
            "[readiness] {} {} ({} ms, exit={})",
            status, report.id, report.duration_ms, report.exit_code
        );
        check_reports.push(report);
    }

    let finished_unix_ms = unix_millis();
    let all_passed = check_reports.iter().all(|check| check.success);
    let report = ReadinessReport {
        schema_version: manifest.schema_version,
        profile: manifest.profile,
        language_version: env!("ENKAI_LANG_VERSION").to_string(),
        cli_version: env!("CARGO_PKG_VERSION").to_string(),
        started_unix_ms,
        finished_unix_ms,
        all_passed,
        skipped_checks,
        checks: check_reports,
    };

    let json = match serde_json::to_string_pretty(&report) {
        Ok(json) => json,
        Err(err) => {
            eprintln!(
                "enkai readiness check: failed to serialize readiness report: {}",
                err
            );
            return 1;
        }
    };

    if let Some(path) = parsed.output.as_ref() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    eprintln!(
                        "enkai readiness check: failed to create output directory {}: {}",
                        parent.display(),
                        err
                    );
                    return 1;
                }
            }
        }
        if let Err(err) = fs::write(path, json.as_bytes()) {
            eprintln!(
                "enkai readiness check: failed to write report {}: {}",
                path.display(),
                err
            );
            return 1;
        }
        println!("[readiness] report written: {}", path.display());
    }

    if parsed.json {
        println!("{}", json);
    }

    if all_passed {
        println!("[readiness] profile '{}' passed", parsed.profile);
        0
    } else {
        eprintln!("[readiness] profile '{}' has failed checks", parsed.profile);
        1
    }
}

fn parse_check_args(args: &[String]) -> Result<ReadinessCheckArgs, String> {
    let mut parsed = ReadinessCheckArgs {
        profile: "production".to_string(),
        json: false,
        output: None,
        skip_checks: Vec::new(),
    };
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--profile" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--profile requires a value".to_string());
                }
                parsed.profile = args[idx].trim().to_string();
                if parsed.profile.is_empty() {
                    return Err("--profile cannot be empty".to_string());
                }
            }
            "--json" => {
                parsed.json = true;
            }
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                parsed.output = Some(PathBuf::from(args[idx].trim()));
            }
            "--skip-check" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--skip-check requires a value".to_string());
                }
                let value = args[idx].trim();
                if value.is_empty() {
                    return Err("--skip-check cannot be empty".to_string());
                }
                parsed.skip_checks.push(value.to_string());
            }
            unknown => {
                return Err(format!("unknown option '{}'", unknown));
            }
        }
        idx += 1;
    }
    Ok(parsed)
}

fn parse_verify_blockers_args(args: &[String]) -> Result<VerifyBlockersArgs, String> {
    let mut parsed = VerifyBlockersArgs {
        profile: "full_platform".to_string(),
        report: PathBuf::new(),
        json: false,
        output: None,
        require_gpu_evidence: false,
        skip_release_evidence: false,
        allow_skipped_required_checks: Vec::new(),
        version: None,
    };
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--profile" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--profile requires a value".to_string());
                }
                parsed.profile = args[idx].trim().to_string();
                if parsed.profile.is_empty() {
                    return Err("--profile cannot be empty".to_string());
                }
            }
            "--report" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--report requires a value".to_string());
                }
                parsed.report = PathBuf::from(args[idx].trim());
            }
            "--json" => {
                parsed.json = true;
            }
            "--output" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--output requires a value".to_string());
                }
                parsed.output = Some(PathBuf::from(args[idx].trim()));
            }
            "--require-gpu-evidence" => {
                parsed.require_gpu_evidence = true;
            }
            "--skip-release-evidence" => {
                parsed.skip_release_evidence = true;
            }
            "--allow-skipped-required-check" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--allow-skipped-required-check requires a value".to_string());
                }
                let value = args[idx].trim();
                if value.is_empty() {
                    return Err("--allow-skipped-required-check cannot be empty".to_string());
                }
                parsed.allow_skipped_required_checks.push(value.to_string());
            }
            "--version" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--version requires a value".to_string());
                }
                let value = args[idx].trim();
                if value.is_empty() {
                    return Err("--version cannot be empty".to_string());
                }
                parsed.version = Some(value.to_string());
            }
            unknown => {
                return Err(format!("unknown option '{}'", unknown));
            }
        }
        idx += 1;
    }
    if parsed.report.as_os_str().is_empty() {
        return Err("--report is required".to_string());
    }
    Ok(parsed)
}

fn load_manifest(profile: &str) -> Result<ReadinessManifest, String> {
    let raw = match profile {
        "production" => include_str!("../contracts/readiness_production_v2_3_0.json"),
        "full_platform" => include_str!("../contracts/readiness_full_platform_v2_5_0.json"),
        _ => {
            return Err(format!(
                "unsupported profile '{}'; expected 'production' or 'full_platform'",
                profile
            ));
        }
    };
    let manifest: ReadinessManifest = serde_json::from_str(raw)
        .map_err(|err| format!("failed to parse readiness manifest: {}", err))?;
    Ok(manifest)
}

fn load_release_blocker_manifest(profile: &str) -> Result<ReleaseBlockerManifest, String> {
    let raw = match profile {
        "full_platform" => include_str!("../contracts/full_platform_release_blockers_v2_5_0.json"),
        _ => {
            return Err(format!(
                "unsupported blocker profile '{}'; expected 'full_platform'",
                profile
            ));
        }
    };
    let manifest: ReleaseBlockerManifest = serde_json::from_str(raw)
        .map_err(|err| format!("failed to parse blocker manifest: {}", err))?;
    Ok(manifest)
}

fn filter_manifest_checks(
    manifest: &ReadinessManifest,
    skip_checks: &[String],
) -> Result<(Vec<ReadinessCheckSpec>, Vec<String>), String> {
    let requested_skips: BTreeSet<String> = skip_checks
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
        .collect();
    if requested_skips.is_empty() {
        return Ok((manifest.checks.clone(), Vec::new()));
    }

    let known_ids: BTreeSet<String> = manifest
        .checks
        .iter()
        .map(|check| check.id.clone())
        .collect();
    let unknown: Vec<String> = requested_skips.difference(&known_ids).cloned().collect();
    if !unknown.is_empty() {
        return Err(format!(
            "unknown --skip-check id(s): {}",
            unknown.join(", ")
        ));
    }

    let mut retained = Vec::with_capacity(manifest.checks.len());
    let mut skipped = Vec::new();
    for check in &manifest.checks {
        if requested_skips.contains(&check.id) {
            skipped.push(check.id.clone());
        } else {
            retained.push(check.clone());
        }
    }
    Ok((retained, skipped))
}

fn verify_release_blockers(context: &VerifyReleaseBlockersContext<'_>) -> ReleaseBlockerReport {
    let mut checks_by_id = BTreeMap::new();
    for check in &context.readiness_report.checks {
        checks_by_id.insert(check.id.clone(), check);
    }
    let skipped: BTreeSet<&str> = context
        .readiness_report
        .skipped_checks
        .iter()
        .map(|value| value.as_str())
        .collect();

    let mut missing_checks = Vec::new();
    let mut failed_checks = Vec::new();
    let mut skipped_required_checks = Vec::new();
    let mut waived_skipped_required_checks = Vec::new();
    let allowed_skips: BTreeSet<&str> = context
        .allow_skipped_required_checks
        .iter()
        .map(|value| value.as_str())
        .collect();
    for check_id in &context
        .manifest
        .release_blockers
        .non_hardware
        .required_checks
    {
        if skipped.contains(check_id.as_str()) {
            if allowed_skips.contains(check_id.as_str()) {
                waived_skipped_required_checks.push(check_id.clone());
            } else {
                skipped_required_checks.push(check_id.clone());
            }
            continue;
        }
        match checks_by_id.get(check_id) {
            Some(check) if check.success => {}
            Some(_) => failed_checks.push(check_id.clone()),
            None => missing_checks.push(check_id.clone()),
        }
    }

    let required_artifacts = if context.skip_release_evidence {
        Vec::new()
    } else {
        context
            .manifest
            .release_blockers
            .release_evidence
            .required_artifacts
            .iter()
            .map(|path| expand_release_artifact_placeholder(path, context.version))
            .collect::<Vec<_>>()
    };
    let missing_artifacts = find_missing_artifacts(context.workspace, &required_artifacts);

    let required_gpu_artifacts = if context.require_gpu_evidence {
        context
            .manifest
            .release_blockers
            .hardware_evidence
            .required_artifacts
            .iter()
            .map(|path| expand_release_artifact_placeholder(path, context.version))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let missing_gpu_artifacts = find_missing_artifacts(context.workspace, &required_gpu_artifacts);

    let all_passed = context.readiness_report.profile == context.manifest.profile
        && missing_checks.is_empty()
        && failed_checks.is_empty()
        && skipped_required_checks.is_empty()
        && missing_artifacts.is_empty()
        && missing_gpu_artifacts.is_empty();

    ReleaseBlockerReport {
        schema_version: context.manifest.schema_version,
        profile: context.manifest.profile.clone(),
        version_line: context.manifest.version_line.clone(),
        version: context.version.to_string(),
        readiness_report: context.readiness_report_path.display().to_string(),
        verified_unix_ms: unix_millis(),
        require_gpu_evidence: context.require_gpu_evidence,
        skip_release_evidence: context.skip_release_evidence,
        all_passed,
        required_checks: context
            .manifest
            .release_blockers
            .non_hardware
            .required_checks
            .clone(),
        missing_checks,
        failed_checks,
        skipped_required_checks,
        waived_skipped_required_checks,
        required_artifacts,
        missing_artifacts,
        required_gpu_artifacts,
        missing_gpu_artifacts,
    }
}

fn expand_release_artifact_placeholder(path: &str, version: &str) -> String {
    path.replace("<version>", version)
}

fn find_missing_artifacts(workspace: &Path, required_artifacts: &[String]) -> Vec<String> {
    required_artifacts
        .iter()
        .filter(|path| !workspace.join(path.as_str()).is_file())
        .cloned()
        .collect()
}

fn run_check(workspace: &Path, check: &ReadinessCheckSpec) -> ReadinessCheckReport {
    if check.command.is_empty() {
        return ReadinessCheckReport {
            id: check.id.clone(),
            description: check.description.clone(),
            command: check.command.clone(),
            success: false,
            exit_code: 1,
            duration_ms: 0,
        };
    }

    let resolved = resolve_command_tokens(&check.command);
    let mut command = Command::new(&resolved[0]);
    if resolved.len() > 1 {
        command.args(&resolved[1..]);
    }
    command.current_dir(workspace);

    let started = Instant::now();
    let status = command.status();
    let duration_ms = started.elapsed().as_millis();

    match status {
        Ok(status) => ReadinessCheckReport {
            id: check.id.clone(),
            description: check.description.clone(),
            command: check.command.clone(),
            success: status.success(),
            exit_code: status.code().unwrap_or(1),
            duration_ms,
        },
        Err(err) => ReadinessCheckReport {
            id: check.id.clone(),
            description: check.description.clone(),
            command: vec![
                format!("{} (launch failed)", resolved.join(" ")),
                err.to_string(),
            ],
            success: false,
            exit_code: 1,
            duration_ms,
        },
    }
}

fn resolve_command_tokens(command: &[String]) -> Vec<String> {
    if command.is_empty() {
        return Vec::new();
    }
    let current_exe = env::current_exe()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|_| "enkai".to_string());
    let resolved_enkai_bin = resolve_enkai_bin().unwrap_or_else(|| current_exe.clone());
    let python_command = resolve_python_command();
    let mut out = Vec::with_capacity(command.len());
    for (index, token) in command.iter().enumerate() {
        if token == "${PYTHON}" {
            match python_command.as_ref() {
                Some(cmd) => out.extend(cmd.iter().cloned()),
                None => out.push("python".to_string()),
            }
            continue;
        }
        let mut resolved = token.replace("${ENKAI_BIN}", &resolved_enkai_bin);
        if index == 0 && resolved == "enkai" {
            resolved = resolved_enkai_bin.clone();
        }
        out.push(resolved);
    }
    out
}

fn resolve_enkai_bin() -> Option<String> {
    if let Ok(path) = env::var("ENKAI_READINESS_ENKAI_BIN") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    let release = if cfg!(windows) {
        workspace_root()
            .join("target")
            .join("release")
            .join("enkai.exe")
    } else {
        workspace_root()
            .join("target")
            .join("release")
            .join("enkai")
    };
    if release.is_file() {
        return Some(release.to_string_lossy().to_string());
    }

    env::current_exe()
        .ok()
        .map(|path| path.to_string_lossy().to_string())
}

fn resolve_python_command() -> Option<Vec<String>> {
    let candidates = if cfg!(windows) {
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
    };
    candidates
        .into_iter()
        .find(|candidate| command_available(candidate))
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

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."))
}

fn unix_millis() -> u128 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis(),
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{}_{}", prefix, unique));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    #[test]
    fn parse_check_args_defaults() {
        let parsed = parse_check_args(&[]).expect("parse");
        assert_eq!(parsed.profile, "production");
        assert!(!parsed.json);
        assert!(parsed.output.is_none());
        assert!(parsed.skip_checks.is_empty());
    }

    #[test]
    fn parse_check_args_overrides() {
        let parsed = parse_check_args(&[
            "--profile".to_string(),
            "full_platform".to_string(),
            "--json".to_string(),
            "--output".to_string(),
            "artifacts/readiness.json".to_string(),
            "--skip-check".to_string(),
            "benchmark-target".to_string(),
            "--skip-check".to_string(),
            "benchmark-build-release".to_string(),
        ])
        .expect("parse");
        assert_eq!(parsed.profile, "full_platform");
        assert!(parsed.json);
        assert_eq!(
            parsed.output,
            Some(PathBuf::from("artifacts/readiness.json"))
        );
        assert_eq!(
            parsed.skip_checks,
            vec![
                "benchmark-target".to_string(),
                "benchmark-build-release".to_string()
            ]
        );
    }

    #[test]
    fn parse_check_args_rejects_unknown_option() {
        let err = parse_check_args(&["--nope".to_string()]).expect_err("unknown");
        assert!(err.contains("unknown option"));
    }

    #[test]
    fn parse_verify_blockers_args_defaults() {
        let parsed = parse_verify_blockers_args(&[
            "--report".to_string(),
            "artifacts/readiness/full_platform.json".to_string(),
            "--allow-skipped-required-check".to_string(),
            "selfhost-mainline".to_string(),
        ])
        .expect("parse");
        assert_eq!(parsed.profile, "full_platform");
        assert_eq!(
            parsed.report,
            PathBuf::from("artifacts/readiness/full_platform.json")
        );
        assert!(!parsed.json);
        assert!(parsed.output.is_none());
        assert!(!parsed.require_gpu_evidence);
        assert!(!parsed.skip_release_evidence);
        assert_eq!(
            parsed.allow_skipped_required_checks,
            vec!["selfhost-mainline".to_string()]
        );
        assert!(parsed.version.is_none());
    }

    #[test]
    fn parse_verify_blockers_args_rejects_missing_report() {
        let err = parse_verify_blockers_args(&["--json".to_string()]).expect_err("missing");
        assert!(err.contains("--report is required"));
    }

    #[test]
    fn load_manifest_production_profile() {
        let manifest = load_manifest("production").expect("manifest");
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.profile, "production");
        assert!(!manifest.checks.is_empty());
    }

    #[test]
    fn load_manifest_full_platform_profile() {
        let manifest = load_manifest("full_platform").expect("manifest");
        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.profile, "full_platform");
        assert!(!manifest.checks.is_empty());
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "simulation-smoke"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "simulation-evidence-verify"));
        assert!(manifest.checks.iter().any(|check| check.id == "grpc-smoke"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "grpc-evidence-verify"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "simulation-native-smoke"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "simulation-native-evidence-verify"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "simulation-stdlib-smoke"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "simulation-stdlib-evidence-verify"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "snn-agent-kernel-smoke"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "snn-agent-kernel-evidence-verify"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "adam0-reference-suite"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "adam0-reference-suite-verify"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "deploy-backend-validate"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "deploy-fullstack-validate"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "cluster-scale-smoke"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "cluster-scale-evidence-verify"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "registry-degraded-smoke"));
        assert!(manifest
            .checks
            .iter()
            .any(|check| check.id == "registry-degraded-evidence-verify"));
    }

    #[test]
    fn filter_manifest_checks_skips_known_ids_in_order() {
        let manifest = ReadinessManifest {
            schema_version: 1,
            profile: "test".to_string(),
            checks: vec![
                ReadinessCheckSpec {
                    id: "fmt".to_string(),
                    description: "fmt".to_string(),
                    command: vec!["cargo".to_string()],
                },
                ReadinessCheckSpec {
                    id: "test".to_string(),
                    description: "test".to_string(),
                    command: vec!["cargo".to_string()],
                },
                ReadinessCheckSpec {
                    id: "bench".to_string(),
                    description: "bench".to_string(),
                    command: vec!["cargo".to_string()],
                },
            ],
        };
        let (checks, skipped) = filter_manifest_checks(
            &manifest,
            &["bench".to_string(), "fmt".to_string(), "fmt".to_string()],
        )
        .expect("filter");
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].id, "test");
        assert_eq!(skipped, vec!["fmt".to_string(), "bench".to_string()]);
    }

    #[test]
    fn filter_manifest_checks_rejects_unknown_id() {
        let manifest = load_manifest("production").expect("manifest");
        let err = filter_manifest_checks(&manifest, &["missing".to_string()]).expect_err("unknown");
        assert!(err.contains("missing"));
    }

    #[test]
    fn verify_release_blockers_detects_failed_checks_and_missing_artifacts() {
        let workspace = temp_dir("enkai_readiness_blockers");
        fs::create_dir_all(workspace.join("artifacts").join("readiness")).expect("mkdir");
        let report_path = workspace
            .join("artifacts")
            .join("readiness")
            .join("full_platform.json");
        fs::write(&report_path, "{}").expect("touch report");

        let manifest = load_release_blocker_manifest("full_platform").expect("manifest");
        let report = ReadinessReport {
            schema_version: 1,
            profile: "full_platform".to_string(),
            language_version: "2.5.9".to_string(),
            cli_version: "2.5.9".to_string(),
            started_unix_ms: 0,
            finished_unix_ms: 1,
            all_passed: false,
            skipped_checks: vec!["deploy-fullstack-validate".to_string()],
            checks: vec![
                ReadinessCheckReport {
                    id: "fmt".to_string(),
                    description: "fmt".to_string(),
                    command: vec!["cargo".to_string()],
                    success: true,
                    exit_code: 0,
                    duration_ms: 1,
                },
                ReadinessCheckReport {
                    id: "clippy".to_string(),
                    description: "clippy".to_string(),
                    command: vec!["cargo".to_string()],
                    success: false,
                    exit_code: 1,
                    duration_ms: 1,
                },
            ],
        };

        let context = VerifyReleaseBlockersContext {
            workspace: &workspace,
            manifest: &manifest,
            readiness_report: &report,
            readiness_report_path: &report_path,
            version: "2.5.9",
            require_gpu_evidence: false,
            skip_release_evidence: false,
            allow_skipped_required_checks: &[],
        };
        let result = verify_release_blockers(&context);

        assert!(!result.all_passed);
        assert!(result.failed_checks.contains(&"clippy".to_string()));
        assert!(result
            .skipped_required_checks
            .contains(&"deploy-fullstack-validate".to_string()));
        assert!(result.missing_checks.contains(&"test".to_string()));
        assert!(result
            .missing_artifacts
            .contains(&"artifacts/selfhost/litec_mainline_ci_report.json".to_string()));

        let _ = fs::remove_dir_all(workspace);
    }

    #[test]
    fn verify_release_blockers_passes_with_required_artifacts() {
        let workspace = temp_dir("enkai_readiness_blockers_ok");
        let manifest = load_release_blocker_manifest("full_platform").expect("manifest");
        let mut required_paths = manifest
            .release_blockers
            .release_evidence
            .required_artifacts
            .iter()
            .map(|path| expand_release_artifact_placeholder(path, "2.9.4"))
            .collect::<Vec<_>>();
        required_paths.push("artifacts/readiness/full_platform.json".to_string());
        required_paths.sort();
        required_paths.dedup();
        for path in &required_paths {
            let full = workspace.join(path);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).expect("mkdir");
            }
            fs::write(full, "{}").expect("write");
        }
        let checks = manifest
            .release_blockers
            .non_hardware
            .required_checks
            .iter()
            .map(|id| ReadinessCheckReport {
                id: id.clone(),
                description: id.clone(),
                command: vec!["ok".to_string()],
                success: true,
                exit_code: 0,
                duration_ms: 1,
            })
            .collect::<Vec<_>>();
        let report = ReadinessReport {
            schema_version: 1,
            profile: "full_platform".to_string(),
            language_version: "2.9.4".to_string(),
            cli_version: "2.9.4".to_string(),
            started_unix_ms: 0,
            finished_unix_ms: 1,
            all_passed: true,
            skipped_checks: Vec::new(),
            checks,
        };

        let report_path = workspace.join("artifacts/readiness/full_platform.json");
        let context = VerifyReleaseBlockersContext {
            workspace: &workspace,
            manifest: &manifest,
            readiness_report: &report,
            readiness_report_path: &report_path,
            version: "2.9.4",
            require_gpu_evidence: false,
            skip_release_evidence: false,
            allow_skipped_required_checks: &[],
        };
        let result = verify_release_blockers(&context);

        assert!(result.all_passed);
        assert!(result.missing_checks.is_empty());
        assert!(result.failed_checks.is_empty());
        assert!(result.missing_artifacts.is_empty());

        let _ = fs::remove_dir_all(workspace);
    }

    #[test]
    fn verify_release_blockers_allows_explicitly_waived_skipped_checks() {
        let workspace = temp_dir("enkai_readiness_blockers_waived");
        let manifest = load_release_blocker_manifest("full_platform").expect("manifest");
        let mut required_paths = manifest
            .release_blockers
            .release_evidence
            .required_artifacts
            .iter()
            .map(|path| expand_release_artifact_placeholder(path, "2.9.4"))
            .collect::<Vec<_>>();
        required_paths.push("artifacts/readiness/full_platform.json".to_string());
        required_paths.sort();
        required_paths.dedup();
        for path in &required_paths {
            let full = workspace.join(path);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).expect("mkdir");
            }
            fs::write(full, "{}").expect("write");
        }
        let checks = manifest
            .release_blockers
            .non_hardware
            .required_checks
            .iter()
            .filter(|id| id.as_str() != "selfhost-mainline")
            .map(|id| ReadinessCheckReport {
                id: id.clone(),
                description: id.clone(),
                command: vec!["ok".to_string()],
                success: true,
                exit_code: 0,
                duration_ms: 1,
            })
            .collect::<Vec<_>>();
        let report = ReadinessReport {
            schema_version: 1,
            profile: "full_platform".to_string(),
            language_version: "2.9.4".to_string(),
            cli_version: "2.9.4".to_string(),
            started_unix_ms: 0,
            finished_unix_ms: 1,
            all_passed: true,
            skipped_checks: vec!["selfhost-mainline".to_string()],
            checks,
        };

        let report_path = workspace.join("artifacts/readiness/full_platform.json");
        let allowed = vec!["selfhost-mainline".to_string()];
        let context = VerifyReleaseBlockersContext {
            workspace: &workspace,
            manifest: &manifest,
            readiness_report: &report,
            readiness_report_path: &report_path,
            version: "2.9.4",
            require_gpu_evidence: false,
            skip_release_evidence: false,
            allow_skipped_required_checks: &allowed,
        };
        let result = verify_release_blockers(&context);

        assert!(result.all_passed);
        assert!(result.skipped_required_checks.is_empty());
        assert_eq!(
            result.waived_skipped_required_checks,
            vec!["selfhost-mainline".to_string()]
        );

        let _ = fs::remove_dir_all(workspace);
    }
}
