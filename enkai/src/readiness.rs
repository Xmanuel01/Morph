use std::collections::BTreeSet;
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

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
struct ReadinessCheckReport {
    id: String,
    description: String,
    command: Vec<String>,
    success: bool,
    exit_code: i32,
    duration_ms: u128,
}

pub fn print_readiness_usage() {
    eprintln!(
        "  enkai readiness check [--profile production|full_platform] [--json] [--output <file>] [--skip-check <id>]"
    );
}

pub fn readiness_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_readiness_usage();
        return 1;
    }
    match args[0].as_str() {
        "check" => readiness_check_command(&args[1..]),
        _ => {
            eprintln!("enkai readiness: unknown subcommand '{}'", args[0]);
            print_readiness_usage();
            1
        }
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
}
