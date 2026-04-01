use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

pub fn print_deploy_usage() {
    eprintln!(
        "  enkai deploy validate <project_dir> --profile <backend|fullstack> --strict [--json] [--output <file>]"
    );
}

pub fn deploy_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_deploy_usage();
        return 1;
    }
    match args[0].as_str() {
        "validate" => deploy_validate_command(&args[1..]),
        other => {
            eprintln!("enkai deploy: unknown subcommand '{}'", other);
            print_deploy_usage();
            1
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeployProfile {
    Backend,
    Fullstack,
}

#[derive(Debug, Clone)]
struct DeployValidateArgs {
    project_dir: PathBuf,
    profile: DeployProfile,
    strict: bool,
    json: bool,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationIssue {
    code: &'static str,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationReport {
    schema_version: u32,
    profile: &'static str,
    project_dir: String,
    strict: bool,
    success: bool,
    issue_count: usize,
    issues: Vec<ValidationIssue>,
}

fn deploy_validate_command(args: &[String]) -> i32 {
    let parsed = match parse_validate_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai deploy validate: {}", err);
            print_deploy_usage();
            return 1;
        }
    };
    if !parsed.strict {
        eprintln!("enkai deploy validate: --strict is required for production validation");
        return 1;
    }

    let mut issues = Vec::new();
    match parsed.profile {
        DeployProfile::Backend => validate_backend_project(&parsed.project_dir, &mut issues),
        DeployProfile::Fullstack => validate_fullstack_project(&parsed.project_dir, &mut issues),
    }
    let success = issues.is_empty();
    let report = ValidationReport {
        schema_version: 1,
        profile: profile_name(parsed.profile),
        project_dir: parsed.project_dir.display().to_string(),
        strict: parsed.strict,
        success,
        issue_count: issues.len(),
        issues,
    };

    if parsed.json {
        match write_validation_report(&report, parsed.output.as_deref()) {
            Ok(()) => {}
            Err(err) => {
                eprintln!("enkai deploy validate: {}", err);
                return 1;
            }
        }
    }

    if success {
        println!(
            "[deploy-validate] ok profile={} project={}",
            profile_name(parsed.profile),
            parsed.project_dir.display()
        );
        0
    } else {
        eprintln!(
            "[deploy-validate] failed profile={} project={}",
            profile_name(parsed.profile),
            parsed.project_dir.display()
        );
        for issue in &report.issues {
            eprintln!("[deploy-validate] {}: {}", issue.code, issue.message);
        }
        1
    }
}

fn parse_validate_args(args: &[String]) -> Result<DeployValidateArgs, String> {
    if args.is_empty() {
        return Err("missing project directory".to_string());
    }
    let mut project_dir: Option<PathBuf> = None;
    let mut profile: Option<DeployProfile> = None;
    let mut strict = false;
    let mut json = false;
    let mut output: Option<PathBuf> = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--profile" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--profile requires a value".to_string())?;
                profile = Some(parse_profile(value)?);
            }
            "--output" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--output requires a value".to_string())?;
                output = Some(PathBuf::from(value));
            }
            "--strict" => strict = true,
            "--json" => json = true,
            other if other.starts_with("--") => {
                return Err(format!("unknown option '{}'", other));
            }
            path => {
                if project_dir.is_some() {
                    return Err(format!("unexpected argument '{}'", path));
                }
                project_dir = Some(PathBuf::from(path));
            }
        }
        index += 1;
    }
    let project_dir = project_dir.ok_or_else(|| "missing project directory".to_string())?;
    if !project_dir.is_dir() {
        return Err(format!(
            "project directory not found: {}",
            project_dir.display()
        ));
    }
    let profile = profile.ok_or_else(|| "--profile is required".to_string())?;
    Ok(DeployValidateArgs {
        project_dir,
        profile,
        strict,
        json,
        output,
    })
}

fn parse_profile(raw: &str) -> Result<DeployProfile, String> {
    match raw.trim() {
        "backend" => Ok(DeployProfile::Backend),
        "fullstack" => Ok(DeployProfile::Fullstack),
        other => Err(format!(
            "invalid --profile '{}'; expected backend|fullstack",
            other
        )),
    }
}

fn profile_name(profile: DeployProfile) -> &'static str {
    match profile {
        DeployProfile::Backend => "backend",
        DeployProfile::Fullstack => "fullstack",
    }
}

fn validate_backend_project(root: &Path, issues: &mut Vec<ValidationIssue>) {
    let required_files = [
        (
            "missing_backend_manifest",
            root.join("enkai.toml"),
            "required backend manifest missing",
        ),
        (
            "missing_backend_entrypoint",
            root.join("src").join("main.enk"),
            "required backend entrypoint missing",
        ),
        (
            "missing_backend_contract_snapshot",
            root.join("contracts").join("backend_api.snapshot.json"),
            "backend API contract snapshot missing",
        ),
        (
            "missing_conversation_schema",
            root.join("contracts")
                .join("conversation_state.schema.json"),
            "conversation schema snapshot missing",
        ),
        (
            "missing_deploy_env_snapshot",
            root.join("contracts").join("deploy_env.snapshot.json"),
            "deploy env snapshot missing",
        ),
        (
            "missing_env_contract_validator",
            root.join("scripts").join("validate_env_contract.py"),
            "deploy env validator script missing",
        ),
        (
            "missing_env_example",
            root.join(".env.example"),
            ".env.example missing",
        ),
        (
            "missing_migration_001",
            root.join("migrations").join("001_conversation_state.sql"),
            "migration 001 missing",
        ),
        (
            "missing_migration_002",
            root.join("migrations")
                .join("002_conversation_state_index.sql"),
            "migration 002 missing",
        ),
        (
            "missing_backend_dockerfile",
            root.join("deploy").join("docker").join("Dockerfile"),
            "backend deploy Dockerfile missing",
        ),
        (
            "missing_backend_compose",
            root.join("deploy").join("docker-compose.yml"),
            "backend deploy docker-compose profile missing",
        ),
        (
            "missing_backend_systemd_unit",
            root.join("deploy")
                .join("systemd")
                .join("enkai-backend.service"),
            "backend deploy systemd unit missing",
        ),
    ];
    for (code, path, msg) in required_files {
        if !path.is_file() {
            issues.push(ValidationIssue {
                code,
                message: format!("{}: {}", msg, path.display()),
            });
        }
    }

    let deploy_snapshot_path = root.join("contracts").join("deploy_env.snapshot.json");
    let env_example_path = root.join(".env.example");
    if deploy_snapshot_path.is_file() && env_example_path.is_file() {
        validate_env_snapshot_alignment(&deploy_snapshot_path, &env_example_path, issues);
    }
    validate_backend_contract_assets(root, issues);

    let validator = root.join("scripts").join("validate_env_contract.py");
    if validator.is_file() && env_example_path.is_file() {
        run_env_validator(root, &validator, &env_example_path, issues);
    }
}

fn validate_fullstack_project(root: &Path, issues: &mut Vec<ValidationIssue>) {
    let backend_dir = root.join("backend");
    let frontend_dir = root.join("frontend");
    if !backend_dir.is_dir() {
        issues.push(ValidationIssue {
            code: "missing_backend_dir",
            message: format!(
                "fullstack backend directory missing: {}",
                backend_dir.display()
            ),
        });
    } else {
        validate_backend_project(&backend_dir, issues);
    }
    if !frontend_dir.is_dir() {
        issues.push(ValidationIssue {
            code: "missing_frontend_dir",
            message: format!(
                "fullstack frontend directory missing: {}",
                frontend_dir.display()
            ),
        });
        return;
    }
    for (code, path, msg) in [
        (
            "missing_frontend_package",
            frontend_dir.join("package.json"),
            "frontend package.json missing",
        ),
        (
            "missing_frontend_sdk_snapshot",
            frontend_dir.join("contracts").join("sdk_api.snapshot.json"),
            "frontend SDK snapshot missing",
        ),
        (
            "missing_frontend_sdk",
            frontend_dir.join("src").join("sdk").join("enkaiClient.ts"),
            "frontend SDK source missing",
        ),
        (
            "missing_frontend_env_example",
            frontend_dir.join(".env.example"),
            "frontend .env.example missing",
        ),
    ] {
        if !path.is_file() {
            issues.push(ValidationIssue {
                code,
                message: format!("{}: {}", msg, path.display()),
            });
        }
    }
    let fullstack_compose = root.join("deploy").join("docker-compose.yml");
    if !fullstack_compose.is_file() {
        issues.push(ValidationIssue {
            code: "missing_fullstack_compose",
            message: format!(
                "fullstack deploy docker-compose profile missing: {}",
                fullstack_compose.display()
            ),
        });
    }

    let backend_snapshot = backend_dir
        .join("contracts")
        .join("backend_api.snapshot.json");
    let frontend_snapshot = frontend_dir.join("contracts").join("sdk_api.snapshot.json");
    if backend_snapshot.is_file() && frontend_snapshot.is_file() {
        validate_api_version_alignment(&backend_snapshot, &frontend_snapshot, issues);
    }
    validate_fullstack_frontend_assets(&frontend_dir, issues);
}

fn validate_backend_contract_assets(root: &Path, issues: &mut Vec<ValidationIssue>) {
    validate_migration_contract(root, issues);
    validate_backend_deploy_assets(root, issues);
}

fn validate_migration_contract(root: &Path, issues: &mut Vec<ValidationIssue>) {
    let migrations_dir = root.join("migrations");
    let entries = match fs::read_dir(&migrations_dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    let mut sql_files = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let is_sql = path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.eq_ignore_ascii_case("sql"))
            .unwrap_or(false);
        if is_sql {
            sql_files.push(path);
        }
    }
    sql_files.sort();
    if sql_files.is_empty() {
        return;
    }

    let mut expected = 1usize;
    for path in &sql_files {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        let prefix = name.split('_').next().unwrap_or_default();
        match prefix.parse::<usize>() {
            Ok(number) if number == expected => expected += 1,
            Ok(number) => issues.push(ValidationIssue {
                code: "migration_sequence_gap",
                message: format!(
                    "expected migration prefix {:03}, found {:03} in {}",
                    expected,
                    number,
                    path.display()
                ),
            }),
            Err(_) => issues.push(ValidationIssue {
                code: "invalid_migration_filename",
                message: format!(
                    "migration filename must start with zero-padded numeric prefix: {}",
                    path.display()
                ),
            }),
        }
    }

    validate_required_migration_content(
        &migrations_dir.join("001_conversation_state.sql"),
        &[
            "CREATE TABLE IF NOT EXISTS schema_migrations",
            "CREATE TABLE IF NOT EXISTS conversation_events",
        ],
        "migration_001_contract_mismatch",
        issues,
    );
    validate_required_migration_content(
        &migrations_dir.join("002_conversation_state_index.sql"),
        &[
            "CREATE INDEX IF NOT EXISTS idx_conversation_events_updated_ms",
            "conversation_events(updated_ms)",
        ],
        "migration_002_contract_mismatch",
        issues,
    );
}

fn validate_required_migration_content(
    path: &Path,
    required_fragments: &[&str],
    code: &'static str,
    issues: &mut Vec<ValidationIssue>,
) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    for fragment in required_fragments {
        if !text.contains(fragment) {
            issues.push(ValidationIssue {
                code,
                message: format!(
                    "required migration fragment '{}' missing from {}",
                    fragment,
                    path.display()
                ),
            });
        }
    }
}

fn validate_backend_deploy_assets(root: &Path, issues: &mut Vec<ValidationIssue>) {
    let env_snapshot_path = root.join("contracts").join("deploy_env.snapshot.json");
    let env_snapshot = match read_json(&env_snapshot_path) {
        Ok(value) => value,
        Err(_) => return,
    };
    let profile = env_snapshot
        .get("profile")
        .and_then(|value| value.as_str())
        .unwrap_or("backend");
    let required_env = env_snapshot
        .get("required_env")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let deploy_compose = root.join("deploy").join("docker-compose.yml");
    if deploy_compose.is_file() {
        validate_deploy_file_contains_env(
            &deploy_compose,
            &required_env,
            "docker_compose_missing_required_env",
            issues,
        );
    }
    let systemd_unit = root
        .join("deploy")
        .join("systemd")
        .join("enkai-backend.service");
    if systemd_unit.is_file() {
        validate_systemd_unit(&systemd_unit, profile, issues);
    }
}

fn validate_deploy_file_contains_env(
    path: &Path,
    required_env: &[serde_json::Value],
    code: &'static str,
    issues: &mut Vec<ValidationIssue>,
) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    for key in required_env.iter().filter_map(|value| value.as_str()) {
        if !text.contains(key) {
            issues.push(ValidationIssue {
                code,
                message: format!(
                    "deploy asset {} does not mention required env key {}",
                    path.display(),
                    key
                ),
            });
        }
    }
}

fn validate_systemd_unit(path: &Path, profile: &str, issues: &mut Vec<ValidationIssue>) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    for fragment in [
        "EnvironmentFile=/opt/enkai-app/.env",
        "ExecStart=/usr/local/bin/enkai serve --host 0.0.0.0 --port 8080 .",
        "Restart=on-failure",
    ] {
        if !text.contains(fragment) {
            issues.push(ValidationIssue {
                code: "systemd_contract_mismatch",
                message: format!(
                    "required systemd fragment '{}' missing from {}",
                    fragment,
                    path.display()
                ),
            });
        }
    }
    if !text.contains(profile) {
        issues.push(ValidationIssue {
            code: "systemd_profile_mismatch",
            message: format!(
                "systemd unit {} does not mention expected profile '{}'",
                path.display(),
                profile
            ),
        });
    }
}

fn validate_fullstack_frontend_assets(frontend_dir: &Path, issues: &mut Vec<ValidationIssue>) {
    let package_json = frontend_dir.join("package.json");
    let sdk_source = frontend_dir.join("src").join("sdk").join("enkaiClient.ts");
    let Ok(package_text) = fs::read_to_string(&package_json) else {
        return;
    };
    if !package_text.contains("\"react\"") || !package_text.contains("\"typescript\"") {
        issues.push(ValidationIssue {
            code: "frontend_package_contract_mismatch",
            message: format!(
                "frontend package missing required react/typescript dependencies: {}",
                package_json.display()
            ),
        });
    }
    let Ok(sdk_text) = fs::read_to_string(&sdk_source) else {
        return;
    };
    for fragment in [
        "x-enkai-api-version",
        "streamChat(",
        "streamChatWs(",
        "/chat/stream",
        "/chat/ws",
    ] {
        if !sdk_text.contains(fragment) {
            issues.push(ValidationIssue {
                code: "frontend_sdk_contract_mismatch",
                message: format!(
                    "frontend SDK missing required fragment '{}' in {}",
                    fragment,
                    sdk_source.display()
                ),
            });
        }
    }
}

fn validate_env_snapshot_alignment(
    deploy_snapshot_path: &Path,
    env_example_path: &Path,
    issues: &mut Vec<ValidationIssue>,
) {
    let deploy_value = match read_json(deploy_snapshot_path) {
        Ok(value) => value,
        Err(err) => {
            issues.push(ValidationIssue {
                code: "invalid_deploy_snapshot_json",
                message: format!("{}: {}", deploy_snapshot_path.display(), err),
            });
            return;
        }
    };
    let required_env = deploy_value
        .get("required_env")
        .and_then(|value| value.as_array());
    let Some(required_env) = required_env else {
        issues.push(ValidationIssue {
            code: "invalid_deploy_snapshot_required_env",
            message: format!(
                "deploy snapshot missing required_env array: {}",
                deploy_snapshot_path.display()
            ),
        });
        return;
    };
    let env_map = parse_env_file(env_example_path);
    for key in required_env.iter().filter_map(|value| value.as_str()) {
        let value = env_map.get(key).cloned().unwrap_or_default();
        if value.trim().is_empty() {
            issues.push(ValidationIssue {
                code: "missing_required_env_value",
                message: format!("{} is missing/empty in {}", key, env_example_path.display()),
            });
        }
    }
}

fn run_env_validator(
    root: &Path,
    validator_path: &Path,
    env_example_path: &Path,
    issues: &mut Vec<ValidationIssue>,
) {
    let python = match resolve_python_command() {
        Some(command) => command,
        None => {
            issues.push(ValidationIssue {
                code: "env_contract_validator_launch_failed",
                message: "python runtime not found (tried python3/python/py -3)".to_string(),
            });
            return;
        }
    };
    let mut cmd = Command::new(&python[0]);
    if python.len() > 1 {
        cmd.args(&python[1..]);
    }
    let status = cmd
        .arg(validator_path)
        .arg("--env-file")
        .arg(env_example_path)
        .current_dir(root)
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            issues.push(ValidationIssue {
                code: "env_contract_validator_failed",
                message: format!(
                    "validator returned exit code {} for {}",
                    status.code().unwrap_or(1),
                    validator_path.display()
                ),
            });
        }
        Err(err) => {
            issues.push(ValidationIssue {
                code: "env_contract_validator_launch_failed",
                message: format!("failed to launch {}: {}", validator_path.display(), err),
            });
        }
    }
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

fn validate_api_version_alignment(
    backend_snapshot: &Path,
    frontend_snapshot: &Path,
    issues: &mut Vec<ValidationIssue>,
) {
    let backend = match read_json(backend_snapshot) {
        Ok(value) => value,
        Err(err) => {
            issues.push(ValidationIssue {
                code: "invalid_backend_snapshot_json",
                message: format!("{}: {}", backend_snapshot.display(), err),
            });
            return;
        }
    };
    let frontend = match read_json(frontend_snapshot) {
        Ok(value) => value,
        Err(err) => {
            issues.push(ValidationIssue {
                code: "invalid_frontend_snapshot_json",
                message: format!("{}: {}", frontend_snapshot.display(), err),
            });
            return;
        }
    };
    let backend_version = backend
        .get("api_version")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let frontend_version = frontend
        .get("api_version")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    if backend_version.is_empty() || frontend_version.is_empty() {
        issues.push(ValidationIssue {
            code: "missing_api_version_in_snapshots",
            message: format!(
                "api_version missing in {} or {}",
                backend_snapshot.display(),
                frontend_snapshot.display()
            ),
        });
        return;
    }
    if backend_version != frontend_version {
        issues.push(ValidationIssue {
            code: "api_version_snapshot_mismatch",
            message: format!(
                "backend api_version '{}' != frontend api_version '{}'",
                backend_version, frontend_version
            ),
        });
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    serde_json::from_str(&text).map_err(|err| format!("invalid JSON: {}", err))
}

fn parse_env_file(path: &Path) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Ok(text) = fs::read_to_string(path) else {
        return out;
    };
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        out.insert(key.trim().to_string(), value.trim().to_string());
    }
    out
}

fn write_validation_report(report: &ValidationReport, output: Option<&Path>) -> Result<(), String> {
    let text = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed to serialize deploy validation report: {}", err))?;
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {}", parent.display(), err))?;
        }
        fs::write(path, &text)
            .map_err(|err| format!("failed to write {}: {}", path.display(), err))?;
    }
    println!("{}", text);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value as JsonValue;
    use tempfile::tempdir;

    #[test]
    fn parse_validate_args_requires_profile() {
        let dir = tempdir().expect("tempdir");
        let err =
            parse_validate_args(&[dir.path().to_string_lossy().to_string()]).expect_err("err");
        assert!(err.contains("--profile"));
    }

    #[test]
    fn parse_validate_args_accepts_backend_strict() {
        let dir = tempdir().expect("tempdir");
        let parsed = parse_validate_args(&[
            dir.path().to_string_lossy().to_string(),
            "--profile".to_string(),
            "backend".to_string(),
            "--strict".to_string(),
        ])
        .expect("parse");
        assert_eq!(parsed.profile, DeployProfile::Backend);
        assert!(parsed.strict);
        assert!(!parsed.json);
    }

    #[test]
    fn parse_validate_args_accepts_json_output() {
        let dir = tempdir().expect("tempdir");
        let parsed = parse_validate_args(&[
            dir.path().to_string_lossy().to_string(),
            "--profile".to_string(),
            "fullstack".to_string(),
            "--strict".to_string(),
            "--json".to_string(),
            "--output".to_string(),
            "artifacts/report.json".to_string(),
        ])
        .expect("parse");
        assert_eq!(parsed.profile, DeployProfile::Fullstack);
        assert!(parsed.strict);
        assert!(parsed.json);
        assert_eq!(
            parsed.output.as_deref(),
            Some(Path::new("artifacts/report.json"))
        );
    }

    #[test]
    fn validate_backend_project_flags_missing_files() {
        let dir = tempdir().expect("tempdir");
        let mut issues = Vec::new();
        validate_backend_project(dir.path(), &mut issues);
        assert!(!issues.is_empty());
        assert!(issues
            .iter()
            .any(|issue| issue.code == "missing_backend_manifest"));
    }

    #[test]
    fn validate_migration_contract_flags_invalid_sequence() {
        let dir = tempdir().expect("tempdir");
        let migrations = dir.path().join("migrations");
        fs::create_dir_all(&migrations).expect("migrations");
        fs::write(
            migrations.join("002_conversation_state_index.sql"),
            "CREATE INDEX foo ON bar(id);",
        )
        .expect("write");
        let mut issues = Vec::new();
        validate_migration_contract(dir.path(), &mut issues);
        assert!(issues
            .iter()
            .any(|issue| issue.code == "migration_sequence_gap"));
    }

    #[test]
    fn write_validation_report_outputs_json_file() {
        let dir = tempdir().expect("tempdir");
        let output = dir.path().join("artifacts").join("deploy.json");
        let report = ValidationReport {
            schema_version: 1,
            profile: "backend",
            project_dir: dir.path().display().to_string(),
            strict: true,
            success: false,
            issue_count: 1,
            issues: vec![ValidationIssue {
                code: "sample_issue",
                message: "sample".to_string(),
            }],
        };
        write_validation_report(&report, Some(&output)).expect("write report");
        let parsed: JsonValue =
            serde_json::from_str(&fs::read_to_string(output).expect("report text")).expect("json");
        assert_eq!(
            parsed.get("profile").and_then(|v| v.as_str()),
            Some("backend")
        );
        assert_eq!(parsed.get("issue_count").and_then(|v| v.as_u64()), Some(1));
    }
}
