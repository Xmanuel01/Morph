use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn print_deploy_usage() {
    eprintln!("  enkai deploy validate <project_dir> --profile <backend|fullstack> --strict");
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
}

#[derive(Debug, Clone)]
struct ValidationIssue {
    code: &'static str,
    message: String,
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

    if issues.is_empty() {
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
        for issue in &issues {
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
            "--strict" => strict = true,
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
