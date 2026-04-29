use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::deploy::{
    DeployProfile, DeployValidateArgs, ProjectLayoutCheck, RequiredPathKind, RequiredPathSpec,
    ValidationIssue, ValidationReport,
};
use crate::systems::DeployValidateManifest;

pub(crate) fn execute_manifest(manifest: &DeployValidateManifest) -> i32 {
    let parsed = match deploy_validate_args_from_manifest(manifest) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("enkai deploy validate: {}", err);
            return 1;
        }
    };
    execute_validate(&parsed)
}

pub(crate) fn deploy_validate_args_from_manifest(
    manifest: &DeployValidateManifest,
) -> Result<DeployValidateArgs, String> {
    let project_dir = std::path::PathBuf::from(&manifest.project_dir);
    if !project_dir.is_dir() {
        return Err(format!(
            "project directory not found: {}",
            project_dir.display()
        ));
    }
    let profile = parse_profile(&manifest.target_profile)?;
    Ok(DeployValidateArgs {
        project_dir,
        profile,
        strict: manifest.strict,
        json: manifest.json,
        output: manifest.output.as_ref().map(std::path::PathBuf::from),
    })
}

pub(crate) fn execute_validate(parsed: &DeployValidateArgs) -> i32 {
    if !parsed.strict {
        eprintln!("enkai deploy validate: --strict is required for production validation");
        return 1;
    }

    let mut issues = Vec::new();
    match parsed.profile {
        DeployProfile::Backend => validate_backend_project(&parsed.project_dir, &mut issues),
        DeployProfile::Fullstack => validate_fullstack_project(&parsed.project_dir, &mut issues),
        DeployProfile::Mobile => validate_mobile_project(&parsed.project_dir, &mut issues),
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

pub(crate) fn profile_name(profile: DeployProfile) -> &'static str {
    match profile {
        DeployProfile::Backend => "backend",
        DeployProfile::Fullstack => "fullstack",
        DeployProfile::Mobile => "mobile",
    }
}

pub(crate) fn backend_required_paths(root: &Path) -> Vec<RequiredPathSpec> {
    vec![
        required_file(
            "missing_backend_manifest",
            root.join("enkai.toml"),
            "required backend manifest missing",
        ),
        required_file(
            "missing_backend_entrypoint",
            root.join("src").join("main.enk"),
            "required backend entrypoint missing",
        ),
        required_file(
            "missing_backend_contract_snapshot",
            root.join("contracts").join("backend_api.snapshot.json"),
            "backend API contract snapshot missing",
        ),
        required_file(
            "missing_conversation_schema",
            root.join("contracts")
                .join("conversation_state.schema.json"),
            "conversation schema snapshot missing",
        ),
        required_file(
            "missing_grpc_contract_snapshot",
            root.join("contracts").join("grpc_api.snapshot.json"),
            "gRPC contract snapshot missing",
        ),
        required_file(
            "missing_worker_queue_snapshot",
            root.join("contracts").join("worker_queue.snapshot.json"),
            "worker queue contract snapshot missing",
        ),
        required_file(
            "missing_db_engines_snapshot",
            root.join("contracts").join("db_engines.snapshot.json"),
            "DB engines contract snapshot missing",
        ),
        required_file(
            "missing_grpc_proto",
            root.join("contracts").join("enkai_chat.proto"),
            "gRPC proto missing",
        ),
        required_file(
            "missing_deploy_env_snapshot",
            root.join("contracts").join("deploy_env.snapshot.json"),
            "deploy env snapshot missing",
        ),
        required_file(
            "missing_env_contract_validator",
            root.join("scripts").join("validate_env_contract.py"),
            "deploy env validator script missing",
        ),
        required_file(
            "missing_env_example",
            root.join(".env.example"),
            ".env.example missing",
        ),
        required_file(
            "missing_migration_001",
            root.join("migrations").join("001_conversation_state.sql"),
            "migration 001 missing",
        ),
        required_file(
            "missing_migration_002",
            root.join("migrations")
                .join("002_conversation_state_index.sql"),
            "migration 002 missing",
        ),
        required_file(
            "missing_worker_handler",
            root.join("worker").join("handler.enk"),
            "worker handler missing",
        ),
        required_file(
            "missing_backend_dockerfile",
            root.join("deploy").join("docker").join("Dockerfile"),
            "backend deploy Dockerfile missing",
        ),
        required_file(
            "missing_backend_compose",
            root.join("deploy").join("docker-compose.yml"),
            "backend deploy docker-compose profile missing",
        ),
        required_file(
            "missing_worker_systemd_unit",
            root.join("deploy")
                .join("systemd")
                .join("enkai-worker.service"),
            "worker deploy systemd unit missing",
        ),
        required_file(
            "missing_backend_systemd_unit",
            root.join("deploy")
                .join("systemd")
                .join("enkai-backend.service"),
            "backend deploy systemd unit missing",
        ),
    ]
}

pub(crate) fn fullstack_required_paths(root: &Path) -> Vec<RequiredPathSpec> {
    let backend_dir = root.join("backend");
    let frontend_dir = root.join("frontend");
    let mut specs = vec![
        required_dir(
            "missing_backend_dir",
            backend_dir.clone(),
            "fullstack backend directory missing",
        ),
        required_dir(
            "missing_frontend_dir",
            frontend_dir.clone(),
            "fullstack frontend directory missing",
        ),
        required_file(
            "missing_fullstack_compose",
            root.join("deploy").join("docker-compose.yml"),
            "fullstack deploy docker-compose profile missing",
        ),
    ];
    specs.extend([
        required_file(
            "missing_frontend_package",
            frontend_dir.join("package.json"),
            "frontend package.json missing",
        ),
        required_file(
            "missing_frontend_sdk_snapshot",
            frontend_dir.join("contracts").join("sdk_api.snapshot.json"),
            "frontend SDK snapshot missing",
        ),
        required_file(
            "missing_frontend_sdk",
            frontend_dir.join("src").join("sdk").join("enkaiClient.ts"),
            "frontend SDK source missing",
        ),
        required_file(
            "missing_frontend_env_example",
            frontend_dir.join(".env.example"),
            "frontend .env.example missing",
        ),
    ]);
    specs
}

pub(crate) fn mobile_required_paths(root: &Path) -> Vec<RequiredPathSpec> {
    vec![
        required_file(
            "missing_mobile_manifest",
            root.join("enkai.toml"),
            "mobile manifest missing",
        ),
        required_file(
            "missing_mobile_package",
            root.join("package.json"),
            "mobile package.json missing",
        ),
        required_file(
            "missing_mobile_app_json",
            root.join("app.json"),
            "mobile app.json missing",
        ),
        required_file(
            "missing_mobile_app_entry",
            root.join("src").join("App.tsx"),
            "mobile App.tsx missing",
        ),
        required_file(
            "missing_mobile_sdk_snapshot",
            root.join("contracts").join("sdk_api.snapshot.json"),
            "mobile SDK snapshot missing",
        ),
        required_file(
            "missing_mobile_sdk",
            root.join("src").join("sdk").join("enkaiClient.ts"),
            "mobile SDK source missing",
        ),
        required_file(
            "missing_mobile_env_example",
            root.join(".env.example"),
            "mobile .env.example missing",
        ),
    ]
}

pub(crate) fn validate_backend_project(root: &Path, issues: &mut Vec<ValidationIssue>) {
    for spec in backend_required_paths(root) {
        if !spec.path.is_file() {
            issues.push(ValidationIssue {
                code: spec.code,
                message: format!("{}: {}", spec.message, spec.path.display()),
            });
        }
    }

    let deploy_snapshot_path = root.join("contracts").join("deploy_env.snapshot.json");
    let env_example_path = root.join(".env.example");
    if deploy_snapshot_path.is_file() && env_example_path.is_file() {
        validate_env_snapshot_alignment(&deploy_snapshot_path, &env_example_path, issues);
    }
    validate_backend_contract_assets(root, issues);

    if deploy_snapshot_path.is_file() && env_example_path.is_file() {
        run_env_validator(&deploy_snapshot_path, &env_example_path, issues);
    }
}

pub(crate) fn validate_fullstack_project(root: &Path, issues: &mut Vec<ValidationIssue>) {
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
    for spec in fullstack_required_paths(root) {
        if spec.code == "missing_backend_dir"
            || spec.code == "missing_frontend_dir"
            || spec.code == "missing_fullstack_compose"
        {
            continue;
        }
        if !spec.path.is_file() {
            issues.push(ValidationIssue {
                code: spec.code,
                message: format!("{}: {}", spec.message, spec.path.display()),
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

pub(crate) fn validate_mobile_project(root: &Path, issues: &mut Vec<ValidationIssue>) {
    for spec in mobile_required_paths(root) {
        if !spec.path.is_file() {
            issues.push(ValidationIssue {
                code: spec.code,
                message: format!("{}: {}", spec.message, spec.path.display()),
            });
        }
    }
    validate_mobile_assets(root, issues);
}

pub(crate) fn evaluate_project_layout_json(
    project_dir: &Path,
    target_profile: &str,
) -> Result<serde_json::Value, String> {
    let profile = parse_profile(target_profile)?;
    let checks = collect_profile_required_paths(project_dir, profile)
        .into_iter()
        .map(|spec| ProjectLayoutCheck {
            code: spec.code,
            path: spec.path.display().to_string(),
            message: spec.message,
            kind: match spec.kind {
                RequiredPathKind::File => "file",
                RequiredPathKind::Dir => "dir",
            },
            present: match spec.kind {
                RequiredPathKind::File => spec.path.is_file(),
                RequiredPathKind::Dir => spec.path.is_dir(),
            },
        })
        .collect::<Vec<_>>();
    let missing_count = checks.iter().filter(|check| !check.present).count();
    Ok(serde_json::json!({
        "profile": profile_name(profile),
        "project_dir": project_dir.display().to_string(),
        "required_paths": checks,
        "missing_required_paths": missing_count,
    }))
}

pub(crate) fn validate_api_version_alignment(
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

pub(crate) fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    serde_json::from_str(&text).map_err(|err| format!("invalid JSON: {}", err))
}

pub(crate) fn parse_env_file(path: &Path) -> HashMap<String, String> {
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

pub(crate) fn write_validation_report(
    report: &ValidationReport,
    output: Option<&Path>,
) -> Result<(), String> {
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

pub(crate) fn validate_env_integer_range(
    env_map: &HashMap<String, String>,
    key: &str,
    min: i64,
    max: i64,
    issues: &mut Vec<ValidationIssue>,
) {
    let value = env_map.get(key).map(|raw| raw.trim()).unwrap_or("");
    if value.is_empty() {
        return;
    }
    match value.parse::<i64>() {
        Ok(parsed) if parsed >= min && parsed <= max => {}
        Ok(_) => {
            issues.push(ValidationIssue {
                code: "env_contract_validator_failed",
                message: format!("{} must be in range {}..{}", key, min, max),
            });
        }
        Err(_) => {
            issues.push(ValidationIssue {
                code: "env_contract_validator_failed",
                message: format!("{} must be an integer", key),
            });
        }
    }
}

pub(crate) fn validate_env_snapshot_alignment(
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

pub(crate) fn run_env_validator(
    deploy_snapshot_path: &Path,
    env_example_path: &Path,
    issues: &mut Vec<ValidationIssue>,
) {
    let deploy_value = match read_json(deploy_snapshot_path) {
        Ok(value) => value,
        Err(err) => {
            issues.push(ValidationIssue {
                code: "env_contract_validator_failed",
                message: format!("{}: {}", deploy_snapshot_path.display(), err),
            });
            return;
        }
    };
    let env_map = parse_env_file(env_example_path);
    let expected_api_version = deploy_value
        .get("api_version")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    let expected_profile = deploy_value
        .get("profile")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    if let Some(required_env) = deploy_value
        .get("required_env")
        .and_then(|value| value.as_array())
    {
        for key in required_env.iter().filter_map(|value| value.as_str()) {
            let value = env_map.get(key).cloned().unwrap_or_default();
            if value.trim().is_empty() {
                issues.push(ValidationIssue {
                    code: "env_contract_validator_failed",
                    message: format!(
                        "missing required env '{}' in {}",
                        key,
                        env_example_path.display()
                    ),
                });
            }
        }
    } else {
        issues.push(ValidationIssue {
            code: "env_contract_validator_failed",
            message: format!(
                "deploy snapshot missing required_env array: {}",
                deploy_snapshot_path.display()
            ),
        });
    }
    let api_version = env_map
        .get("ENKAI_API_VERSION")
        .map(|value| value.trim())
        .unwrap_or("");
    if !api_version.is_empty()
        && !expected_api_version.is_empty()
        && api_version != expected_api_version
    {
        issues.push(ValidationIssue {
            code: "env_contract_validator_failed",
            message: format!(
                "ENKAI_API_VERSION mismatch: expected {}, got {}",
                expected_api_version, api_version
            ),
        });
    }
    let profile = env_map
        .get("ENKAI_APP_PROFILE")
        .map(|value| value.trim())
        .unwrap_or("");
    if !profile.is_empty() && !expected_profile.is_empty() && profile != expected_profile {
        issues.push(ValidationIssue {
            code: "env_contract_validator_failed",
            message: format!(
                "ENKAI_APP_PROFILE mismatch: expected {}, got {}",
                expected_profile, profile
            ),
        });
    }
    validate_env_integer_range(&env_map, "ENKAI_SERVE_PORT", 1, 65535, issues);
    validate_env_integer_range(&env_map, "ENKAI_GRPC_PORT", 1, 65535, issues);
    validate_env_integer_range(&env_map, "ENKAI_DB_POOL_MAX", 1, 64, issues);
    let db_engine = env_map
        .get("ENKAI_DB_ENGINE")
        .map(|value| value.trim())
        .unwrap_or("");
    if !db_engine.is_empty() && !matches!(db_engine, "sqlite" | "postgres" | "mysql") {
        issues.push(ValidationIssue {
            code: "env_contract_validator_failed",
            message: "ENKAI_DB_ENGINE must be one of sqlite|postgres|mysql".to_string(),
        });
    }
}

fn parse_profile(raw: &str) -> Result<DeployProfile, String> {
    match raw.trim() {
        "backend" => Ok(DeployProfile::Backend),
        "fullstack" => Ok(DeployProfile::Fullstack),
        "mobile" => Ok(DeployProfile::Mobile),
        other => Err(format!(
            "invalid --profile '{}'; expected backend|fullstack|mobile",
            other
        )),
    }
}

fn collect_profile_required_paths(root: &Path, profile: DeployProfile) -> Vec<RequiredPathSpec> {
    match profile {
        DeployProfile::Backend => backend_required_paths(root),
        DeployProfile::Fullstack => fullstack_required_paths(root),
        DeployProfile::Mobile => mobile_required_paths(root),
    }
}

fn required_file(
    code: &'static str,
    path: std::path::PathBuf,
    message: &'static str,
) -> RequiredPathSpec {
    RequiredPathSpec {
        code,
        path,
        message,
        kind: RequiredPathKind::File,
    }
}

fn required_dir(
    code: &'static str,
    path: std::path::PathBuf,
    message: &'static str,
) -> RequiredPathSpec {
    RequiredPathSpec {
        code,
        path,
        message,
        kind: RequiredPathKind::Dir,
    }
}

fn validate_backend_contract_assets(root: &Path, issues: &mut Vec<ValidationIssue>) {
    validate_migration_contract(root, issues);
    validate_backend_deploy_assets(root, issues);
    validate_backend_protocol_contracts(root, issues);
}

fn validate_backend_protocol_contracts(root: &Path, issues: &mut Vec<ValidationIssue>) {
    let grpc_snapshot = root.join("contracts").join("grpc_api.snapshot.json");
    if let Ok(text) = fs::read_to_string(&grpc_snapshot) {
        for fragment in [
            "\"package\": \"enkai.chat.v1\"",
            "\"name\": \"ChatService\"",
            "\"name\": \"StreamChat\"",
        ] {
            if !text.contains(fragment) {
                issues.push(ValidationIssue {
                    code: "grpc_contract_mismatch",
                    message: format!(
                        "gRPC contract snapshot missing '{}' in {}",
                        fragment,
                        grpc_snapshot.display()
                    ),
                });
            }
        }
    }
    let worker_snapshot = root.join("contracts").join("worker_queue.snapshot.json");
    if let Ok(text) = fs::read_to_string(&worker_snapshot) {
        for fragment in [
            "\"queue_kind\": \"file_jsonl\"",
            "\"durable_enqueue\": true",
            "\"dead_letter\": true",
        ] {
            if !text.contains(fragment) {
                issues.push(ValidationIssue {
                    code: "worker_queue_contract_mismatch",
                    message: format!(
                        "worker queue snapshot missing '{}' in {}",
                        fragment,
                        worker_snapshot.display()
                    ),
                });
            }
        }
    }
    let db_snapshot = root.join("contracts").join("db_engines.snapshot.json");
    if let Ok(text) = fs::read_to_string(&db_snapshot) {
        for fragment in [
            "\"sqlite\"",
            "\"postgres\"",
            "\"mysql\"",
            "\"schema_migrations\"",
        ] {
            if !text.contains(fragment) {
                issues.push(ValidationIssue {
                    code: "db_engine_contract_mismatch",
                    message: format!(
                        "DB engine snapshot missing '{}' in {}",
                        fragment,
                        db_snapshot.display()
                    ),
                });
            }
        }
    }
    let proto = root.join("contracts").join("enkai_chat.proto");
    if let Ok(text) = fs::read_to_string(&proto) {
        for fragment in [
            "service ChatService",
            "rpc Chat",
            "rpc StreamChat",
            "package enkai.chat.v1;",
        ] {
            if !text.contains(fragment) {
                issues.push(ValidationIssue {
                    code: "grpc_proto_contract_mismatch",
                    message: format!("gRPC proto missing '{}' in {}", fragment, proto.display()),
                });
            }
        }
    }
}

pub(crate) fn validate_migration_contract(root: &Path, issues: &mut Vec<ValidationIssue>) {
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
    let worker_unit = root
        .join("deploy")
        .join("systemd")
        .join("enkai-worker.service");
    if worker_unit.is_file() {
        validate_worker_systemd_unit(&worker_unit, issues);
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

fn validate_worker_systemd_unit(path: &Path, issues: &mut Vec<ValidationIssue>) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    for fragment in ["EnvironmentFile=/opt/enkai-app/.env", "Restart=on-failure"] {
        if !text.contains(fragment) {
            issues.push(ValidationIssue {
                code: "worker_systemd_contract_mismatch",
                message: format!(
                    "required worker systemd fragment '{}' missing from {}",
                    fragment,
                    path.display()
                ),
            });
        }
    }

    let exec_line = text
        .lines()
        .find(|line| line.trim_start().starts_with("ExecStart="))
        .unwrap_or_default();
    for fragment in [
        "/usr/local/bin/enkai worker run",
        "--queue default",
        "--handler worker/handler.enk",
    ] {
        if !exec_line.contains(fragment) {
            issues.push(ValidationIssue {
                code: "worker_systemd_contract_mismatch",
                message: format!(
                    "worker systemd ExecStart missing '{}' in {}",
                    fragment,
                    path.display()
                ),
            });
        }
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

fn validate_mobile_assets(root: &Path, issues: &mut Vec<ValidationIssue>) {
    let package_json = root.join("package.json");
    let sdk_source = root.join("src").join("sdk").join("enkaiClient.ts");
    let env_example = root.join(".env.example");
    if let Ok(package_text) = fs::read_to_string(&package_json) {
        for fragment in ["expo", "react-native", "typescript"] {
            if !package_text.contains(fragment) {
                issues.push(ValidationIssue {
                    code: "mobile_package_contract_mismatch",
                    message: format!(
                        "mobile package missing '{}' in {}",
                        fragment,
                        package_json.display()
                    ),
                });
            }
        }
    }
    if let Ok(sdk_text) = fs::read_to_string(&sdk_source) {
        for fragment in ["generated target: mobile", "streamChat(", "streamChatWs("] {
            if !sdk_text.contains(fragment) {
                issues.push(ValidationIssue {
                    code: "mobile_sdk_contract_mismatch",
                    message: format!(
                        "mobile SDK missing required fragment '{}' in {}",
                        fragment,
                        sdk_source.display()
                    ),
                });
            }
        }
    }
    if let Ok(env_text) = fs::read_to_string(&env_example) {
        for fragment in [
            "EXPO_PUBLIC_ENKAI_API_BASE_URL",
            "EXPO_PUBLIC_ENKAI_API_VERSION",
        ] {
            if !env_text.contains(fragment) {
                issues.push(ValidationIssue {
                    code: "mobile_env_contract_mismatch",
                    message: format!(
                        "mobile env example missing '{}' in {}",
                        fragment,
                        env_example.display()
                    ),
                });
            }
        }
    }
}
