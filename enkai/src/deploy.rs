#[cfg(test)]
use std::fs;
use std::path::PathBuf;

use crate::systems::DeployValidateManifest;
use serde::Serialize;

pub fn print_deploy_usage() {
    eprintln!(
        "  enkai deploy validate <project_dir> --profile <backend|fullstack|mobile> --strict [--json] [--output <file>]"
    );
}

pub fn deploy_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_deploy_usage();
        return 1;
    }
    match args[0].as_str() {
        "validate" => {
            let manifest = match crate::systems::build_deploy_validate_manifest(args) {
                Ok(value) => value,
                Err(err) => {
                    eprintln!("enkai deploy validate: {}", err);
                    print_deploy_usage();
                    return 1;
                }
            };
            execute_deploy_manifest(&manifest)
        }
        other => {
            eprintln!("enkai deploy: unknown subcommand '{}'", other);
            print_deploy_usage();
            1
        }
    }
}

pub(crate) fn execute_deploy_manifest(manifest: &DeployValidateManifest) -> i32 {
    crate::deploy_runtime::execute_manifest(manifest)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeployProfile {
    Backend,
    Fullstack,
    Mobile,
}

#[derive(Debug, Clone)]
pub(crate) struct DeployValidateArgs {
    pub(crate) project_dir: PathBuf,
    pub(crate) profile: DeployProfile,
    pub(crate) strict: bool,
    pub(crate) json: bool,
    pub(crate) output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ValidationIssue {
    pub(crate) code: &'static str,
    pub(crate) message: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ValidationReport {
    pub(crate) schema_version: u32,
    pub(crate) profile: &'static str,
    pub(crate) project_dir: String,
    pub(crate) strict: bool,
    pub(crate) success: bool,
    pub(crate) issue_count: usize,
    pub(crate) issues: Vec<ValidationIssue>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RequiredPathKind {
    File,
    Dir,
}

#[derive(Debug, Clone)]
pub(crate) struct RequiredPathSpec {
    pub(crate) code: &'static str,
    pub(crate) path: PathBuf,
    pub(crate) message: &'static str,
    pub(crate) kind: RequiredPathKind,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ProjectLayoutCheck {
    pub(crate) code: &'static str,
    pub(crate) path: String,
    pub(crate) message: &'static str,
    pub(crate) kind: &'static str,
    pub(crate) present: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value as JsonValue;
    use tempfile::tempdir;

    #[test]
    fn parse_validate_args_requires_profile() {
        let dir = tempdir().expect("tempdir");
        let err = crate::systems::build_deploy_validate_manifest(&[
            "validate".to_string(),
            dir.path().to_string_lossy().to_string(),
        ])
        .expect_err("err");
        assert!(err.contains("--profile"));
    }

    #[test]
    fn parse_validate_args_accepts_backend_strict() {
        let dir = tempdir().expect("tempdir");
        let manifest = crate::systems::build_deploy_validate_manifest(&[
            "validate".to_string(),
            dir.path().to_string_lossy().to_string(),
            "--profile".to_string(),
            "backend".to_string(),
            "--strict".to_string(),
        ])
        .expect("parse");
        let parsed =
            crate::deploy_runtime::deploy_validate_args_from_manifest(&manifest).expect("args");
        assert_eq!(parsed.profile, DeployProfile::Backend);
        assert!(parsed.strict);
        assert!(!parsed.json);
    }

    #[test]
    fn parse_validate_args_accepts_json_output() {
        let dir = tempdir().expect("tempdir");
        let manifest = crate::systems::build_deploy_validate_manifest(&[
            "validate".to_string(),
            dir.path().to_string_lossy().to_string(),
            "--profile".to_string(),
            "fullstack".to_string(),
            "--strict".to_string(),
            "--json".to_string(),
            "--output".to_string(),
            "artifacts/report.json".to_string(),
        ])
        .expect("parse");
        let parsed =
            crate::deploy_runtime::deploy_validate_args_from_manifest(&manifest).expect("args");
        assert_eq!(parsed.profile, DeployProfile::Fullstack);
        assert!(parsed.strict);
        assert!(parsed.json);
        assert_eq!(
            parsed.output.as_deref(),
            Some(std::path::Path::new("artifacts/report.json"))
        );
    }

    #[test]
    fn parse_validate_args_accepts_mobile_profile() {
        let dir = tempdir().expect("tempdir");
        let manifest = crate::systems::build_deploy_validate_manifest(&[
            "validate".to_string(),
            dir.path().to_string_lossy().to_string(),
            "--profile".to_string(),
            "mobile".to_string(),
            "--strict".to_string(),
        ])
        .expect("parse");
        let parsed =
            crate::deploy_runtime::deploy_validate_args_from_manifest(&manifest).expect("args");
        assert_eq!(parsed.profile, DeployProfile::Mobile);
    }

    #[test]
    fn validate_backend_project_flags_missing_files() {
        let dir = tempdir().expect("tempdir");
        let mut issues = Vec::new();
        crate::deploy_runtime::validate_backend_project(dir.path(), &mut issues);
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
        crate::deploy_runtime::validate_migration_contract(dir.path(), &mut issues);
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
        crate::deploy_runtime::write_validation_report(&report, Some(&output))
            .expect("write report");
        let parsed: JsonValue =
            serde_json::from_str(&fs::read_to_string(output).expect("report text")).expect("json");
        assert_eq!(
            parsed.get("profile").and_then(|v| v.as_str()),
            Some("backend")
        );
        assert_eq!(parsed.get("issue_count").and_then(|v| v.as_u64()), Some(1));
    }
}
