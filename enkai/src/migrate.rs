use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Map, Value};

use crate::train;

const EXIT_OK: i32 = 0;
const EXIT_ERROR: i32 = 1;
const EXIT_BLOCKED: i32 = 2;

#[derive(Debug)]
struct DoctorOptions {
    root: PathBuf,
    json: bool,
    strict_contracts: bool,
}

#[derive(Debug, Default, Clone, Serialize)]
struct DoctorReport {
    blockers: Vec<String>,
    warnings: Vec<String>,
}

pub fn migrate_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_usage();
        return EXIT_ERROR;
    }
    match args[0].as_str() {
        "config-v1" => migrate_config_v1(&args[1..]),
        "checkpoint-meta-v1" => migrate_checkpoint_meta_v1(&args[1..]),
        _ => {
            eprintln!("Unknown migrate subcommand: {}", args[0]);
            print_usage();
            EXIT_ERROR
        }
    }
}

pub fn doctor_command(args: &[String]) -> i32 {
    let options = match parse_doctor_options(args) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{}", err);
            eprintln!("Usage: enkai doctor [path] [--json] [--strict-contracts|--lenient]");
            return EXIT_ERROR;
        }
    };
    let report = run_doctor(&options.root, options.strict_contracts);
    let blocked = !report.blockers.is_empty();
    if options.json {
        let payload = serde_json::json!({
            "status": if blocked { "blocked" } else { "ok" },
            "path": options.root.display().to_string(),
            "strict_contracts": options.strict_contracts,
            "blockers": report.blockers,
            "warnings": report.warnings,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string())
        );
    } else if !blocked {
        println!("doctor: ready for v2.0 strict config/checkpoint contracts");
        if !report.warnings.is_empty() {
            for warning in &report.warnings {
                println!("warning: {}", warning);
            }
        }
    } else {
        eprintln!("doctor: v2.0 readiness blockers found:");
        for issue in &report.blockers {
            eprintln!("- {}", issue);
        }
        for warning in &report.warnings {
            eprintln!("warning: {}", warning);
        }
    }
    if blocked {
        EXIT_BLOCKED
    } else {
        EXIT_OK
    }
}

pub fn print_usage() {
    eprintln!("  enkai migrate config-v1 <in_config.enk> <out_config.enk|out.json>");
    eprintln!(
        "  enkai migrate checkpoint-meta-v1 <checkpoint_dir> [--dry-run] [--verify] [--strict-contracts]"
    );
    eprintln!("  enkai doctor [path] [--json] [--strict-contracts|--lenient]");
}

fn parse_doctor_options(args: &[String]) -> Result<DoctorOptions, String> {
    let mut root: Option<PathBuf> = None;
    let mut json = false;
    let mut strict_contracts = true;
    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            "--strict-contracts" => strict_contracts = true,
            "--lenient" => strict_contracts = false,
            _ if arg.starts_with('-') => return Err(format!("Unknown option: {}", arg)),
            _ => {
                if root.is_some() {
                    return Err("enkai doctor accepts at most one path".to_string());
                }
                root = Some(PathBuf::from(arg));
            }
        }
    }
    Ok(DoctorOptions {
        root: root.unwrap_or_else(|| PathBuf::from(".")),
        json,
        strict_contracts,
    })
}

fn run_doctor(root: &Path, strict_contracts: bool) -> DoctorReport {
    let mut blockers = Vec::new();
    let mut warnings = Vec::new();
    let config_files = collect_candidate_config_files(root);
    for config in config_files {
        match train::load_config_json(&config) {
            Ok(json) => {
                if !looks_like_train_config(&json) {
                    continue;
                }
                if json.get("config_version").is_none() {
                    blockers.push(format!(
                        "{}: missing config_version (run: enkai migrate config-v1 \"{}\" <out>)",
                        config.display(),
                        config.display()
                    ));
                } else if json
                    .get("config_version")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
                    != 1
                {
                    blockers.push(format!(
                        "{}: unsupported config_version (expected 1)",
                        config.display()
                    ));
                }
                let validation = if strict_contracts {
                    train::validate_train_config(&config, false)
                } else {
                    train::validate_train_config_with_mode(&config, false, false)
                };
                if let Err(err) = validation {
                    blockers.push(format!(
                        "{}: invalid train config: {}",
                        config.display(),
                        err
                    ));
                }
            }
            Err(err) => {
                blockers.push(format!(
                    "{}: failed to evaluate config: {}",
                    config.display(),
                    err
                ));
            }
        }
    }

    let checkpoint_roots = collect_checkpoint_roots(root);
    let mut seen = HashSet::new();
    for checkpoint_dir in checkpoint_roots {
        if !seen.insert(checkpoint_dir.clone()) {
            continue;
        }
        match verify_checkpoint_tree(&checkpoint_dir, strict_contracts) {
            Ok(report) => {
                if report.meta_files == 0 {
                    continue;
                }
                for issue in report.blockers {
                    blockers.push(format!("{}: {}", checkpoint_dir.display(), issue));
                }
                for issue in report.warnings {
                    warnings.push(format!("{}: {}", checkpoint_dir.display(), issue));
                }
            }
            Err(err) => blockers.push(format!("{}: {}", checkpoint_dir.display(), err)),
        }
    }
    DoctorReport { blockers, warnings }
}

fn migrate_config_v1(args: &[String]) -> i32 {
    if args.len() != 2 {
        eprintln!("Usage: enkai migrate config-v1 <in_config.enk> <out_config.enk|out.json>");
        return EXIT_ERROR;
    }
    let input = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    let config = match train::migrate_config_v1_json(&input) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("config migration failed: {}", err);
            return EXIT_BLOCKED;
        }
    };
    if let Err(err) = write_migrated_config(&output, &config) {
        eprintln!("failed to write {}: {}", output.display(), err);
        return EXIT_ERROR;
    }
    println!(
        "migrated config to v1: {} -> {}",
        input.display(),
        output.display()
    );
    EXIT_OK
}

fn migrate_checkpoint_meta_v1(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!(
            "Usage: enkai migrate checkpoint-meta-v1 <checkpoint_dir> [--dry-run] [--verify] [--strict-contracts]"
        );
        return EXIT_ERROR;
    }
    let checkpoint_dir = PathBuf::from(&args[0]);
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let verify_only = args.iter().any(|a| a == "--verify");
    let strict_contracts = args.iter().any(|a| a == "--strict-contracts");
    let mut unexpected = Vec::new();
    for arg in &args[1..] {
        if arg != "--dry-run" && arg != "--verify" && arg != "--strict-contracts" {
            unexpected.push(arg.clone());
        }
    }
    if !unexpected.is_empty() {
        eprintln!("Unknown options: {}", unexpected.join(", "));
        eprintln!(
            "Usage: enkai migrate checkpoint-meta-v1 <checkpoint_dir> [--dry-run] [--verify] [--strict-contracts]"
        );
        return EXIT_ERROR;
    }

    let report = if verify_only {
        match verify_checkpoint_tree(&checkpoint_dir, strict_contracts) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("checkpoint verify failed: {}", err);
                return EXIT_BLOCKED;
            }
        }
    } else {
        match upgrade_checkpoint_tree(&checkpoint_dir, dry_run) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("checkpoint migration failed: {}", err);
                return EXIT_BLOCKED;
            }
        }
    };

    for warning in &report.warnings {
        eprintln!("warning: {}", warning);
    }
    if !report.blockers.is_empty() {
        eprintln!("checkpoint contract blockers:");
        for issue in &report.blockers {
            eprintln!("- {}", issue);
        }
        return EXIT_BLOCKED;
    }
    if verify_only {
        println!(
            "checkpoint verify ok: {} meta files checked (legacy_missing={}, strict_contracts={})",
            report.meta_files, report.legacy_missing, strict_contracts
        );
    } else if dry_run {
        println!(
            "checkpoint migration dry-run: {} files would be updated ({} checked)",
            report.updated_files, report.meta_files
        );
    } else {
        println!(
            "checkpoint migration complete: {} files updated ({} checked)",
            report.updated_files, report.meta_files
        );
    }
    EXIT_OK
}

fn write_migrated_config(path: &Path, config: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("create {}: {}", parent.display(), err))?;
        }
    }
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());
    if matches!(ext.as_deref(), Some("json")) {
        let text = serde_json::to_string_pretty(config).map_err(|err| err.to_string())?;
        fs::write(path, text).map_err(|err| err.to_string())?;
        return Ok(());
    }
    let json_text = serde_json::to_string(config).map_err(|err| err.to_string())?;
    let escaped = json_text.replace('\\', "\\\\").replace('\"', "\\\"");
    let source = format!("fn main() ::\n    return json.parse(\"{}\")\n::\n", escaped);
    fs::write(path, source).map_err(|err| err.to_string())
}

#[derive(Default)]
struct CheckpointReport {
    meta_files: usize,
    updated_files: usize,
    legacy_missing: usize,
    blockers: Vec<String>,
    warnings: Vec<String>,
}

fn verify_checkpoint_tree(root: &Path, strict_contracts: bool) -> Result<CheckpointReport, String> {
    analyze_checkpoint_tree(root, false, true, strict_contracts)
}

fn upgrade_checkpoint_tree(root: &Path, dry_run: bool) -> Result<CheckpointReport, String> {
    analyze_checkpoint_tree(root, dry_run, false, false)
}

fn analyze_checkpoint_tree(
    root: &Path,
    dry_run: bool,
    verify_only: bool,
    strict_contracts: bool,
) -> Result<CheckpointReport, String> {
    let meta_files = collect_checkpoint_meta_files(root)?;
    if meta_files.is_empty() {
        return Err(format!(
            "no checkpoint meta.json files found in {}",
            root.display()
        ));
    }
    let mut report = CheckpointReport {
        meta_files: meta_files.len(),
        ..CheckpointReport::default()
    };
    let mut hashes = HashSet::new();
    let mut sigs = HashSet::new();
    let mut dtypes = HashSet::new();
    let mut devices = HashSet::new();

    for meta_path in meta_files {
        let raw = fs::read_to_string(&meta_path)
            .map_err(|err| format!("read {}: {}", meta_path.display(), err))?;
        let value: Value = serde_json::from_str(&raw)
            .map_err(|err| format!("parse {}: {}", meta_path.display(), err))?;
        let mut object = match value {
            Value::Object(map) => map,
            _ => {
                report
                    .blockers
                    .push(format!("{} is not a JSON object", meta_path.display()));
                continue;
            }
        };

        let mut changed = false;
        if !object.contains_key("format_version") {
            report.legacy_missing += 1;
            changed = true;
            object.insert("format_version".to_string(), Value::from(1u64));
        }
        changed |= ensure_u64_field(&mut object, "step", infer_step_from_path(&meta_path));
        changed |= ensure_u64_field(&mut object, "tokens", Some(0));
        changed |= ensure_f64_field(&mut object, "loss", Some(0.0));
        changed |= ensure_string_field(&mut object, "config_hash", Some(""));
        changed |= ensure_string_field(&mut object, "model_sig", Some(""));
        changed |= ensure_string_field(&mut object, "dtype", Some(""));
        changed |= ensure_string_field(&mut object, "device", Some(""));

        if let Err(err) = validate_meta_types(&object) {
            report
                .blockers
                .push(format!("{}: {}", meta_path.display(), err));
            continue;
        }
        collect_consistency_values(&object, &mut hashes, &mut sigs, &mut dtypes, &mut devices);

        if verify_only {
            if changed {
                let issue = format!(
                    "{} missing v1 metadata keys (run checkpoint-meta-v1 migration)",
                    meta_path.display()
                );
                if strict_contracts {
                    report.blockers.push(issue);
                } else {
                    report.warnings.push(issue);
                }
            }
            continue;
        }
        if changed {
            report.updated_files += 1;
            if !dry_run {
                let text = serde_json::to_string_pretty(&Value::Object(object))
                    .map_err(|e| e.to_string())?;
                fs::write(&meta_path, text)
                    .map_err(|err| format!("write {}: {}", meta_path.display(), err))?;
            }
        }
    }

    if hashes.len() > 1 {
        report
            .blockers
            .push("checkpoint config_hash mismatch across checkpoint tree".to_string());
    }
    if sigs.len() > 1 {
        report
            .blockers
            .push("checkpoint model_sig mismatch across checkpoint tree".to_string());
    }
    if dtypes.len() > 1 {
        report
            .blockers
            .push("checkpoint dtype mismatch across checkpoint tree".to_string());
    }
    if devices.len() > 1 {
        report
            .blockers
            .push("checkpoint device mismatch across checkpoint tree".to_string());
    }
    Ok(report)
}

fn ensure_u64_field(map: &mut Map<String, Value>, key: &str, default: Option<u64>) -> bool {
    match map.get(key) {
        Some(Value::Number(n)) if n.as_u64().is_some() => false,
        Some(_) => false,
        None => {
            if let Some(value) = default {
                map.insert(key.to_string(), Value::from(value));
                true
            } else {
                false
            }
        }
    }
}

fn ensure_f64_field(map: &mut Map<String, Value>, key: &str, default: Option<f64>) -> bool {
    match map.get(key) {
        Some(Value::Number(n)) if n.as_f64().is_some() => false,
        Some(_) => false,
        None => {
            if let Some(value) = default {
                map.insert(key.to_string(), Value::from(value));
                true
            } else {
                false
            }
        }
    }
}

fn ensure_string_field(map: &mut Map<String, Value>, key: &str, default: Option<&str>) -> bool {
    match map.get(key) {
        Some(Value::String(_)) => false,
        Some(_) => false,
        None => {
            if let Some(value) = default {
                map.insert(key.to_string(), Value::from(value));
                true
            } else {
                false
            }
        }
    }
}

fn validate_meta_types(map: &Map<String, Value>) -> Result<(), String> {
    expect_u64(map, "format_version")?;
    expect_u64(map, "step")?;
    expect_u64(map, "tokens")?;
    expect_f64(map, "loss")?;
    expect_string(map, "config_hash")?;
    expect_string(map, "model_sig")?;
    expect_string(map, "dtype")?;
    expect_string(map, "device")?;
    Ok(())
}

fn expect_u64(map: &Map<String, Value>, key: &str) -> Result<(), String> {
    match map.get(key) {
        Some(Value::Number(n)) if n.as_u64().is_some() => Ok(()),
        Some(_) => Err(format!("{} must be unsigned integer", key)),
        None => Err(format!("{} missing", key)),
    }
}

fn expect_f64(map: &Map<String, Value>, key: &str) -> Result<(), String> {
    match map.get(key) {
        Some(Value::Number(n)) if n.as_f64().is_some() => Ok(()),
        Some(_) => Err(format!("{} must be number", key)),
        None => Err(format!("{} missing", key)),
    }
}

fn expect_string(map: &Map<String, Value>, key: &str) -> Result<(), String> {
    match map.get(key) {
        Some(Value::String(_)) => Ok(()),
        Some(_) => Err(format!("{} must be string", key)),
        None => Err(format!("{} missing", key)),
    }
}

fn collect_consistency_values(
    map: &Map<String, Value>,
    hashes: &mut HashSet<String>,
    sigs: &mut HashSet<String>,
    dtypes: &mut HashSet<String>,
    devices: &mut HashSet<String>,
) {
    if let Some(Value::String(v)) = map.get("config_hash") {
        if !v.trim().is_empty() {
            hashes.insert(v.clone());
        }
    }
    if let Some(Value::String(v)) = map.get("model_sig") {
        if !v.trim().is_empty() {
            sigs.insert(v.clone());
        }
    }
    if let Some(Value::String(v)) = map.get("dtype") {
        if !v.trim().is_empty() {
            dtypes.insert(v.clone());
        }
    }
    if let Some(Value::String(v)) = map.get("device") {
        if !v.trim().is_empty() {
            devices.insert(v.clone());
        }
    }
}

fn infer_step_from_path(path: &Path) -> Option<u64> {
    for ancestor in path.ancestors() {
        let name = ancestor.file_name()?.to_string_lossy();
        if let Some(step) = name.strip_prefix("step_") {
            if let Ok(parsed) = step.parse::<u64>() {
                return Some(parsed);
            }
        }
    }
    None
}

fn collect_checkpoint_meta_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    collect_checkpoint_meta_files_rec(root, &mut out)?;
    out.sort();
    out.dedup();
    Ok(out)
}

fn collect_checkpoint_meta_files_rec(root: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if root.is_file() {
        return Ok(());
    }
    for entry in fs::read_dir(root).map_err(|err| format!("read {}: {}", root.display(), err))? {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_checkpoint_meta_files_rec(&path, out)?;
            continue;
        }
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.eq_ignore_ascii_case("meta.json"))
            .unwrap_or(false)
            && infer_step_from_path(&path).is_some()
        {
            out.push(path);
        }
    }
    Ok(())
}

fn collect_checkpoint_roots(root: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut meta_files = Vec::new();
    if collect_checkpoint_meta_files_rec(root, &mut meta_files).is_err() {
        return roots;
    }
    for meta in meta_files {
        for ancestor in meta.ancestors() {
            let Some(name) = ancestor.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.starts_with("step_") {
                if let Some(parent) = ancestor.parent() {
                    roots.push(parent.to_path_buf());
                }
                break;
            }
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

fn collect_candidate_config_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut out = Vec::new();
    collect_candidate_config_files_rec(path, &mut out);
    out.sort();
    out.dedup();
    out
}

fn collect_candidate_config_files_rec(path: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let file = entry.path();
        if file.is_dir() {
            collect_candidate_config_files_rec(&file, out);
            continue;
        }
        if !is_source_extension(&file) {
            continue;
        }
        let name = file
            .file_name()
            .and_then(|v| v.to_str())
            .map(|v| v.to_ascii_lowercase())
            .unwrap_or_default();
        if name.contains("config") {
            out.push(file);
        }
    }
}

fn is_source_extension(path: &Path) -> bool {
    let ext = path.extension().and_then(|ext| ext.to_str());
    matches!(ext, Some("enk") | Some("en") | Some("enkai"))
}

fn looks_like_train_config(value: &Value) -> bool {
    let Some(map) = value.as_object() else {
        return false;
    };
    map.contains_key("checkpoint_dir")
        || map.contains_key("dataset_path")
        || map.contains_key("tokenizer_path")
        || map.contains_key("tokenizer_train")
}

#[cfg(all(test, not(windows)))]
mod tests {
    use super::*;

    fn write_legacy_config(path: &Path) {
        let config = serde_json::json!({
            "backend":"cpu",
            "vocab_size":8,
            "hidden_size":4,
            "seq_len":4,
            "batch_size":2,
            "lr":0.1,
            "dataset_path":"data.txt",
            "checkpoint_dir":"ckpt",
            "max_steps":2,
            "tokenizer_train":{"path":"data.txt","vocab_size":8}
        })
        .to_string();
        let escaped = config.replace('\\', "\\\\").replace('\"', "\\\"");
        let source = format!("fn main() ::\n    return json.parse(\"{}\")\n::\n", escaped);
        fs::write(path, source).expect("write config");
    }

    #[test]
    fn migrate_config_v1_generates_config_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("legacy_config.enk");
        let output = dir.path().join("migrated_config.enk");
        write_legacy_config(&input);
        let code = migrate_command(&[
            "config-v1".to_string(),
            input.to_string_lossy().to_string(),
            output.to_string_lossy().to_string(),
        ]);
        assert_eq!(code, EXIT_OK);
        let json = train::load_config_json(&output).expect("load migrated");
        assert_eq!(json.get("config_version").and_then(|v| v.as_u64()), Some(1));
    }

    #[test]
    fn migrate_checkpoint_meta_v1_fills_required_fields() {
        let dir = tempfile::tempdir().expect("tempdir");
        let step = dir.path().join("step_00000001");
        fs::create_dir_all(&step).expect("mkdir step");
        fs::write(step.join("meta.json"), r#"{"step":1}"#).expect("write meta");
        let code = migrate_command(&[
            "checkpoint-meta-v1".to_string(),
            dir.path().to_string_lossy().to_string(),
        ]);
        assert_eq!(code, EXIT_OK);
        let text = fs::read_to_string(step.join("meta.json")).expect("read meta");
        let json: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(json.get("format_version").and_then(|v| v.as_u64()), Some(1));
        assert!(json.get("config_hash").is_some());
    }

    #[test]
    fn doctor_reports_blockers_for_legacy_artifacts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = dir.path().join("train_config.enk");
        write_legacy_config(&config);
        let step = dir.path().join("checkpoints").join("step_00000001");
        fs::create_dir_all(&step).expect("mkdir step");
        fs::write(step.join("meta.json"), r#"{"step":1}"#).expect("write meta");
        let code = doctor_command(&[dir.path().to_string_lossy().to_string()]);
        assert_eq!(code, EXIT_BLOCKED);
    }

    #[test]
    fn strict_checkpoint_verify_blocks_missing_v1_fields() {
        let dir = tempfile::tempdir().expect("tempdir");
        let step = dir.path().join("step_00000001");
        fs::create_dir_all(&step).expect("mkdir step");
        fs::write(step.join("meta.json"), r#"{"step":1}"#).expect("write meta");
        let code = migrate_command(&[
            "checkpoint-meta-v1".to_string(),
            dir.path().to_string_lossy().to_string(),
            "--verify".to_string(),
            "--strict-contracts".to_string(),
        ]);
        assert_eq!(code, EXIT_BLOCKED);
    }

    #[test]
    fn lenient_doctor_allows_missing_checkpoint_fields_as_warning() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = dir.path().join("train_config.enk");
        let json = serde_json::json!({
            "config_version":1,
            "backend":"cpu",
            "vocab_size":8,
            "hidden_size":4,
            "seq_len":4,
            "batch_size":2,
            "lr":0.1,
            "dataset_path":"data.txt",
            "checkpoint_dir":"checkpoints",
            "max_steps":2,
            "tokenizer_train":{"path":"data.txt","vocab_size":8}
        })
        .to_string();
        let escaped = json.replace('\\', "\\\\").replace('\"', "\\\"");
        fs::write(
            &config,
            format!("fn main() ::\n    return json.parse(\"{}\")\n::\n", escaped),
        )
        .expect("config");
        let step = dir.path().join("checkpoints").join("step_00000001");
        fs::create_dir_all(&step).expect("mkdir step");
        fs::write(step.join("meta.json"), r#"{"step":1}"#).expect("write meta");
        let strict = doctor_command(&[dir.path().to_string_lossy().to_string()]);
        assert_eq!(strict, EXIT_BLOCKED);
        let lenient = doctor_command(&[
            dir.path().to_string_lossy().to_string(),
            "--lenient".to_string(),
        ]);
        assert_eq!(lenient, EXIT_OK);
    }
}
