use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const REMOTE_MANIFEST_FILE: &str = "remote.manifest.json";
const REMOTE_SIGNATURE_FILE: &str = "remote.manifest.sig";
const AUDIT_LOG_FILE: &str = "audit.log.jsonl";
const SIGNING_KEY_ENV: &str = "ENKAI_MODEL_SIGNING_KEY";
const ARTIFACT_POINTER_FILE: &str = "artifact_path.txt";
const CHECKPOINT_POINTER_FILE: &str = "checkpoint_path.txt";
const MODEL_MANIFEST_FILE: &str = "model.meta.json";
const DEFAULT_ARTIFACT_KIND: &str = "checkpoint";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ModelRegistry {
    schema_version: u32,
    models: BTreeMap<String, ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ModelEntry {
    active: Option<String>,
    versions: BTreeMap<String, ModelVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelVersion {
    status: String,
    checkpoint_path: String,
    #[serde(default = "default_artifact_kind")]
    artifact_kind: String,
    #[serde(default)]
    artifact_manifest_path: Option<String>,
    #[serde(default)]
    lineage_manifest_path: Option<String>,
    created_ms: u64,
    updated_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ServeState {
    schema_version: u32,
    loaded: BTreeMap<String, BTreeMap<String, ServeLoadedVersion>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServeLoadedVersion {
    checkpoint_path: String,
    loaded_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RemoteArtifactManifest {
    schema_version: u32,
    name: String,
    version: String,
    status: String,
    checkpoint_path: String,
    artifact_kind: String,
    artifact_manifest_path: Option<String>,
    lineage_manifest_path: Option<String>,
    source_registry: String,
    pushed_ms: u64,
    artifact_digest: String,
    files: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegistryAuditEvent {
    schema_version: u32,
    timestamp_ms: u64,
    operation: String,
    status: String,
    model: String,
    version: String,
    registry_path: String,
    remote_registry: Option<String>,
    code: Option<String>,
    detail: Option<String>,
}

#[derive(Debug, Clone)]
struct ModelOpError {
    code: &'static str,
    message: String,
}

impl ModelOpError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone)]
struct RemoteCommandArgs {
    local_registry: PathBuf,
    remote_registry: PathBuf,
    model: String,
    version: String,
    verify_signature: bool,
    fallback_local: bool,
    sign: bool,
}

#[derive(Debug, Clone)]
struct RegisterCommandArgs {
    registry_dir: PathBuf,
    name: String,
    version: String,
    artifact_path: PathBuf,
    activate: bool,
    artifact_kind: String,
    artifact_manifest_path: Option<PathBuf>,
    lineage_manifest_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct AuditAppend<'a> {
    operation: &'a str,
    status: &'a str,
    model: &'a str,
    version: &'a str,
    remote_registry: Option<&'a Path>,
    code: Option<&'a str>,
    detail: Option<&'a str>,
}

pub fn model_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_model_usage();
        return 1;
    }
    match args[0].as_str() {
        "register" => model_register(&args[1..]),
        "list" => model_list(&args[1..]),
        "load" => model_load(&args[1..]),
        "unload" => model_unload(&args[1..]),
        "loaded" => model_loaded(&args[1..]),
        "push" => model_push_remote(&args[1..]),
        "pull" => model_pull_remote(&args[1..]),
        "verify-signature" => model_verify_signature(&args[1..]),
        "promote-remote" => model_sync_remote_state("promote", &args[1..]),
        "retire-remote" => model_sync_remote_state("retire", &args[1..]),
        "rollback-remote" => model_sync_remote_state("rollback", &args[1..]),
        "promote" => model_promote_like("promote", &args[1..]),
        "retire" => model_retire(&args[1..]),
        "rollback" => model_promote_like("rollback", &args[1..]),
        _ => {
            eprintln!("enkai model: unknown subcommand '{}'", args[0]);
            print_model_usage();
            1
        }
    }
}

pub fn print_model_usage() {
    eprintln!(
        "  enkai model register <registry_dir> <name> <version> <artifact_path> [--activate] [--artifact-kind <checkpoint|simulation|environment|native-extension>] [--artifact-manifest <file>] [--lineage-manifest <file>]"
    );
    eprintln!("  enkai model list <registry_dir> [name] [--json]");
    eprintln!("  enkai model load <registry_dir> <name> <version>");
    eprintln!("  enkai model unload <registry_dir> <name> <version>");
    eprintln!("  enkai model loaded <registry_dir> [name] [--json]");
    eprintln!(
        "  enkai model push <registry_dir> <name> <version> --registry <remote_registry_dir> [--sign]"
    );
    eprintln!(
        "  enkai model pull <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]"
    );
    eprintln!(
        "  enkai model verify-signature <registry_dir> <name> <version> --registry <remote_registry_dir>"
    );
    eprintln!(
        "  enkai model promote-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]"
    );
    eprintln!(
        "  enkai model retire-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]"
    );
    eprintln!(
        "  enkai model rollback-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]"
    );
    eprintln!("  enkai model promote <registry_dir> <name> <version>");
    eprintln!("  enkai model retire <registry_dir> <name> <version>");
    eprintln!("  enkai model rollback <registry_dir> <name> <version>");
}

fn default_artifact_kind() -> String {
    DEFAULT_ARTIFACT_KIND.to_string()
}

fn model_register(args: &[String]) -> i32 {
    let parsed = match parse_register_args(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai model register: {}", err);
            return 1;
        }
    };
    let checkpoint_path = match fs::canonicalize(&parsed.artifact_path) {
        Ok(path) => path,
        Err(err) => {
            eprintln!(
                "enkai model register: artifact path {} is invalid: {}",
                parsed.artifact_path.display(),
                err
            );
            return 1;
        }
    };
    if !checkpoint_path.exists() {
        eprintln!(
            "enkai model register: artifact path not found: {}",
            checkpoint_path.display()
        );
        return 1;
    }
    let artifact_manifest_path =
        match canonicalize_optional_file(parsed.artifact_manifest_path.as_deref()) {
            Ok(path) => path,
            Err(err) => {
                eprintln!("enkai model register: {}", err);
                return 1;
            }
        };
    let lineage_manifest_path =
        match canonicalize_optional_file(parsed.lineage_manifest_path.as_deref()) {
            Ok(path) => path,
            Err(err) => {
                eprintln!("enkai model register: {}", err);
                return 1;
            }
        };
    if let Err(err) = fs::create_dir_all(&parsed.registry_dir) {
        eprintln!(
            "enkai model register: failed to create registry dir {}: {}",
            parsed.registry_dir.display(),
            err
        );
        return 1;
    }
    let mut registry = match load_registry(&parsed.registry_dir) {
        Ok(registry) => registry,
        Err(err) => {
            eprintln!("enkai model register: {}", err);
            return 1;
        }
    };
    let model_dir = parsed.registry_dir.join(&parsed.name).join(&parsed.version);
    if let Err(err) = fs::create_dir_all(&model_dir) {
        eprintln!(
            "enkai model register: failed to create model version dir {}: {}",
            model_dir.display(),
            err
        );
        return 1;
    }
    let now = now_ms();
    let checkpoint_text = checkpoint_path.to_string_lossy().to_string();
    let artifact_manifest_text = artifact_manifest_path
        .as_ref()
        .map(|path| path.to_string_lossy().to_string());
    let lineage_manifest_text = lineage_manifest_path
        .as_ref()
        .map(|path| path.to_string_lossy().to_string());
    let entry = registry.models.entry(parsed.name.clone()).or_default();
    let existing_created = entry
        .versions
        .get(&parsed.version)
        .map(|existing| existing.created_ms)
        .unwrap_or(now);
    entry.versions.insert(
        parsed.version.clone(),
        ModelVersion {
            status: "registered".to_string(),
            checkpoint_path: checkpoint_text.clone(),
            artifact_kind: parsed.artifact_kind.clone(),
            artifact_manifest_path: materialize_registry_manifest_path(
                &model_dir,
                artifact_manifest_text.as_deref(),
                "artifact.manifest.json",
            ),
            lineage_manifest_path: materialize_registry_manifest_path(
                &model_dir,
                lineage_manifest_text.as_deref(),
                "lineage.manifest.json",
            ),
            created_ms: existing_created,
            updated_ms: now,
        },
    );
    if parsed.activate || entry.active.is_none() {
        entry.active = Some(parsed.version.clone());
        set_version_status(entry, &parsed.version, "active");
    }
    if let Err(err) = write_artifact_pointer(&model_dir, &checkpoint_text) {
        eprintln!("enkai model register: {}", err);
        return 1;
    }
    if let Some(source) = artifact_manifest_path.as_ref() {
        let destination = model_dir.join(file_name_or("artifact.manifest.json", source));
        if let Err(err) = fs::copy(source, &destination) {
            eprintln!(
                "enkai model register: failed to copy artifact manifest {} -> {}: {}",
                source.display(),
                destination.display(),
                err
            );
            return 1;
        }
    }
    if let Some(source) = lineage_manifest_path.as_ref() {
        let destination = model_dir.join(file_name_or("lineage.manifest.json", source));
        if let Err(err) = fs::copy(source, &destination) {
            eprintln!(
                "enkai model register: failed to copy lineage manifest {} -> {}: {}",
                source.display(),
                destination.display(),
                err
            );
            return 1;
        }
    }
    if let Err(err) = write_model_manifest(&model_dir, &parsed.name, &parsed.version, entry) {
        eprintln!("enkai model register: {}", err);
        return 1;
    }
    if let Some(active) = &entry.active {
        if let Err(err) = fs::write(
            parsed
                .registry_dir
                .join(&parsed.name)
                .join(".active_version"),
            active,
        ) {
            eprintln!(
                "enkai model register: failed to write active version pointer: {}",
                err
            );
            return 1;
        }
    }
    if let Err(err) = save_registry(&parsed.registry_dir, &registry) {
        eprintln!("enkai model register: {}", err);
        return 1;
    }
    let _ = append_audit_event(
        &parsed.registry_dir,
        AuditAppend {
            operation: "register",
            status: "ok",
            model: &parsed.name,
            version: &parsed.version,
            remote_registry: None,
            code: None,
            detail: None,
        },
    );
    println!(
        "registered model {} {} (kind: {}, artifact: {})",
        parsed.name,
        parsed.version,
        parsed.artifact_kind,
        checkpoint_path.display()
    );
    0
}

fn parse_register_args(args: &[String]) -> Result<RegisterCommandArgs, String> {
    if args.len() < 4 {
        return Err("Usage: enkai model register <registry_dir> <name> <version> <artifact_path> [--activate] [--artifact-kind <checkpoint|simulation|environment|native-extension>] [--artifact-manifest <file>] [--lineage-manifest <file>]".to_string());
    }
    let registry_dir = PathBuf::from(&args[0]);
    let name = args[1].trim().to_string();
    let version = args[2].trim().to_string();
    let artifact_path = PathBuf::from(&args[3]);
    if name.is_empty() || version.is_empty() {
        return Err("name/version cannot be empty".to_string());
    }
    let mut activate = false;
    let mut artifact_kind = default_artifact_kind();
    let mut artifact_manifest_path: Option<PathBuf> = None;
    let mut lineage_manifest_path: Option<PathBuf> = None;
    let mut idx = 4usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--activate" => activate = true,
            "--artifact-kind" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--artifact-kind requires a value".to_string());
                }
                artifact_kind = normalize_artifact_kind(&args[idx])?;
            }
            "--artifact-manifest" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--artifact-manifest requires a value".to_string());
                }
                artifact_manifest_path = Some(PathBuf::from(&args[idx]));
            }
            "--lineage-manifest" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--lineage-manifest requires a value".to_string());
                }
                lineage_manifest_path = Some(PathBuf::from(&args[idx]));
            }
            other => return Err(format!("unknown option '{}'", other)),
        }
        idx += 1;
    }
    Ok(RegisterCommandArgs {
        registry_dir,
        name,
        version,
        artifact_path,
        activate,
        artifact_kind,
        artifact_manifest_path,
        lineage_manifest_path,
    })
}

fn model_list(args: &[String]) -> i32 {
    if args.is_empty() || args.len() > 3 {
        eprintln!("Usage: enkai model list <registry_dir> [name] [--json]");
        return 1;
    }
    let registry_dir = PathBuf::from(&args[0]);
    let mut name: Option<&str> = None;
    let mut as_json = false;
    for arg in &args[1..] {
        if arg == "--json" {
            as_json = true;
        } else if name.is_none() {
            name = Some(arg.as_str());
        } else {
            eprintln!("enkai model list: unexpected argument '{}'", arg);
            return 1;
        }
    }
    let registry = match load_registry(&registry_dir) {
        Ok(registry) => registry,
        Err(err) => {
            eprintln!("enkai model list: {}", err);
            return 1;
        }
    };
    if as_json {
        if let Some(name) = name {
            match registry.models.get(name) {
                Some(model) => match serde_json::to_string_pretty(model) {
                    Ok(text) => println!("{}", text),
                    Err(err) => {
                        eprintln!("enkai model list: failed to serialize JSON: {}", err);
                        return 1;
                    }
                },
                None => {
                    eprintln!("enkai model list: model '{}' not found", name);
                    return 1;
                }
            }
        } else {
            match serde_json::to_string_pretty(&registry) {
                Ok(text) => println!("{}", text),
                Err(err) => {
                    eprintln!("enkai model list: failed to serialize JSON: {}", err);
                    return 1;
                }
            }
        }
        return 0;
    }
    if let Some(name) = name {
        let Some(model) = registry.models.get(name) else {
            eprintln!("enkai model list: model '{}' not found", name);
            return 1;
        };
        print_model_entry(name, model);
        return 0;
    }
    if registry.models.is_empty() {
        println!("(no models registered)");
        return 0;
    }
    for (model_name, model) in &registry.models {
        print_model_entry(model_name, model);
    }
    0
}

fn normalize_artifact_kind(raw: &str) -> Result<String, String> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "checkpoint" | "simulation" | "environment" | "native-extension" => Ok(normalized),
        _ => Err(format!(
            "unsupported --artifact-kind '{}'; expected checkpoint|simulation|environment|native-extension",
            raw
        )),
    }
}

fn canonicalize_optional_file(path: Option<&Path>) -> Result<Option<PathBuf>, String> {
    match path {
        Some(path) => fs::canonicalize(path)
            .map(Some)
            .map_err(|err| format!("invalid metadata file {}: {}", path.display(), err)),
        None => Ok(None),
    }
}

fn file_name_or<'a>(fallback: &'a str, path: &'a Path) -> &'a std::ffi::OsStr {
    path.file_name().unwrap_or_else(|| fallback.as_ref())
}

fn model_promote_like(kind: &str, args: &[String]) -> i32 {
    if args.len() != 3 {
        eprintln!(
            "Usage: enkai model {} <registry_dir> <name> <version>",
            kind
        );
        return 1;
    }
    let registry_dir = PathBuf::from(&args[0]);
    let name = args[1].trim();
    let version = args[2].trim();
    if name.is_empty() || version.is_empty() {
        eprintln!("enkai model {}: name/version cannot be empty", kind);
        return 1;
    }
    let mut registry = match load_registry(&registry_dir) {
        Ok(registry) => registry,
        Err(err) => {
            eprintln!("enkai model {}: {}", kind, err);
            return 1;
        }
    };
    let Some(entry) = registry.models.get_mut(name) else {
        eprintln!("enkai model {}: model '{}' not found", kind, name);
        return 1;
    };
    if !entry.versions.contains_key(version) {
        eprintln!(
            "enkai model {}: model '{}' version '{}' not found",
            kind, name, version
        );
        return 1;
    }
    entry.active = Some(version.to_string());
    set_version_status(entry, version, "active");
    if let Err(err) = fs::write(registry_dir.join(name).join(".active_version"), version) {
        eprintln!(
            "enkai model {}: failed to write active version pointer: {}",
            kind, err
        );
        return 1;
    }
    if let Err(err) = write_model_manifest_for_version(&registry_dir, name, version, entry) {
        eprintln!("enkai model {}: {}", kind, err);
        return 1;
    }
    if let Err(err) = save_registry(&registry_dir, &registry) {
        eprintln!("enkai model {}: {}", kind, err);
        return 1;
    }
    let _ = append_audit_event(
        &registry_dir,
        AuditAppend {
            operation: kind,
            status: "ok",
            model: name,
            version,
            remote_registry: None,
            code: None,
            detail: None,
        },
    );
    println!("{}d model {} {}", kind, name, version);
    0
}

fn model_load(args: &[String]) -> i32 {
    if args.len() != 3 {
        eprintln!("Usage: enkai model load <registry_dir> <name> <version>");
        return 1;
    }
    let registry_dir = PathBuf::from(&args[0]);
    let name = args[1].trim();
    let version = args[2].trim();
    if name.is_empty() || version.is_empty() {
        eprintln!("enkai model load: name/version cannot be empty");
        return 1;
    }
    let registry = match load_registry(&registry_dir) {
        Ok(registry) => registry,
        Err(err) => {
            eprintln!("enkai model load: {}", err);
            return 1;
        }
    };
    let Some(model_entry) = registry.models.get(name) else {
        eprintln!("enkai model load: model '{}' not found", name);
        return 1;
    };
    let Some(version_entry) = model_entry.versions.get(version) else {
        eprintln!(
            "enkai model load: model '{}' version '{}' not found",
            name, version
        );
        return 1;
    };
    if version_entry.artifact_kind != DEFAULT_ARTIFACT_KIND {
        eprintln!(
            "enkai model load: model '{}' version '{}' is artifact kind '{}' and cannot be loaded for serve",
            name, version, version_entry.artifact_kind
        );
        return 1;
    }
    if version_entry.status == "retired" {
        eprintln!(
            "enkai model load: model '{}' version '{}' is retired",
            name, version
        );
        return 1;
    }
    let mut serve_state = match load_serve_state(&registry_dir) {
        Ok(state) => state,
        Err(err) => {
            eprintln!("enkai model load: {}", err);
            return 1;
        }
    };
    serve_state
        .loaded
        .entry(name.to_string())
        .or_default()
        .insert(
            version.to_string(),
            ServeLoadedVersion {
                checkpoint_path: version_entry.checkpoint_path.clone(),
                loaded_at_ms: now_ms(),
            },
        );
    if let Err(err) = save_serve_state(&registry_dir, &serve_state) {
        eprintln!("enkai model load: {}", err);
        return 1;
    }
    let _ = append_audit_event(
        &registry_dir,
        AuditAppend {
            operation: "load",
            status: "ok",
            model: name,
            version,
            remote_registry: None,
            code: None,
            detail: None,
        },
    );
    println!("loaded model {} {}", name, version);
    0
}

fn model_unload(args: &[String]) -> i32 {
    if args.len() != 3 {
        eprintln!("Usage: enkai model unload <registry_dir> <name> <version>");
        return 1;
    }
    let registry_dir = PathBuf::from(&args[0]);
    let name = args[1].trim();
    let version = args[2].trim();
    if name.is_empty() || version.is_empty() {
        eprintln!("enkai model unload: name/version cannot be empty");
        return 1;
    }
    let mut serve_state = match load_serve_state(&registry_dir) {
        Ok(state) => state,
        Err(err) => {
            eprintln!("enkai model unload: {}", err);
            return 1;
        }
    };
    let mut unloaded = false;
    if let Some(model_loaded) = serve_state.loaded.get_mut(name) {
        unloaded = model_loaded.remove(version).is_some();
        if model_loaded.is_empty() {
            serve_state.loaded.remove(name);
        }
    }
    if !unloaded {
        eprintln!(
            "enkai model unload: model '{}' version '{}' is not loaded",
            name, version
        );
        return 1;
    }
    if let Err(err) = save_serve_state(&registry_dir, &serve_state) {
        eprintln!("enkai model unload: {}", err);
        return 1;
    }
    let _ = append_audit_event(
        &registry_dir,
        AuditAppend {
            operation: "unload",
            status: "ok",
            model: name,
            version,
            remote_registry: None,
            code: None,
            detail: None,
        },
    );
    println!("unloaded model {} {}", name, version);
    0
}

fn model_loaded(args: &[String]) -> i32 {
    if args.is_empty() || args.len() > 3 {
        eprintln!("Usage: enkai model loaded <registry_dir> [name] [--json]");
        return 1;
    }
    let registry_dir = PathBuf::from(&args[0]);
    let mut name: Option<&str> = None;
    let mut as_json = false;
    for arg in &args[1..] {
        if arg == "--json" {
            as_json = true;
        } else if name.is_none() {
            name = Some(arg.as_str());
        } else {
            eprintln!("enkai model loaded: unexpected argument '{}'", arg);
            return 1;
        }
    }
    let state = match load_serve_state(&registry_dir) {
        Ok(state) => state,
        Err(err) => {
            eprintln!("enkai model loaded: {}", err);
            return 1;
        }
    };
    if as_json {
        if let Some(name) = name {
            let Some(entry) = state.loaded.get(name) else {
                eprintln!(
                    "enkai model loaded: model '{}' has no loaded versions",
                    name
                );
                return 1;
            };
            match serde_json::to_string_pretty(entry) {
                Ok(text) => println!("{}", text),
                Err(err) => {
                    eprintln!("enkai model loaded: failed to serialize JSON: {}", err);
                    return 1;
                }
            }
        } else {
            match serde_json::to_string_pretty(&state) {
                Ok(text) => println!("{}", text),
                Err(err) => {
                    eprintln!("enkai model loaded: failed to serialize JSON: {}", err);
                    return 1;
                }
            }
        }
        return 0;
    }
    if let Some(name) = name {
        let Some(entry) = state.loaded.get(name) else {
            println!("model: {} (no loaded versions)", name);
            return 0;
        };
        println!("model: {}", name);
        for (version, loaded) in entry {
            println!(
                "  - {} checkpoint={} loaded_at_ms={}",
                version, loaded.checkpoint_path, loaded.loaded_at_ms
            );
        }
        return 0;
    }
    if state.loaded.is_empty() {
        println!("(no loaded models)");
        return 0;
    }
    for (model, versions) in state.loaded {
        println!("model: {}", model);
        for (version, loaded) in versions {
            println!(
                "  - {} checkpoint={} loaded_at_ms={}",
                version, loaded.checkpoint_path, loaded.loaded_at_ms
            );
        }
    }
    0
}

fn model_push_remote(args: &[String]) -> i32 {
    let parsed = match parse_remote_args("push", args, true) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai model push: {}", err);
            return 1;
        }
    };
    let result = push_remote_model(&parsed);
    match result {
        Ok(message) => {
            println!("{}", message);
            0
        }
        Err(err) => {
            eprintln!("enkai model push [{}]: {}", err.code, err.message);
            let _ = append_audit_event(
                &parsed.local_registry,
                AuditAppend {
                    operation: "push_remote",
                    status: "failed",
                    model: &parsed.model,
                    version: &parsed.version,
                    remote_registry: Some(&parsed.remote_registry),
                    code: Some(err.code),
                    detail: Some(&err.message),
                },
            );
            1
        }
    }
}

fn model_pull_remote(args: &[String]) -> i32 {
    let parsed = match parse_remote_args("pull", args, false) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai model pull: {}", err);
            return 1;
        }
    };
    match pull_remote_model(&parsed) {
        Ok(message) => {
            println!("{}", message);
            0
        }
        Err(err) => {
            if parsed.fallback_local
                && has_local_model_version(&parsed.local_registry, &parsed.model, &parsed.version)
            {
                let detail = format!(
                    "remote unavailable/invalid ({}: {}); using pinned local cache",
                    err.code, err.message
                );
                let _ = append_audit_event(
                    &parsed.local_registry,
                    AuditAppend {
                        operation: "pull_remote",
                        status: "fallback_local",
                        model: &parsed.model,
                        version: &parsed.version,
                        remote_registry: Some(&parsed.remote_registry),
                        code: Some(err.code),
                        detail: Some(&detail),
                    },
                );
                println!("{}", detail);
                return 0;
            }
            eprintln!("enkai model pull [{}]: {}", err.code, err.message);
            let _ = append_audit_event(
                &parsed.local_registry,
                AuditAppend {
                    operation: "pull_remote",
                    status: "failed",
                    model: &parsed.model,
                    version: &parsed.version,
                    remote_registry: Some(&parsed.remote_registry),
                    code: Some(err.code),
                    detail: Some(&err.message),
                },
            );
            1
        }
    }
}

fn model_verify_signature(args: &[String]) -> i32 {
    let parsed = match parse_remote_args("verify-signature", args, false) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai model verify-signature: {}", err);
            return 1;
        }
    };
    let remote_version_dir = parsed
        .remote_registry
        .join(&parsed.model)
        .join(&parsed.version);
    match maybe_verify_remote_manifest(&remote_version_dir, &parsed.model, &parsed.version, true) {
        Ok(()) => {
            let _ = append_audit_event(
                &parsed.local_registry,
                AuditAppend {
                    operation: "verify_signature",
                    status: "ok",
                    model: &parsed.model,
                    version: &parsed.version,
                    remote_registry: Some(&parsed.remote_registry),
                    code: None,
                    detail: None,
                },
            );
            println!(
                "verified signature for model {} {} in remote registry {}",
                parsed.model,
                parsed.version,
                parsed.remote_registry.display()
            );
            0
        }
        Err(err) => {
            eprintln!(
                "enkai model verify-signature [{}]: {}",
                err.code, err.message
            );
            let _ = append_audit_event(
                &parsed.local_registry,
                AuditAppend {
                    operation: "verify_signature",
                    status: "failed",
                    model: &parsed.model,
                    version: &parsed.version,
                    remote_registry: Some(&parsed.remote_registry),
                    code: Some(err.code),
                    detail: Some(&err.message),
                },
            );
            1
        }
    }
}

fn model_sync_remote_state(kind: &str, args: &[String]) -> i32 {
    let parsed = match parse_remote_args(kind, args, false) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("enkai model {}-remote: {}", kind, err);
            return 1;
        }
    };
    match sync_remote_state_operation(kind, &parsed) {
        Ok(message) => {
            println!("{}", message);
            0
        }
        Err(err) => {
            if parsed.fallback_local
                && has_local_model_version(&parsed.local_registry, &parsed.model, &parsed.version)
            {
                let rc = match kind {
                    "promote" | "rollback" => model_promote_like(
                        kind,
                        &[
                            parsed.local_registry.to_string_lossy().to_string(),
                            parsed.model.clone(),
                            parsed.version.clone(),
                        ],
                    ),
                    "retire" => model_retire(&[
                        parsed.local_registry.to_string_lossy().to_string(),
                        parsed.model.clone(),
                        parsed.version.clone(),
                    ]),
                    _ => 1,
                };
                if rc == 0 {
                    let detail = format!(
                        "remote sync degraded ({}: {}); applied {} to local cache only",
                        err.code, err.message, kind
                    );
                    let operation = format!("{}_remote", kind);
                    let _ = append_audit_event(
                        &parsed.local_registry,
                        AuditAppend {
                            operation: &operation,
                            status: "fallback_local",
                            model: &parsed.model,
                            version: &parsed.version,
                            remote_registry: Some(&parsed.remote_registry),
                            code: Some(err.code),
                            detail: Some(&detail),
                        },
                    );
                    println!("{}", detail);
                    return 0;
                }
            }
            eprintln!(
                "enkai model {}-remote [{}]: {}",
                kind, err.code, err.message
            );
            let operation = format!("{}_remote", kind);
            let _ = append_audit_event(
                &parsed.local_registry,
                AuditAppend {
                    operation: &operation,
                    status: "failed",
                    model: &parsed.model,
                    version: &parsed.version,
                    remote_registry: Some(&parsed.remote_registry),
                    code: Some(err.code),
                    detail: Some(&err.message),
                },
            );
            1
        }
    }
}

fn model_retire(args: &[String]) -> i32 {
    if args.len() != 3 {
        eprintln!("Usage: enkai model retire <registry_dir> <name> <version>");
        return 1;
    }
    let registry_dir = PathBuf::from(&args[0]);
    let name = args[1].trim();
    let version = args[2].trim();
    if name.is_empty() || version.is_empty() {
        eprintln!("enkai model retire: name/version cannot be empty");
        return 1;
    }
    let mut registry = match load_registry(&registry_dir) {
        Ok(registry) => registry,
        Err(err) => {
            eprintln!("enkai model retire: {}", err);
            return 1;
        }
    };
    let Some(entry) = registry.models.get_mut(name) else {
        eprintln!("enkai model retire: model '{}' not found", name);
        return 1;
    };
    let Some(version_entry) = entry.versions.get_mut(version) else {
        eprintln!(
            "enkai model retire: model '{}' version '{}' not found",
            name, version
        );
        return 1;
    };
    version_entry.status = "retired".to_string();
    version_entry.updated_ms = now_ms();
    if entry.active.as_deref() == Some(version) {
        entry.active = None;
        let active_pointer = registry_dir.join(name).join(".active_version");
        let _ = fs::remove_file(active_pointer);
    }
    if let Err(err) = write_model_manifest_for_version(&registry_dir, name, version, entry) {
        eprintln!("enkai model retire: {}", err);
        return 1;
    }
    if let Err(err) = save_registry(&registry_dir, &registry) {
        eprintln!("enkai model retire: {}", err);
        return 1;
    }
    let _ = unload_model_version(&registry_dir, name, version);
    let _ = append_audit_event(
        &registry_dir,
        AuditAppend {
            operation: "retire",
            status: "ok",
            model: name,
            version,
            remote_registry: None,
            code: None,
            detail: None,
        },
    );
    println!("retired model {} {}", name, version);
    0
}

fn parse_remote_args(
    operation: &str,
    args: &[String],
    allow_sign: bool,
) -> Result<RemoteCommandArgs, String> {
    if args.len() < 5 {
        return Err(format!(
            "Usage: enkai model {} <registry_dir> <name> <version> --registry <remote_registry_dir>{}",
            operation,
            if allow_sign {
                " [--sign] [--verify-signature] [--fallback-local]"
            } else {
                " [--verify-signature] [--fallback-local]"
            }
        ));
    }
    let local_registry = PathBuf::from(&args[0]);
    let model = args[1].trim().to_string();
    let version = args[2].trim().to_string();
    if model.is_empty() || version.is_empty() {
        return Err("name/version cannot be empty".to_string());
    }

    let mut remote_registry: Option<PathBuf> = None;
    let mut verify_signature = false;
    let mut fallback_local = false;
    let mut sign = false;

    let mut idx = 3usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--registry" => {
                idx += 1;
                if idx >= args.len() {
                    return Err("--registry requires a value".to_string());
                }
                remote_registry = Some(PathBuf::from(&args[idx]));
            }
            "--verify-signature" => verify_signature = true,
            "--fallback-local" => fallback_local = true,
            "--sign" if allow_sign => sign = true,
            unknown => {
                return Err(format!("unknown option '{}'", unknown));
            }
        }
        idx += 1;
    }

    let remote_registry =
        remote_registry.ok_or_else(|| "missing --registry <remote_registry_dir>".to_string())?;

    Ok(RemoteCommandArgs {
        local_registry,
        remote_registry,
        model,
        version,
        verify_signature,
        fallback_local,
        sign,
    })
}

fn push_remote_model(args: &RemoteCommandArgs) -> Result<String, ModelOpError> {
    let local_registry = load_registry(&args.local_registry).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_LOCAL_REGISTRY",
            format!(
                "failed to load local registry {}: {}",
                args.local_registry.display(),
                err
            ),
        )
    })?;
    let local_entry = local_registry.models.get(&args.model).ok_or_else(|| {
        ModelOpError::new("E_MODEL_NOT_FOUND", "model not found in local registry")
    })?;
    let local_version_meta = local_entry.versions.get(&args.version).ok_or_else(|| {
        ModelOpError::new(
            "E_MODEL_VERSION_NOT_FOUND",
            "model version not found in local registry",
        )
    })?;
    let local_version_dir = args.local_registry.join(&args.model).join(&args.version);
    ensure_model_version_dir(&local_version_dir, "local")?;
    let manifest = build_remote_manifest(
        &args.local_registry,
        &args.model,
        &args.version,
        local_version_meta,
        &local_version_dir,
    )?;
    let remote_version_dir = args.remote_registry.join(&args.model).join(&args.version);
    fs::create_dir_all(&remote_version_dir).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_REMOTE_IO",
            format!(
                "failed to create remote version directory {}: {}",
                remote_version_dir.display(),
                err
            ),
        )
    })?;

    if remote_version_dir.join(REMOTE_MANIFEST_FILE).is_file() {
        let existing = load_remote_manifest(&remote_version_dir)?;
        verify_remote_manifest_integrity(&remote_version_dir, &existing)?;
        if existing.name != args.model || existing.version != args.version {
            return Err(ModelOpError::new(
                "E_MODEL_REMOTE_IMMUTABLE",
                format!(
                    "remote artifact manifest identity mismatch for {} {}",
                    args.model, args.version
                ),
            ));
        }
        if existing.artifact_digest != manifest.artifact_digest {
            return Err(ModelOpError::new(
                "E_MODEL_REMOTE_IMMUTABLE",
                format!(
                    "remote artifact for {} {} already exists with different immutable digest",
                    args.model, args.version
                ),
            ));
        }
    }

    copy_required_version_files(&local_version_dir, &remote_version_dir)?;
    copy_optional_version_manifests(local_version_meta, &local_version_dir, &remote_version_dir)?;

    let mut remote_registry = load_registry(&args.remote_registry).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_REMOTE_REGISTRY",
            format!(
                "failed to load remote registry {}: {}",
                args.remote_registry.display(),
                err
            ),
        )
    })?;
    let remote_entry = remote_registry
        .models
        .entry(args.model.clone())
        .or_default();
    let now = now_ms();
    let created_ms = remote_entry
        .versions
        .get(&args.version)
        .map(|item| item.created_ms)
        .unwrap_or(now);
    remote_entry.versions.insert(
        args.version.clone(),
        ModelVersion {
            status: local_version_meta.status.clone(),
            checkpoint_path: local_version_meta.checkpoint_path.clone(),
            artifact_kind: local_version_meta.artifact_kind.clone(),
            artifact_manifest_path: materialize_registry_manifest_path(
                &remote_version_dir,
                local_version_meta.artifact_manifest_path.as_deref(),
                "artifact.manifest.json",
            ),
            lineage_manifest_path: materialize_registry_manifest_path(
                &remote_version_dir,
                local_version_meta.lineage_manifest_path.as_deref(),
                "lineage.manifest.json",
            ),
            created_ms,
            updated_ms: now,
        },
    );
    if local_entry.active.as_deref() == Some(args.version.as_str()) || remote_entry.active.is_none()
    {
        remote_entry.active = Some(args.version.clone());
        set_version_status(remote_entry, &args.version, "active");
        fs::write(
            args.remote_registry
                .join(&args.model)
                .join(".active_version"),
            &args.version,
        )
        .map_err(|err| {
            ModelOpError::new(
                "E_MODEL_REMOTE_IO",
                format!("failed to write remote active pointer: {}", err),
            )
        })?;
    }
    write_model_manifest_for_version(
        &args.remote_registry,
        &args.model,
        &args.version,
        remote_entry,
    )
    .map_err(|err| ModelOpError::new("E_MODEL_REMOTE_IO", err))?;
    let final_remote_meta = remote_entry.versions.get(&args.version).ok_or_else(|| {
        ModelOpError::new(
            "E_MODEL_REMOTE_IO",
            "remote registry lost version metadata after sync",
        )
    })?;
    let final_manifest = build_remote_manifest(
        &args.remote_registry,
        &args.model,
        &args.version,
        final_remote_meta,
        &remote_version_dir,
    )?;
    write_remote_manifest(&remote_version_dir, &final_manifest, args.sign)?;
    save_registry(&args.remote_registry, &remote_registry)
        .map_err(|err| ModelOpError::new("E_MODEL_REMOTE_IO", err))?;

    let _ = append_audit_event(
        &args.local_registry,
        AuditAppend {
            operation: "push_remote",
            status: "ok",
            model: &args.model,
            version: &args.version,
            remote_registry: Some(&args.remote_registry),
            code: None,
            detail: None,
        },
    );
    let _ = append_audit_event(
        &args.remote_registry,
        AuditAppend {
            operation: "push_remote",
            status: "ok",
            model: &args.model,
            version: &args.version,
            remote_registry: Some(&args.local_registry),
            code: None,
            detail: None,
        },
    );
    Ok(format!(
        "pushed model {} {} to remote registry {}",
        args.model,
        args.version,
        args.remote_registry.display()
    ))
}

fn pull_remote_model(args: &RemoteCommandArgs) -> Result<String, ModelOpError> {
    let remote_version_dir = args.remote_registry.join(&args.model).join(&args.version);
    ensure_model_version_dir(&remote_version_dir, "remote")?;
    maybe_verify_remote_manifest(
        &remote_version_dir,
        &args.model,
        &args.version,
        args.verify_signature,
    )?;

    let remote_registry = load_registry(&args.remote_registry).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_REMOTE_REGISTRY",
            format!(
                "failed to load remote registry {}: {}",
                args.remote_registry.display(),
                err
            ),
        )
    })?;
    let remote_entry = remote_registry.models.get(&args.model).ok_or_else(|| {
        ModelOpError::new("E_MODEL_NOT_FOUND", "model missing in remote registry")
    })?;
    let remote_meta = remote_entry.versions.get(&args.version).ok_or_else(|| {
        ModelOpError::new(
            "E_MODEL_VERSION_NOT_FOUND",
            "model version missing in remote registry",
        )
    })?;
    let local_version_dir = args.local_registry.join(&args.model).join(&args.version);
    fs::create_dir_all(&local_version_dir).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_LOCAL_IO",
            format!(
                "failed to create local version directory {}: {}",
                local_version_dir.display(),
                err
            ),
        )
    })?;
    copy_required_version_files(&remote_version_dir, &local_version_dir)?;
    copy_optional_version_manifests(remote_meta, &remote_version_dir, &local_version_dir)?;
    let _ = copy_optional_file(
        &remote_version_dir.join(REMOTE_MANIFEST_FILE),
        &local_version_dir.join(REMOTE_MANIFEST_FILE),
    );
    let _ = copy_optional_file(
        &remote_version_dir.join(REMOTE_SIGNATURE_FILE),
        &local_version_dir.join(REMOTE_SIGNATURE_FILE),
    );

    let mut local_registry = load_registry(&args.local_registry).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_LOCAL_REGISTRY",
            format!(
                "failed to load local registry {}: {}",
                args.local_registry.display(),
                err
            ),
        )
    })?;
    let entry = local_registry.models.entry(args.model.clone()).or_default();
    let mut local_meta = remote_meta.clone();
    local_meta.artifact_manifest_path = materialize_registry_manifest_path(
        &local_version_dir,
        remote_meta.artifact_manifest_path.as_deref(),
        "artifact.manifest.json",
    );
    local_meta.lineage_manifest_path = materialize_registry_manifest_path(
        &local_version_dir,
        remote_meta.lineage_manifest_path.as_deref(),
        "lineage.manifest.json",
    );
    entry.versions.insert(args.version.clone(), local_meta);
    if entry.active.is_none() && remote_entry.active.as_deref() == Some(args.version.as_str()) {
        entry.active = Some(args.version.clone());
        set_version_status(entry, &args.version, "active");
        fs::write(
            args.local_registry
                .join(&args.model)
                .join(".active_version"),
            &args.version,
        )
        .map_err(|err| {
            ModelOpError::new(
                "E_MODEL_LOCAL_IO",
                format!("failed to write local active pointer: {}", err),
            )
        })?;
    }
    write_model_manifest_for_version(&args.local_registry, &args.model, &args.version, entry)
        .map_err(|err| ModelOpError::new("E_MODEL_LOCAL_IO", err))?;
    save_registry(&args.local_registry, &local_registry)
        .map_err(|err| ModelOpError::new("E_MODEL_LOCAL_IO", err))?;

    let _ = append_audit_event(
        &args.local_registry,
        AuditAppend {
            operation: "pull_remote",
            status: "ok",
            model: &args.model,
            version: &args.version,
            remote_registry: Some(&args.remote_registry),
            code: None,
            detail: None,
        },
    );
    Ok(format!(
        "pulled model {} {} from remote registry {}",
        args.model,
        args.version,
        args.remote_registry.display()
    ))
}

fn sync_remote_state_operation(
    kind: &str,
    args: &RemoteCommandArgs,
) -> Result<String, ModelOpError> {
    let remote_version_dir = args.remote_registry.join(&args.model).join(&args.version);
    ensure_model_version_dir(&remote_version_dir, "remote")?;
    maybe_verify_remote_manifest(
        &remote_version_dir,
        &args.model,
        &args.version,
        args.verify_signature,
    )?;

    let mut remote_registry = load_registry(&args.remote_registry).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_REMOTE_REGISTRY",
            format!(
                "failed to load remote registry {}: {}",
                args.remote_registry.display(),
                err
            ),
        )
    })?;
    let remote_entry = remote_registry.models.get_mut(&args.model).ok_or_else(|| {
        ModelOpError::new("E_MODEL_NOT_FOUND", "model missing in remote registry")
    })?;
    apply_state_transition(
        kind,
        &args.model,
        &args.version,
        remote_entry,
        &args.remote_registry,
    )?;
    let remote_version_dir = args.remote_registry.join(&args.model).join(&args.version);
    let remote_meta = remote_entry.versions.get(&args.version).ok_or_else(|| {
        ModelOpError::new(
            "E_MODEL_REMOTE_IO",
            "remote registry lost version metadata after state transition",
        )
    })?;
    let remote_manifest = build_remote_manifest(
        &args.remote_registry,
        &args.model,
        &args.version,
        remote_meta,
        &remote_version_dir,
    )?;
    let should_sign = remote_version_dir.join(REMOTE_SIGNATURE_FILE).is_file();
    write_remote_manifest(&remote_version_dir, &remote_manifest, should_sign)?;
    save_registry(&args.remote_registry, &remote_registry)
        .map_err(|err| ModelOpError::new("E_MODEL_REMOTE_IO", err))?;

    if !has_local_model_version(&args.local_registry, &args.model, &args.version) {
        pull_remote_model(args)?;
    }
    let mut local_registry = load_registry(&args.local_registry).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_LOCAL_REGISTRY",
            format!(
                "failed to load local registry {}: {}",
                args.local_registry.display(),
                err
            ),
        )
    })?;
    let local_entry = local_registry.models.get_mut(&args.model).ok_or_else(|| {
        ModelOpError::new(
            "E_MODEL_LOCAL_REGISTRY",
            "model missing in local registry after sync pull",
        )
    })?;
    apply_state_transition(
        kind,
        &args.model,
        &args.version,
        local_entry,
        &args.local_registry,
    )?;
    save_registry(&args.local_registry, &local_registry)
        .map_err(|err| ModelOpError::new("E_MODEL_LOCAL_IO", err))?;

    let operation = format!("{}_remote", kind);
    let _ = append_audit_event(
        &args.local_registry,
        AuditAppend {
            operation: &operation,
            status: "ok",
            model: &args.model,
            version: &args.version,
            remote_registry: Some(&args.remote_registry),
            code: None,
            detail: None,
        },
    );
    let _ = append_audit_event(
        &args.remote_registry,
        AuditAppend {
            operation: &operation,
            status: "ok",
            model: &args.model,
            version: &args.version,
            remote_registry: Some(&args.local_registry),
            code: None,
            detail: None,
        },
    );
    Ok(format!(
        "{}d model {} {} on remote {} and local cache {}",
        kind,
        args.model,
        args.version,
        args.remote_registry.display(),
        args.local_registry.display()
    ))
}

fn apply_state_transition(
    kind: &str,
    model: &str,
    version: &str,
    entry: &mut ModelEntry,
    registry_dir: &Path,
) -> Result<(), ModelOpError> {
    if !entry.versions.contains_key(version) {
        return Err(ModelOpError::new(
            "E_MODEL_VERSION_NOT_FOUND",
            format!("model '{}' version '{}' not found", model, version),
        ));
    }
    let now = now_ms();
    match kind {
        "promote" | "rollback" => {
            entry.active = Some(version.to_string());
            set_version_status(entry, version, "active");
            fs::write(registry_dir.join(model).join(".active_version"), version).map_err(
                |err| {
                    ModelOpError::new(
                        "E_MODEL_IO",
                        format!("failed to write active version pointer: {}", err),
                    )
                },
            )?;
        }
        "retire" => {
            if let Some(meta) = entry.versions.get_mut(version) {
                meta.status = "retired".to_string();
                meta.updated_ms = now;
            }
            if entry.active.as_deref() == Some(version) {
                entry.active = None;
                let _ = fs::remove_file(registry_dir.join(model).join(".active_version"));
            }
            let _ = unload_model_version(registry_dir, model, version);
        }
        other => {
            return Err(ModelOpError::new(
                "E_MODEL_UNSUPPORTED_OPERATION",
                format!("unsupported remote state operation '{}'", other),
            ));
        }
    }
    write_model_manifest_for_version(registry_dir, model, version, entry)
        .map_err(|err| ModelOpError::new("E_MODEL_IO", err))?;
    Ok(())
}

fn ensure_model_version_dir(path: &Path, label: &str) -> Result<(), ModelOpError> {
    if !path.is_dir() {
        return Err(ModelOpError::new(
            "E_MODEL_VERSION_DIR_MISSING",
            format!(
                "{} model version directory not found: {}",
                label,
                path.display()
            ),
        ));
    }
    let manifest = path.join(MODEL_MANIFEST_FILE);
    if !manifest.is_file() {
        return Err(ModelOpError::new(
            "E_MODEL_VERSION_FILE_MISSING",
            format!("{} model artifact missing: {}", label, manifest.display()),
        ));
    }
    if resolve_artifact_pointer(path).is_none() {
        return Err(ModelOpError::new(
            "E_MODEL_VERSION_FILE_MISSING",
            format!(
                "{} model artifact missing: {} or {}",
                label,
                path.join(ARTIFACT_POINTER_FILE).display(),
                path.join(CHECKPOINT_POINTER_FILE).display()
            ),
        ));
    }
    Ok(())
}

fn copy_required_version_files(src: &Path, dst: &Path) -> Result<(), ModelOpError> {
    fs::create_dir_all(dst).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_IO",
            format!("failed to create {}: {}", dst.display(), err),
        )
    })?;
    for file in [
        MODEL_MANIFEST_FILE,
        CHECKPOINT_POINTER_FILE,
        ARTIFACT_POINTER_FILE,
    ] {
        let src_path = src.join(file);
        if !src_path.is_file() {
            if file == ARTIFACT_POINTER_FILE {
                continue;
            }
            return Err(ModelOpError::new(
                "E_MODEL_IO",
                format!("missing required model artifact {}", src_path.display()),
            ));
        }
        let dst_path = dst.join(file);
        fs::copy(&src_path, &dst_path).map_err(|err| {
            ModelOpError::new(
                "E_MODEL_IO",
                format!(
                    "failed to copy model artifact {} -> {}: {}",
                    src_path.display(),
                    dst_path.display(),
                    err
                ),
            )
        })?;
    }
    Ok(())
}

fn copy_optional_file(src: &Path, dst: &Path) -> Result<(), ModelOpError> {
    if !src.is_file() {
        return Ok(());
    }
    fs::copy(src, dst).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_IO",
            format!(
                "failed to copy optional artifact {} -> {}: {}",
                src.display(),
                dst.display(),
                err
            ),
        )
    })?;
    Ok(())
}

fn copy_optional_version_manifests(
    meta: &ModelVersion,
    src_dir: &Path,
    dst_dir: &Path,
) -> Result<(), ModelOpError> {
    for (raw_path, fallback) in [
        (
            meta.artifact_manifest_path.as_deref(),
            "artifact.manifest.json",
        ),
        (
            meta.lineage_manifest_path.as_deref(),
            "lineage.manifest.json",
        ),
    ] {
        let Some(raw_path) = raw_path else {
            continue;
        };
        let file_name = file_name_or(fallback, Path::new(raw_path));
        copy_optional_file(&src_dir.join(file_name), &dst_dir.join(file_name))?;
    }
    Ok(())
}

fn build_remote_manifest(
    source_registry: &Path,
    model: &str,
    version: &str,
    meta: &ModelVersion,
    version_dir: &Path,
) -> Result<RemoteArtifactManifest, ModelOpError> {
    let files = version_files_for_manifest(version_dir, meta)?;
    let artifact_digest = artifact_digest(
        model,
        version,
        &meta.checkpoint_path,
        &meta.artifact_kind,
        &meta.artifact_manifest_path,
        &meta.lineage_manifest_path,
        &files,
    );
    Ok(RemoteArtifactManifest {
        schema_version: 1,
        name: model.to_string(),
        version: version.to_string(),
        status: meta.status.clone(),
        checkpoint_path: meta.checkpoint_path.clone(),
        artifact_kind: meta.artifact_kind.clone(),
        artifact_manifest_path: meta.artifact_manifest_path.clone(),
        lineage_manifest_path: meta.lineage_manifest_path.clone(),
        source_registry: source_registry.to_string_lossy().to_string(),
        pushed_ms: now_ms(),
        artifact_digest,
        files,
    })
}

fn artifact_digest(
    model: &str,
    version: &str,
    checkpoint_path: &str,
    artifact_kind: &str,
    artifact_manifest_path: &Option<String>,
    lineage_manifest_path: &Option<String>,
    files: &BTreeMap<String, String>,
) -> String {
    let payload = serde_json::json!({
        "model": model,
        "version": version,
        "checkpoint_path": checkpoint_path,
        "artifact_kind": artifact_kind,
        "artifact_manifest_path": artifact_manifest_path,
        "lineage_manifest_path": lineage_manifest_path,
        "files": files,
    });
    sha256_hex(payload.to_string().as_bytes())
}

fn maybe_verify_remote_manifest(
    version_dir: &Path,
    model: &str,
    version: &str,
    verify_signature: bool,
) -> Result<(), ModelOpError> {
    let manifest_path = version_dir.join(REMOTE_MANIFEST_FILE);
    if !manifest_path.is_file() {
        if verify_signature {
            verify_remote_manifest_signature(version_dir)?;
        }
        return Ok(());
    }
    if verify_signature {
        verify_remote_manifest_signature(version_dir)?;
    }
    let manifest = load_remote_manifest(version_dir)?;
    if manifest.name != model || manifest.version != version {
        return Err(ModelOpError::new(
            "E_MODEL_MANIFEST",
            format!(
                "manifest identity mismatch: expected {} {} but found {} {}",
                model, version, manifest.name, manifest.version
            ),
        ));
    }
    verify_remote_manifest_integrity(version_dir, &manifest)
}

fn verify_remote_manifest_integrity(
    version_dir: &Path,
    manifest: &RemoteArtifactManifest,
) -> Result<(), ModelOpError> {
    if manifest.schema_version != 1 {
        return Err(ModelOpError::new(
            "E_MODEL_MANIFEST",
            format!(
                "unsupported remote manifest schema_version {} in {}",
                manifest.schema_version,
                version_dir.join(REMOTE_MANIFEST_FILE).display()
            ),
        ));
    }
    for (file, expected) in &manifest.files {
        let actual = file_sha256_hex(&version_dir.join(file))?;
        if actual != *expected {
            return Err(ModelOpError::new(
                "E_MODEL_MANIFEST",
                format!(
                    "remote manifest file hash mismatch for {} (expected {}, got {})",
                    file, expected, actual
                ),
            ));
        }
    }
    let pointer_path = resolve_artifact_pointer_path(version_dir)
        .unwrap_or_else(|| version_dir.join(CHECKPOINT_POINTER_FILE));
    let pointer = fs::read_to_string(&pointer_path).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_MANIFEST",
            format!("failed to read {}: {}", pointer_path.display(), err),
        )
    })?;
    if pointer.trim() != manifest.checkpoint_path.trim() {
        return Err(ModelOpError::new(
            "E_MODEL_MANIFEST",
            format!(
                "remote manifest checkpoint path mismatch for {}",
                version_dir.join(REMOTE_MANIFEST_FILE).display()
            ),
        ));
    }
    let expected_digest = artifact_digest(
        &manifest.name,
        &manifest.version,
        &manifest.checkpoint_path,
        &manifest.artifact_kind,
        &manifest.artifact_manifest_path,
        &manifest.lineage_manifest_path,
        &manifest.files,
    );
    if manifest.artifact_digest != expected_digest {
        return Err(ModelOpError::new(
            "E_MODEL_MANIFEST",
            format!(
                "remote manifest digest mismatch in {}",
                version_dir.join(REMOTE_MANIFEST_FILE).display()
            ),
        ));
    }
    Ok(())
}

fn write_remote_manifest(
    version_dir: &Path,
    manifest: &RemoteArtifactManifest,
    sign: bool,
) -> Result<(), ModelOpError> {
    let text = serde_json::to_string_pretty(manifest).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_MANIFEST",
            format!("failed to serialize remote manifest: {}", err),
        )
    })?;
    fs::write(version_dir.join(REMOTE_MANIFEST_FILE), &text).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_MANIFEST",
            format!(
                "failed to write {}: {}",
                version_dir.join(REMOTE_MANIFEST_FILE).display(),
                err
            ),
        )
    })?;
    if sign {
        let key = read_signing_key()?;
        let signature = sign_manifest(&text, &key);
        fs::write(version_dir.join(REMOTE_SIGNATURE_FILE), signature).map_err(|err| {
            ModelOpError::new(
                "E_MODEL_SIGNATURE",
                format!(
                    "failed to write {}: {}",
                    version_dir.join(REMOTE_SIGNATURE_FILE).display(),
                    err
                ),
            )
        })?;
    }
    Ok(())
}

fn load_remote_manifest(version_dir: &Path) -> Result<RemoteArtifactManifest, ModelOpError> {
    let path = version_dir.join(REMOTE_MANIFEST_FILE);
    let text = fs::read_to_string(&path).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_MANIFEST",
            format!("failed to read {}: {}", path.display(), err),
        )
    })?;
    serde_json::from_str::<RemoteArtifactManifest>(&text).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_MANIFEST",
            format!("invalid {}: {}", path.display(), err),
        )
    })
}

fn verify_remote_manifest_signature(version_dir: &Path) -> Result<(), ModelOpError> {
    let manifest_path = version_dir.join(REMOTE_MANIFEST_FILE);
    let signature_path = version_dir.join(REMOTE_SIGNATURE_FILE);
    let manifest_text = fs::read_to_string(&manifest_path).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_SIGNATURE_MISSING",
            format!("failed to read {}: {}", manifest_path.display(), err),
        )
    })?;
    let signature = fs::read_to_string(&signature_path).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_SIGNATURE_MISSING",
            format!("failed to read {}: {}", signature_path.display(), err),
        )
    })?;
    let key = read_signing_key()?;
    let expected = sign_manifest(&manifest_text, &key);
    if signature.trim() != expected {
        return Err(ModelOpError::new(
            "E_MODEL_SIGNATURE_INVALID",
            format!(
                "signature verification failed for {}",
                signature_path.display()
            ),
        ));
    }
    Ok(())
}

fn read_signing_key() -> Result<String, ModelOpError> {
    let key = std::env::var(SIGNING_KEY_ENV).map_err(|_| {
        ModelOpError::new(
            "E_MODEL_SIGNATURE_KEY_MISSING",
            format!("{} is required for signature operations", SIGNING_KEY_ENV),
        )
    })?;
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return Err(ModelOpError::new(
            "E_MODEL_SIGNATURE_KEY_MISSING",
            format!("{} cannot be empty", SIGNING_KEY_ENV),
        ));
    }
    Ok(trimmed.to_string())
}

fn sign_manifest(manifest_text: &str, key: &str) -> String {
    let digest = sha256_hex(manifest_text.as_bytes());
    sha256_hex(format!("enkai-model-signature-v1:{}:{}", key, digest).as_bytes())
}

fn file_sha256_hex(path: &Path) -> Result<String, ModelOpError> {
    let bytes = fs::read(path).map_err(|err| {
        ModelOpError::new(
            "E_MODEL_IO",
            format!("failed to read {}: {}", path.display(), err),
        )
    })?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn has_local_model_version(registry_dir: &Path, name: &str, version: &str) -> bool {
    match load_registry(registry_dir) {
        Ok(registry) => registry
            .models
            .get(name)
            .map(|entry| entry.versions.contains_key(version))
            .unwrap_or(false),
        Err(_) => false,
    }
}

fn append_audit_event(registry_dir: &Path, input: AuditAppend<'_>) -> Result<(), String> {
    fs::create_dir_all(registry_dir)
        .map_err(|err| format!("failed to create {}: {}", registry_dir.display(), err))?;
    let path = registry_dir.join(AUDIT_LOG_FILE);
    let event = RegistryAuditEvent {
        schema_version: 1,
        timestamp_ms: now_ms(),
        operation: input.operation.to_string(),
        status: input.status.to_string(),
        model: input.model.to_string(),
        version: input.version.to_string(),
        registry_path: registry_dir.to_string_lossy().to_string(),
        remote_registry: input
            .remote_registry
            .map(|item| item.to_string_lossy().to_string()),
        code: input.code.map(|value| value.to_string()),
        detail: input.detail.map(|value| value.to_string()),
    };
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|err| format!("failed to open {}: {}", path.display(), err))?;
    let line = serde_json::to_string(&event)
        .map_err(|err| format!("failed to encode audit event JSON: {}", err))?;
    writeln!(file, "{}", line)
        .map_err(|err| format!("failed to append {}: {}", path.display(), err))
}

fn print_model_entry(name: &str, model: &ModelEntry) {
    println!("model: {}", name);
    if let Some(active) = &model.active {
        println!("  active: {}", active);
    } else {
        println!("  active: (none)");
    }
    for (version, metadata) in &model.versions {
        println!(
            "  - {} [{}] kind={} artifact={}",
            version, metadata.status, metadata.artifact_kind, metadata.checkpoint_path
        );
    }
}

fn set_version_status(entry: &mut ModelEntry, active_version: &str, active_status: &str) {
    let now = now_ms();
    for (version, metadata) in entry.versions.iter_mut() {
        if version == active_version {
            metadata.status = active_status.to_string();
        } else if metadata.status == "active" {
            metadata.status = "registered".to_string();
        }
        metadata.updated_ms = now;
    }
}

fn load_registry(registry_dir: &Path) -> Result<ModelRegistry, String> {
    let path = registry_dir.join("registry.json");
    if !path.is_file() {
        return Ok(ModelRegistry {
            schema_version: 1,
            models: BTreeMap::new(),
        });
    }
    let text = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    let mut registry: ModelRegistry = serde_json::from_str(&text)
        .map_err(|err| format!("invalid {}: {}", path.display(), err))?;
    if registry.schema_version == 0 {
        registry.schema_version = 1;
    }
    Ok(registry)
}

fn save_registry(registry_dir: &Path, registry: &ModelRegistry) -> Result<(), String> {
    let path = registry_dir.join("registry.json");
    let mut to_write = registry.clone();
    to_write.schema_version = 1;
    let text = serde_json::to_string_pretty(&to_write)
        .map_err(|err| format!("failed to serialize registry: {}", err))?;
    let mut file = fs::File::create(&path)
        .map_err(|err| format!("failed to write {}: {}", path.display(), err))?;
    file.write_all(text.as_bytes())
        .map_err(|err| format!("failed to write {}: {}", path.display(), err))
}

fn serve_state_path(registry_dir: &Path) -> PathBuf {
    registry_dir.join(".serve_state.json")
}

fn load_serve_state(registry_dir: &Path) -> Result<ServeState, String> {
    let path = serve_state_path(registry_dir);
    if !path.is_file() {
        return Ok(ServeState {
            schema_version: 1,
            loaded: BTreeMap::new(),
        });
    }
    let text = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    let mut state: ServeState = serde_json::from_str(&text)
        .map_err(|err| format!("invalid {}: {}", path.display(), err))?;
    if state.schema_version == 0 {
        state.schema_version = 1;
    }
    Ok(state)
}

fn save_serve_state(registry_dir: &Path, state: &ServeState) -> Result<(), String> {
    let path = serve_state_path(registry_dir);
    let mut to_write = state.clone();
    to_write.schema_version = 1;
    let text = serde_json::to_string_pretty(&to_write)
        .map_err(|err| format!("failed to serialize serve state: {}", err))?;
    fs::write(&path, text).map_err(|err| format!("failed to write {}: {}", path.display(), err))
}

fn unload_model_version(registry_dir: &Path, name: &str, version: &str) -> Result<(), String> {
    let mut state = load_serve_state(registry_dir)?;
    if let Some(model_loaded) = state.loaded.get_mut(name) {
        model_loaded.remove(version);
        if model_loaded.is_empty() {
            state.loaded.remove(name);
        }
        save_serve_state(registry_dir, &state)?;
    }
    Ok(())
}

fn write_artifact_pointer(version_dir: &Path, artifact_path: &str) -> Result<(), String> {
    for file in [CHECKPOINT_POINTER_FILE, ARTIFACT_POINTER_FILE] {
        fs::write(version_dir.join(file), artifact_path).map_err(|err| {
            format!(
                "failed to write {}: {}",
                version_dir.join(file).display(),
                err
            )
        })?;
    }
    Ok(())
}

fn write_model_manifest(
    version_dir: &Path,
    name: &str,
    version: &str,
    entry: &ModelEntry,
) -> Result<(), String> {
    let version_meta = entry
        .versions
        .get(version)
        .ok_or_else(|| "version metadata missing".to_string())?;
    let payload = serde_json::json!({
        "schema_version": 1,
        "name": name,
        "version": version,
        "active": entry.active.as_deref() == Some(version),
        "status": version_meta.status,
        "artifact_kind": version_meta.artifact_kind,
        "artifact_path": version_meta.checkpoint_path,
        "checkpoint_path": version_meta.checkpoint_path,
        "artifact_manifest_path": version_meta.artifact_manifest_path,
        "lineage_manifest_path": version_meta.lineage_manifest_path,
        "created_ms": version_meta.created_ms,
        "updated_ms": version_meta.updated_ms,
    });
    let text = serde_json::to_string_pretty(&payload)
        .map_err(|err| format!("failed to serialize model manifest: {}", err))?;
    fs::write(version_dir.join("model.meta.json"), text).map_err(|err| {
        format!(
            "failed to write {}: {}",
            version_dir.join("model.meta.json").display(),
            err
        )
    })
}

fn write_model_manifest_for_version(
    registry_dir: &Path,
    name: &str,
    version: &str,
    entry: &ModelEntry,
) -> Result<(), String> {
    write_model_manifest(&registry_dir.join(name).join(version), name, version, entry)
}

fn now_ms() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis().min(u64::MAX as u128) as u64,
        Err(_) => 0,
    }
}

pub(crate) fn read_active_model_version(model_root: &Path) -> Option<String> {
    let path = model_root.join(".active_version");
    let raw = fs::read_to_string(path).ok()?;
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    Some(value.to_string())
}

pub(crate) fn resolve_checkpoint_pointer(version_dir: &Path) -> Option<PathBuf> {
    let pointer = resolve_artifact_pointer_path(version_dir)?;
    let raw = fs::read_to_string(pointer).ok()?;
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    let path = PathBuf::from(value);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn resolve_artifact_pointer_path(version_dir: &Path) -> Option<PathBuf> {
    for file in [ARTIFACT_POINTER_FILE, CHECKPOINT_POINTER_FILE] {
        let path = version_dir.join(file);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn resolve_artifact_pointer(version_dir: &Path) -> Option<PathBuf> {
    let pointer = resolve_artifact_pointer_path(version_dir)?;
    let raw = fs::read_to_string(pointer).ok()?;
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    Some(PathBuf::from(value))
}

fn version_files_for_manifest(
    version_dir: &Path,
    meta: &ModelVersion,
) -> Result<BTreeMap<String, String>, ModelOpError> {
    let mut files = BTreeMap::new();
    for file in [
        MODEL_MANIFEST_FILE,
        CHECKPOINT_POINTER_FILE,
        ARTIFACT_POINTER_FILE,
    ] {
        let path = version_dir.join(file);
        if path.is_file() {
            files.insert(file.to_string(), file_sha256_hex(&path)?);
        }
    }
    if let Some(path) = meta.artifact_manifest_path.as_deref() {
        let local_path = version_dir.join(file_name_or("artifact.manifest.json", Path::new(path)));
        if local_path.is_file() {
            let key = local_path
                .file_name()
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_else(|| "artifact.manifest.json".to_string());
            files.insert(key, file_sha256_hex(&local_path)?);
        }
    }
    if let Some(path) = meta.lineage_manifest_path.as_deref() {
        let local_path = version_dir.join(file_name_or("lineage.manifest.json", Path::new(path)));
        if local_path.is_file() {
            let key = local_path
                .file_name()
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_else(|| "lineage.manifest.json".to_string());
            files.insert(key, file_sha256_hex(&local_path)?);
        }
    }
    Ok(files)
}

fn materialize_registry_manifest_path(
    version_dir: &Path,
    raw_path: Option<&str>,
    fallback: &str,
) -> Option<String> {
    raw_path.map(|path| {
        version_dir
            .join(file_name_or(fallback, Path::new(path)))
            .to_string_lossy()
            .to_string()
    })
}

pub(crate) fn is_model_loaded(
    registry_dir: &Path,
    name: &str,
    version: &str,
) -> Result<bool, String> {
    let state = load_serve_state(registry_dir)?;
    Ok(state
        .loaded
        .get(name)
        .and_then(|versions| versions.get(version))
        .is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        GUARD
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|err| err.into_inner())
    }

    #[test]
    fn register_promote_retire_and_list_json() {
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let checkpoint = dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint).expect("checkpoint dir");

        let register = model_command(&[
            "register".to_string(),
            registry.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
            checkpoint.to_string_lossy().to_string(),
            "--activate".to_string(),
        ]);
        assert_eq!(register, 0);

        let promote = model_command(&[
            "promote".to_string(),
            registry.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
        ]);
        assert_eq!(promote, 0);

        let retire = model_command(&[
            "retire".to_string(),
            registry.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
        ]);
        assert_eq!(retire, 0);

        let list = model_command(&[
            "list".to_string(),
            registry.to_string_lossy().to_string(),
            "--json".to_string(),
        ]);
        assert_eq!(list, 0);
    }

    #[test]
    fn load_and_unload_model_versions() {
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let checkpoint = dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint).expect("checkpoint dir");

        let register = model_command(&[
            "register".to_string(),
            registry.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
            checkpoint.to_string_lossy().to_string(),
            "--activate".to_string(),
        ]);
        assert_eq!(register, 0);

        let load = model_command(&[
            "load".to_string(),
            registry.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
        ]);
        assert_eq!(load, 0);
        assert!(is_model_loaded(&registry, "chat", "v1.0.0").expect("loaded"));

        let loaded = model_command(&[
            "loaded".to_string(),
            registry.to_string_lossy().to_string(),
            "chat".to_string(),
            "--json".to_string(),
        ]);
        assert_eq!(loaded, 0);

        let unload = model_command(&[
            "unload".to_string(),
            registry.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
        ]);
        assert_eq!(unload, 0);
        assert!(!is_model_loaded(&registry, "chat", "v1.0.0").expect("loaded"));
    }

    #[test]
    fn active_version_pointer_roundtrip() {
        let dir = tempdir().expect("tempdir");
        let model_root = dir.path().join("chat");
        fs::create_dir_all(&model_root).expect("model root");
        fs::write(model_root.join(".active_version"), "v1.2.3\n").expect("write");
        let active = read_active_model_version(&model_root).expect("active");
        assert_eq!(active, "v1.2.3");
    }

    #[test]
    fn checkpoint_pointer_roundtrip() {
        let dir = tempdir().expect("tempdir");
        let version_dir = dir.path().join("chat").join("v1");
        fs::create_dir_all(&version_dir).expect("version dir");
        let checkpoint = dir.path().join("checkpoints").join("step_1");
        fs::create_dir_all(&checkpoint).expect("checkpoint");
        fs::write(
            version_dir.join("checkpoint_path.txt"),
            checkpoint.to_string_lossy().to_string(),
        )
        .expect("pointer");
        let resolved = resolve_checkpoint_pointer(&version_dir).expect("resolved");
        assert_eq!(resolved, checkpoint);
    }

    #[test]
    fn remote_push_pull_with_signature_and_fallback() {
        let _guard = env_guard();
        let prev = std::env::var(SIGNING_KEY_ENV).ok();
        std::env::set_var(SIGNING_KEY_ENV, "test-signing-key");

        let dir = tempdir().expect("tempdir");
        let local = dir.path().join("local");
        let remote = dir.path().join("remote");
        let ckpt = dir.path().join("checkpoint");
        fs::create_dir_all(&ckpt).expect("ckpt");

        let register = model_command(&[
            "register".to_string(),
            local.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
            ckpt.to_string_lossy().to_string(),
            "--activate".to_string(),
        ]);
        assert_eq!(register, 0);

        let push = model_command(&[
            "push".to_string(),
            local.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
            "--registry".to_string(),
            remote.to_string_lossy().to_string(),
            "--sign".to_string(),
        ]);
        assert_eq!(push, 0);

        let local_cache = dir.path().join("local_cache");
        fs::create_dir_all(&local_cache).expect("local cache");
        let pull = model_command(&[
            "pull".to_string(),
            local_cache.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
            "--registry".to_string(),
            remote.to_string_lossy().to_string(),
            "--verify-signature".to_string(),
        ]);
        assert_eq!(pull, 0);
        assert!(has_local_model_version(&local_cache, "chat", "v1.0.0"));

        let missing_remote = dir.path().join("missing_remote");
        let fallback_pull = model_command(&[
            "pull".to_string(),
            local_cache.to_string_lossy().to_string(),
            "chat".to_string(),
            "v1.0.0".to_string(),
            "--registry".to_string(),
            missing_remote.to_string_lossy().to_string(),
            "--verify-signature".to_string(),
            "--fallback-local".to_string(),
        ]);
        assert_eq!(fallback_pull, 0);

        let audit_log = fs::read_to_string(local_cache.join(AUDIT_LOG_FILE)).expect("audit");
        assert!(audit_log.contains("\"operation\":\"pull_remote\""));
        assert!(audit_log.contains("\"status\":\"fallback_local\""));

        if let Some(value) = prev {
            std::env::set_var(SIGNING_KEY_ENV, value);
        } else {
            std::env::remove_var(SIGNING_KEY_ENV);
        }
    }

    #[test]
    fn remote_state_sync_promote_retire_and_rollback() {
        let _guard = env_guard();
        let prev = std::env::var(SIGNING_KEY_ENV).ok();
        std::env::set_var(SIGNING_KEY_ENV, "test-signing-key");

        let dir = tempdir().expect("tempdir");
        let local = dir.path().join("local");
        let remote = dir.path().join("remote");
        let ckpt = dir.path().join("checkpoint");
        fs::create_dir_all(&ckpt).expect("ckpt");

        assert_eq!(
            model_command(&[
                "register".to_string(),
                local.to_string_lossy().to_string(),
                "chat".to_string(),
                "v1.0.0".to_string(),
                ckpt.to_string_lossy().to_string(),
                "--activate".to_string(),
            ]),
            0
        );
        assert_eq!(
            model_command(&[
                "push".to_string(),
                local.to_string_lossy().to_string(),
                "chat".to_string(),
                "v1.0.0".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
                "--sign".to_string(),
            ]),
            0
        );

        assert_eq!(
            model_command(&[
                "retire-remote".to_string(),
                local.to_string_lossy().to_string(),
                "chat".to_string(),
                "v1.0.0".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
                "--verify-signature".to_string(),
            ]),
            0
        );
        let local_registry = load_registry(&local).expect("local registry");
        assert_eq!(
            local_registry
                .models
                .get("chat")
                .and_then(|entry| entry.versions.get("v1.0.0"))
                .map(|meta| meta.status.as_str()),
            Some("retired")
        );

        assert_eq!(
            model_command(&[
                "rollback-remote".to_string(),
                local.to_string_lossy().to_string(),
                "chat".to_string(),
                "v1.0.0".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
                "--verify-signature".to_string(),
            ]),
            0
        );
        let local_registry = load_registry(&local).expect("local registry");
        assert_eq!(
            local_registry
                .models
                .get("chat")
                .and_then(|entry| entry.active.as_deref()),
            Some("v1.0.0")
        );

        if let Some(value) = prev {
            std::env::set_var(SIGNING_KEY_ENV, value);
        } else {
            std::env::remove_var(SIGNING_KEY_ENV);
        }
    }

    #[test]
    fn remote_pull_rejects_signature_mismatch_without_and_with_fallback() {
        let _guard = env_guard();
        let prev = std::env::var(SIGNING_KEY_ENV).ok();
        std::env::set_var(SIGNING_KEY_ENV, "test-signing-key");

        let dir = tempdir().expect("tempdir");
        let local = dir.path().join("local");
        let remote = dir.path().join("remote");
        let ckpt = dir.path().join("checkpoint");
        fs::create_dir_all(&ckpt).expect("ckpt");

        assert_eq!(
            model_command(&[
                "register".to_string(),
                local.to_string_lossy().to_string(),
                "chat".to_string(),
                "v1.0.1".to_string(),
                ckpt.to_string_lossy().to_string(),
                "--activate".to_string(),
            ]),
            0
        );
        assert_eq!(
            model_command(&[
                "push".to_string(),
                local.to_string_lossy().to_string(),
                "chat".to_string(),
                "v1.0.1".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
                "--sign".to_string(),
            ]),
            0
        );

        let remote_sig = remote
            .join("chat")
            .join("v1.0.1")
            .join(REMOTE_SIGNATURE_FILE);
        fs::write(&remote_sig, "tampered-signature").expect("tamper signature");

        let strict_cache = dir.path().join("strict_cache");
        fs::create_dir_all(&strict_cache).expect("strict cache");
        assert_eq!(
            model_command(&[
                "pull".to_string(),
                strict_cache.to_string_lossy().to_string(),
                "chat".to_string(),
                "v1.0.1".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
                "--verify-signature".to_string(),
            ]),
            1
        );

        let fallback_cache = dir.path().join("fallback_cache");
        fs::create_dir_all(&fallback_cache).expect("fallback cache");
        assert_eq!(
            model_command(&[
                "pull".to_string(),
                fallback_cache.to_string_lossy().to_string(),
                "chat".to_string(),
                "v1.0.1".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
            ]),
            0
        );
        assert_eq!(
            model_command(&[
                "pull".to_string(),
                fallback_cache.to_string_lossy().to_string(),
                "chat".to_string(),
                "v1.0.1".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
                "--verify-signature".to_string(),
                "--fallback-local".to_string(),
            ]),
            0
        );
        let audit = fs::read_to_string(fallback_cache.join(AUDIT_LOG_FILE)).expect("audit");
        assert!(audit.contains("\"status\":\"fallback_local\""));
        assert!(audit.contains("E_MODEL_SIGNATURE_INVALID"));

        if let Some(value) = prev {
            std::env::set_var(SIGNING_KEY_ENV, value);
        } else {
            std::env::remove_var(SIGNING_KEY_ENV);
        }
    }

    #[test]
    fn remote_pull_rejects_manifest_hash_mismatch() {
        let _guard = env_guard();
        let prev = std::env::var(SIGNING_KEY_ENV).ok();
        std::env::set_var(SIGNING_KEY_ENV, "test-signing-key");

        let dir = tempdir().expect("tempdir");
        let local = dir.path().join("local");
        let remote = dir.path().join("remote");
        let ckpt = dir.path().join("checkpoint");
        fs::create_dir_all(&ckpt).expect("ckpt");

        assert_eq!(
            model_command(&[
                "register".to_string(),
                local.to_string_lossy().to_string(),
                "chat".to_string(),
                "v2.0.0".to_string(),
                ckpt.to_string_lossy().to_string(),
                "--activate".to_string(),
            ]),
            0
        );
        assert_eq!(
            model_command(&[
                "push".to_string(),
                local.to_string_lossy().to_string(),
                "chat".to_string(),
                "v2.0.0".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
                "--sign".to_string(),
            ]),
            0
        );

        let remote_pointer = remote
            .join("chat")
            .join("v2.0.0")
            .join("checkpoint_path.txt");
        fs::write(remote_pointer, "/tmp/tampered-checkpoint\n").expect("tamper checkpoint path");

        let local_cache = dir.path().join("local_cache");
        fs::create_dir_all(&local_cache).expect("local cache");
        assert_eq!(
            model_command(&[
                "pull".to_string(),
                local_cache.to_string_lossy().to_string(),
                "chat".to_string(),
                "v2.0.0".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
            ]),
            1
        );

        if let Some(value) = prev {
            std::env::set_var(SIGNING_KEY_ENV, value);
        } else {
            std::env::remove_var(SIGNING_KEY_ENV);
        }
    }

    #[test]
    fn register_supports_simulation_artifact_metadata() {
        let dir = tempdir().expect("tempdir");
        let registry = dir.path().join("registry");
        let artifact = dir.path().join("snapshot.json");
        let artifact_manifest = dir.path().join("snapshot.manifest.json");
        let lineage_manifest = dir.path().join("lineage.json");
        fs::write(&artifact, "{}").expect("artifact");
        fs::write(&artifact_manifest, "{\"schema_version\":1}").expect("artifact manifest");
        fs::write(&lineage_manifest, "{\"schema_version\":1}").expect("lineage manifest");

        assert_eq!(
            model_command(&[
                "register".to_string(),
                registry.to_string_lossy().to_string(),
                "adam0".to_string(),
                "v2.8.0".to_string(),
                artifact.to_string_lossy().to_string(),
                "--artifact-kind".to_string(),
                "simulation".to_string(),
                "--artifact-manifest".to_string(),
                artifact_manifest.to_string_lossy().to_string(),
                "--lineage-manifest".to_string(),
                lineage_manifest.to_string_lossy().to_string(),
            ]),
            0
        );

        let registry_json = load_registry(&registry).expect("registry");
        let version = registry_json
            .models
            .get("adam0")
            .and_then(|entry| entry.versions.get("v2.8.0"))
            .expect("version");
        assert_eq!(version.artifact_kind, "simulation");
        assert!(version.artifact_manifest_path.is_some());
        assert!(version.lineage_manifest_path.is_some());
        assert!(registry
            .join("adam0")
            .join("v2.8.0")
            .join(ARTIFACT_POINTER_FILE)
            .is_file());
    }

    #[test]
    fn verify_signature_command_passes_for_signed_remote_artifact() {
        let _guard = env_guard();
        let prev = std::env::var(SIGNING_KEY_ENV).ok();
        std::env::set_var(SIGNING_KEY_ENV, "test-signing-key");

        let dir = tempdir().expect("tempdir");
        let local = dir.path().join("local");
        let remote = dir.path().join("remote");
        let artifact = dir.path().join("artifact.json");
        fs::write(&artifact, "{}").expect("artifact");

        assert_eq!(
            model_command(&[
                "register".to_string(),
                local.to_string_lossy().to_string(),
                "artifact".to_string(),
                "v2.8.0".to_string(),
                artifact.to_string_lossy().to_string(),
                "--artifact-kind".to_string(),
                "environment".to_string(),
            ]),
            0
        );
        assert_eq!(
            model_command(&[
                "push".to_string(),
                local.to_string_lossy().to_string(),
                "artifact".to_string(),
                "v2.8.0".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
                "--sign".to_string(),
            ]),
            0
        );
        assert_eq!(
            model_command(&[
                "verify-signature".to_string(),
                local.to_string_lossy().to_string(),
                "artifact".to_string(),
                "v2.8.0".to_string(),
                "--registry".to_string(),
                remote.to_string_lossy().to_string(),
            ]),
            0
        );

        if let Some(value) = prev {
            std::env::set_var(SIGNING_KEY_ENV, value);
        } else {
            std::env::remove_var(SIGNING_KEY_ENV);
        }
    }
}
