use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

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
    created_ms: u64,
    updated_ms: u64,
}

pub fn model_command(args: &[String]) -> i32 {
    if args.is_empty() {
        print_model_usage();
        return 1;
    }
    match args[0].as_str() {
        "register" => model_register(&args[1..]),
        "list" => model_list(&args[1..]),
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
        "  enkai model register <registry_dir> <name> <version> <checkpoint_path> [--activate]"
    );
    eprintln!("  enkai model list <registry_dir> [name] [--json]");
    eprintln!("  enkai model promote <registry_dir> <name> <version>");
    eprintln!("  enkai model retire <registry_dir> <name> <version>");
    eprintln!("  enkai model rollback <registry_dir> <name> <version>");
}

fn model_register(args: &[String]) -> i32 {
    if args.len() < 4 || args.len() > 5 {
        eprintln!("Usage: enkai model register <registry_dir> <name> <version> <checkpoint_path> [--activate]");
        return 1;
    }
    let registry_dir = PathBuf::from(&args[0]);
    let name = args[1].trim();
    let version = args[2].trim();
    let checkpoint_path = PathBuf::from(&args[3]);
    let activate = args.get(4).map(|v| v.as_str()) == Some("--activate");
    if args.len() == 5 && !activate {
        eprintln!("enkai model register: unknown option '{}'", args[4]);
        return 1;
    }
    if name.is_empty() || version.is_empty() {
        eprintln!("enkai model register: name/version cannot be empty");
        return 1;
    }
    let checkpoint_path = match fs::canonicalize(&checkpoint_path) {
        Ok(path) => path,
        Err(err) => {
            eprintln!(
                "enkai model register: checkpoint path {} is invalid: {}",
                checkpoint_path.display(),
                err
            );
            return 1;
        }
    };
    if !checkpoint_path.exists() {
        eprintln!(
            "enkai model register: checkpoint path not found: {}",
            checkpoint_path.display()
        );
        return 1;
    }
    if let Err(err) = fs::create_dir_all(&registry_dir) {
        eprintln!(
            "enkai model register: failed to create registry dir {}: {}",
            registry_dir.display(),
            err
        );
        return 1;
    }
    let mut registry = match load_registry(&registry_dir) {
        Ok(registry) => registry,
        Err(err) => {
            eprintln!("enkai model register: {}", err);
            return 1;
        }
    };
    let model_dir = registry_dir.join(name).join(version);
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
    let entry = registry.models.entry(name.to_string()).or_default();
    let existing_created = entry
        .versions
        .get(version)
        .map(|existing| existing.created_ms)
        .unwrap_or(now);
    entry.versions.insert(
        version.to_string(),
        ModelVersion {
            status: "registered".to_string(),
            checkpoint_path: checkpoint_text.clone(),
            created_ms: existing_created,
            updated_ms: now,
        },
    );
    if activate || entry.active.is_none() {
        entry.active = Some(version.to_string());
        set_version_status(entry, version, "active");
    }
    if let Err(err) = write_checkpoint_pointer(&model_dir, &checkpoint_text) {
        eprintln!("enkai model register: {}", err);
        return 1;
    }
    if let Err(err) = write_model_manifest(&model_dir, name, version, entry) {
        eprintln!("enkai model register: {}", err);
        return 1;
    }
    if let Some(active) = &entry.active {
        if let Err(err) = fs::write(registry_dir.join(name).join(".active_version"), active) {
            eprintln!(
                "enkai model register: failed to write active version pointer: {}",
                err
            );
            return 1;
        }
    }
    if let Err(err) = save_registry(&registry_dir, &registry) {
        eprintln!("enkai model register: {}", err);
        return 1;
    }
    println!(
        "registered model {} {} (checkpoint: {})",
        name,
        version,
        checkpoint_path.display()
    );
    0
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
    println!("{}d model {} {}", kind, name, version);
    0
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
    println!("retired model {} {}", name, version);
    0
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
            "  - {} [{}] checkpoint={}",
            version, metadata.status, metadata.checkpoint_path
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

fn write_checkpoint_pointer(version_dir: &Path, checkpoint_path: &str) -> Result<(), String> {
    fs::write(version_dir.join("checkpoint_path.txt"), checkpoint_path).map_err(|err| {
        format!(
            "failed to write {}: {}",
            version_dir.join("checkpoint_path.txt").display(),
            err
        )
    })
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
        "checkpoint_path": version_meta.checkpoint_path,
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
    let pointer = version_dir.join("checkpoint_path.txt");
    if !pointer.is_file() {
        return None;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
}
