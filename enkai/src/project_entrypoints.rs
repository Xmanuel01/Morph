use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use toml::Value as TomlValue;

use enkai_compiler::bytecode::Program;
use enkai_compiler::modules::load_package;
use enkai_compiler::TypeChecker;
use enkai_runtime::VM;

use crate::bootstrap;

#[derive(Debug, Clone)]
struct DependencySpec {
    name: String,
    path: PathBuf,
    version_req: Option<String>,
}

#[derive(Debug, Clone)]
struct ResolvedDependency {
    name: String,
    path: PathBuf,
    version: Option<String>,
    package: Option<String>,
}

#[derive(Debug, Clone)]
struct ManifestInfo {
    dependencies: Vec<DependencySpec>,
}

#[derive(Debug, Clone, Default)]
struct PackageInfo {
    name: Option<String>,
    version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildCacheMeta {
    cache_version: u32,
    lang_version: String,
    entry: String,
    hash: String,
    program: String,
    compiler_backend: Option<String>,
}

const BUILD_CACHE_VERSION: u32 = 1;
const LOCK_VERSION: i64 = 1;

pub(crate) fn check_command(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("enkai check requires a file or directory");
        return 1;
    }
    let target = PathBuf::from(&args[0]);
    let (root, entry) = match resolve_entry(&target) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    match bootstrap::try_check_selfhost_entry(&entry) {
        Ok(Some(outcome)) => {
            crate::emit_command_backend_report("check", &entry, &root, outcome.backend);
            return outcome.exit_code;
        }
        Ok(None) => {}
        Err(_) => {}
    }
    let package = match load_package(&entry) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    crate::emit_command_backend_report("check", &entry, &root, bootstrap::SelfhostRunBackend::Rust);
    match TypeChecker::check_package(&package) {
        Ok(_) => 0,
        Err(err) => {
            if let Some(diagnostic) = err.diagnostic() {
                eprintln!("{}", diagnostic);
            } else {
                eprintln!(
                    "Type error: {} at {}:{}",
                    err.message, err.span.line, err.span.col
                );
            }
            1
        }
    }
}

pub(crate) fn test_command(args: &[String]) -> i32 {
    let target = if args.is_empty() {
        PathBuf::from(".")
    } else {
        PathBuf::from(&args[0])
    };
    let root = if target.is_dir() {
        find_project_root(&target).unwrap_or_else(|| target.clone())
    } else {
        let parent = target
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        find_project_root(&parent).unwrap_or(parent)
    };
    let tests_dir = root.join("tests");
    if !tests_dir.is_dir() {
        eprintln!("tests directory not found: {}", tests_dir.display());
        return 1;
    }
    let mut files = Vec::new();
    if let Err(err) = collect_source_files_in_dir(&tests_dir, &mut files) {
        eprintln!("{}", err);
        return 1;
    }
    if files.is_empty() {
        eprintln!("No .enk/.en tests found in {}", tests_dir.display());
        return 1;
    }
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut backend_rows: Vec<serde_json::Value> = Vec::new();
    for file in files {
        match bootstrap::try_run_selfhost_entry(&file) {
            Ok(Some(_outcome)) => {
                println!("[pass] {}", file.display());
                backend_rows.push(serde_json::json!({
                    "file": file.to_string_lossy(),
                    "compiler_backend": "selfhost",
                    "runtime_backend": "selfhost",
                    "status": "pass",
                }));
                passed += 1;
            }
            Ok(None) | Err(_) => {
                let (program, compiler_backend) =
                    match crate::compile_program_prefer_selfhost(&file) {
                        Ok(value) => value,
                        Err(err) => {
                            eprintln!("[fail] {}: {}", file.display(), err);
                            backend_rows.push(serde_json::json!({
                                "file": file.to_string_lossy(),
                                "compiler_backend": "error",
                                "runtime_backend": "none",
                                "status": "fail",
                                "error": err,
                            }));
                            failed += 1;
                            continue;
                        }
                    };
                let mut vm = VM::new(false, false, false, false);
                match vm.run(&program) {
                    Ok(_) => {
                        println!("[pass] {}", file.display());
                        backend_rows.push(serde_json::json!({
                            "file": file.to_string_lossy(),
                            "compiler_backend": compiler_backend.as_str(),
                            "runtime_backend": "rust",
                            "status": "pass",
                        }));
                        passed += 1;
                    }
                    Err(err) => {
                        eprintln!("[fail] {}: Runtime error: {}", file.display(), err);
                        backend_rows.push(serde_json::json!({
                            "file": file.to_string_lossy(),
                            "compiler_backend": compiler_backend.as_str(),
                            "runtime_backend": "rust",
                            "status": "fail",
                            "error": err.to_string(),
                        }));
                        failed += 1;
                    }
                }
            }
        }
    }
    emit_test_backend_report(&root, passed, failed, &backend_rows);
    println!("Tests: {} passed, {} failed", passed, failed);
    if failed == 0 {
        0
    } else {
        1
    }
}

pub(crate) fn build_command(args: &[String]) -> i32 {
    let target = if args.is_empty() {
        PathBuf::from(".")
    } else {
        PathBuf::from(&args[0])
    };
    let (root, entry) = match resolve_entry(&target) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    if find_manifest(&root).is_none() {
        eprintln!("enkai.toml not found in {}", root.display());
        return 1;
    }
    let manifest = match read_manifest_info(&root) {
        Ok(info) => info,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let deps = match resolve_dependencies(&root, &manifest) {
        Ok(list) => list,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    if let Err(err) = write_lockfile(&root, &deps) {
        eprintln!("{}", err);
        return 1;
    }
    let hash = match compute_build_hash(&root, &deps) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    let entry_rel = entry
        .strip_prefix(&root)
        .unwrap_or(&entry)
        .to_string_lossy()
        .to_string();
    if let Ok(Some(meta)) = load_build_cache(&root) {
        if is_cache_valid(&meta, &entry_rel, &hash) {
            if let Some(backend) = meta.compiler_backend.as_deref() {
                let report_backend = match backend {
                    "selfhost" => bootstrap::SelfhostRunBackend::Selfhost,
                    _ => bootstrap::SelfhostRunBackend::Rust,
                };
                crate::emit_command_backend_report("build", &entry, &root, report_backend);
            }
            println!("build up to date");
            return 0;
        }
    }
    let (program, backend) = match crate::compile_program_prefer_selfhost(&entry) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return 1;
        }
    };
    if let Err(err) = write_build_cache(&root, &entry_rel, &hash, &program, backend) {
        eprintln!("{}", err);
        return 1;
    }
    crate::emit_command_backend_report("build", &entry, &root, backend);
    println!("build ok");
    0
}

pub(crate) fn resolve_entry(target: &Path) -> Result<(PathBuf, PathBuf), String> {
    let metadata = fs::metadata(target)
        .map_err(|err| format!("Failed to read {}: {}", target.display(), err))?;
    if metadata.is_dir() {
        let root = target.to_path_buf();
        let _manifest = find_manifest(&root)
            .ok_or_else(|| format!("enkai.toml not found in {}", root.display()))?;
        let entry = find_entry_file(&root)
            .ok_or_else(|| "Entry file not found (main.enk/main.en/main.enkai)".to_string())?;
        Ok((root, entry))
    } else {
        let file = target.to_path_buf();
        let parent = file
            .parent()
            .ok_or_else(|| "Invalid file path".to_string())?;
        let root = find_project_root(parent).unwrap_or_else(|| parent.to_path_buf());
        Ok((root, file))
    }
}

pub(crate) fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start);
    while let Some(dir) = current {
        if find_manifest(dir).is_some() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    None
}

pub(crate) fn collect_source_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        if is_source_extension(path) {
            return Ok(vec![path.to_path_buf()]);
        }
        return Err(format!("Not an .enk/.en file: {}", path.display()));
    }
    if !path.is_dir() {
        return Err(format!("Path not found: {}", path.display()));
    }
    let scan_root = if find_manifest(path).is_some() {
        let src = path.join("src");
        if src.is_dir() {
            src
        } else {
            path.to_path_buf()
        }
    } else {
        path.to_path_buf()
    };
    let mut files = Vec::new();
    collect_source_files_in_dir(&scan_root, &mut files)?;
    if files.is_empty() {
        return Err(format!(
            "No .enk/.en files found in {}",
            scan_root.display()
        ));
    }
    Ok(files)
}

pub(crate) fn collect_source_files_in_dir(
    dir: &Path,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|err| format!("Failed to read {}: {}", dir.display(), err))?
    {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_source_files_in_dir(&path, files)?;
        } else if is_source_extension(&path) {
            files.push(path);
        }
    }
    Ok(())
}

pub(crate) fn load_cached_program(root: &Path, entry: &Path) -> Result<Option<Program>, String> {
    let Some(meta) = load_build_cache(root)? else {
        return Ok(None);
    };
    let lock_path = root.join("enkai.lock");
    if !lock_path.is_file() {
        return Ok(None);
    }
    let entry_rel = entry
        .strip_prefix(root)
        .unwrap_or(entry)
        .to_string_lossy()
        .to_string();
    if !is_cache_valid(&meta, &entry_rel, &meta.hash) {
        return Ok(None);
    }
    let deps = read_lock_dependencies(root)?;
    let hash = compute_build_hash(root, &deps)?;
    if meta.hash != hash {
        return Ok(None);
    }
    let program_path = cache_program_path(root, &meta);
    if !program_path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&program_path)
        .map_err(|err| format!("Failed to read {}: {}", program_path.display(), err))?;
    let program: Program =
        bincode::deserialize(&bytes).map_err(|err| format!("Cache decode failed: {}", err))?;
    Ok(Some(program))
}

fn emit_test_backend_report(root: &Path, passed: usize, failed: usize, rows: &[serde_json::Value]) {
    let payload = serde_json::json!({
        "command": "test",
        "root": root.to_string_lossy(),
        "passed": passed,
        "failed": failed,
        "entries": rows,
    });
    if let Err(err) = crate::write_json_report_to_env_path("ENKAI_TEST_BACKEND_REPORT", &payload) {
        eprintln!("[test] failed to write backend report: {}", err);
    }
}

fn cache_dir(root: &Path) -> PathBuf {
    root.join("target").join("enkai")
}

fn cache_meta_path(root: &Path) -> PathBuf {
    cache_dir(root).join("build.json")
}

fn cache_program_path(root: &Path, meta: &BuildCacheMeta) -> PathBuf {
    cache_dir(root).join(&meta.program)
}

fn load_build_cache(root: &Path) -> Result<Option<BuildCacheMeta>, String> {
    let path = cache_meta_path(root);
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    let meta: BuildCacheMeta =
        serde_json::from_str(&text).map_err(|err| format!("Invalid cache: {}", err))?;
    Ok(Some(meta))
}

fn is_cache_valid(meta: &BuildCacheMeta, entry: &str, hash: &str) -> bool {
    meta.cache_version == BUILD_CACHE_VERSION
        && meta.lang_version == crate::language_version()
        && meta.entry == entry
        && meta.hash == hash
}

fn write_build_cache(
    root: &Path,
    entry: &str,
    hash: &str,
    program: &Program,
    backend: bootstrap::SelfhostRunBackend,
) -> Result<(), String> {
    let dir = cache_dir(root);
    fs::create_dir_all(&dir)
        .map_err(|err| format!("Failed to create {}: {}", dir.display(), err))?;
    let program_path = dir.join("program.bin");
    let encoded =
        bincode::serialize(program).map_err(|err| format!("Cache serialize failed: {}", err))?;
    fs::write(&program_path, encoded)
        .map_err(|err| format!("Failed to write {}: {}", program_path.display(), err))?;
    let meta = BuildCacheMeta {
        cache_version: BUILD_CACHE_VERSION,
        lang_version: crate::language_version().to_string(),
        entry: entry.to_string(),
        hash: hash.to_string(),
        program: "program.bin".to_string(),
        compiler_backend: Some(backend.as_str().to_string()),
    };
    let meta_text =
        serde_json::to_string_pretty(&meta).map_err(|err| format!("Cache meta error: {}", err))?;
    let meta_path = cache_meta_path(root);
    fs::write(&meta_path, meta_text)
        .map_err(|err| format!("Failed to write {}: {}", meta_path.display(), err))?;
    Ok(())
}

fn read_manifest_info(root: &Path) -> Result<ManifestInfo, String> {
    let path = root.join("enkai.toml");
    let source = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    let value = source
        .parse::<TomlValue>()
        .map_err(|err| format!("Failed to parse {}: {}", path.display(), err))?;
    let mut dependencies = Vec::new();
    if let Some(table) = value.get("dependencies").and_then(|val| val.as_table()) {
        for (name, entry) in table {
            let (path, version_req) = if let Some(path) = entry.as_str() {
                (PathBuf::from(path), None)
            } else if let Some(dep_table) = entry.as_table() {
                let path = dep_table
                    .get("path")
                    .and_then(|val| val.as_str())
                    .ok_or_else(|| format!("Dependency {} missing path", name))?;
                let version_req = dep_table
                    .get("version")
                    .and_then(|val| val.as_str())
                    .map(|s| s.to_string());
                (PathBuf::from(path), version_req)
            } else {
                return Err(format!("Dependency {} must be path or table", name));
            };
            dependencies.push(DependencySpec {
                name: name.to_string(),
                path,
                version_req,
            });
        }
    }
    Ok(ManifestInfo { dependencies })
}

fn read_package_info(root: &Path) -> Result<PackageInfo, String> {
    let path = root.join("enkai.toml");
    if !path.is_file() {
        return Ok(PackageInfo::default());
    }
    let source = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    let value = source
        .parse::<TomlValue>()
        .map_err(|err| format!("Failed to parse {}: {}", path.display(), err))?;
    let mut info = PackageInfo::default();
    if let Some(table) = value.get("package").and_then(|val| val.as_table()) {
        info.name = table
            .get("name")
            .and_then(|val| val.as_str())
            .map(|s| s.to_string());
        info.version = table
            .get("version")
            .and_then(|val| val.as_str())
            .map(|s| s.to_string());
    }
    Ok(info)
}

fn resolve_dependencies(
    root: &Path,
    manifest: &ManifestInfo,
) -> Result<Vec<ResolvedDependency>, String> {
    let mut resolved = Vec::new();
    let mut seen = HashSet::new();
    for dep in &manifest.dependencies {
        if !seen.insert(dep.name.clone()) {
            return Err(format!("Duplicate dependency name {}", dep.name));
        }
        let dep_root = root.join(&dep.path);
        if !dep_root.is_dir() {
            return Err(format!(
                "Dependency {} path not found: {}",
                dep.name,
                dep_root.display()
            ));
        }
        let info = read_package_info(&dep_root)?;
        if let Some(req) = &dep.version_req {
            match info.version.as_ref() {
                Some(version) if version == req => {}
                Some(version) => {
                    return Err(format!(
                        "Dependency {} version mismatch (expected {}, found {})",
                        dep.name, req, version
                    ))
                }
                None => {
                    return Err(format!(
                        "Dependency {} missing package.version (expected {})",
                        dep.name, req
                    ))
                }
            }
        }
        resolved.push(ResolvedDependency {
            name: dep.name.clone(),
            path: dep.path.clone(),
            version: info.version,
            package: info.name,
        });
    }
    resolved.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(resolved)
}

fn write_lockfile(root: &Path, deps: &[ResolvedDependency]) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("lock_version = 1\n");
    for dep in deps {
        out.push_str("\n[[dependency]]\n");
        out.push_str(&format!("name = \"{}\"\n", toml_escape(&dep.name)));
        out.push_str(&format!(
            "path = \"{}\"\n",
            toml_escape(&dep.path.to_string_lossy())
        ));
        if let Some(version) = &dep.version {
            out.push_str(&format!("version = \"{}\"\n", toml_escape(version)));
        }
        if let Some(package) = &dep.package {
            out.push_str(&format!("package = \"{}\"\n", toml_escape(package)));
        }
    }
    let path = root.join("enkai.lock");
    fs::write(&path, out).map_err(|err| format!("Failed to write {}: {}", path.display(), err))?;
    Ok(())
}

fn read_lock_dependencies(root: &Path) -> Result<Vec<ResolvedDependency>, String> {
    let path = root.join("enkai.lock");
    let source = fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
    let value = source
        .parse::<TomlValue>()
        .map_err(|err| format!("Failed to parse {}: {}", path.display(), err))?;
    if let Some(lock_version) = value.get("lock_version").and_then(|v| v.as_integer()) {
        if lock_version != LOCK_VERSION {
            return Err(format!("Unsupported lock_version {}", lock_version));
        }
    }
    let mut deps = Vec::new();
    if let Some(entries) = value.get("dependency").and_then(|v| v.as_array()) {
        for entry in entries {
            let name = entry
                .get("name")
                .and_then(|val| val.as_str())
                .ok_or_else(|| "Dependency entry missing name".to_string())?;
            let path_value = entry
                .get("path")
                .and_then(|val| val.as_str())
                .ok_or_else(|| format!("Dependency {} missing path", name))?;
            let version = entry
                .get("version")
                .and_then(|val| val.as_str())
                .map(|s| s.to_string());
            let package = entry
                .get("package")
                .and_then(|val| val.as_str())
                .map(|s| s.to_string());
            deps.push(ResolvedDependency {
                name: name.to_string(),
                path: PathBuf::from(path_value),
                version,
                package,
            });
        }
    }
    deps.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(deps)
}

fn compute_build_hash(root: &Path, deps: &[ResolvedDependency]) -> Result<String, String> {
    let mut files = collect_build_inputs(root, deps)?;
    files.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    files.dedup();
    let mut hasher = Sha256::new();
    for path in files {
        let name = path.to_string_lossy();
        hasher.update(name.as_bytes());
        hasher.update([0u8]);
        let data =
            fs::read(&path).map_err(|err| format!("Failed to read {}: {}", path.display(), err))?;
        hasher.update(&data);
    }
    let digest = hasher.finalize();
    Ok(format!("{:x}", digest))
}

fn collect_build_inputs(root: &Path, deps: &[ResolvedDependency]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let src_dir = if root.join("src").is_dir() {
        root.join("src")
    } else {
        root.to_path_buf()
    };
    collect_source_files_in_dir(&src_dir, &mut files)?;
    let manifest = root.join("enkai.toml");
    if manifest.is_file() {
        files.push(manifest);
    }
    let lock = root.join("enkai.lock");
    if lock.is_file() {
        files.push(lock);
    }
    for dep in deps {
        let dep_root = if dep.path.is_absolute() {
            dep.path.clone()
        } else {
            root.join(&dep.path)
        };
        let dep_src = if dep_root.join("src").is_dir() {
            dep_root.join("src")
        } else {
            dep_root.clone()
        };
        if dep_src.is_dir() {
            collect_source_files_in_dir(&dep_src, &mut files)?;
        }
        let dep_manifest = dep_root.join("enkai.toml");
        if dep_manifest.is_file() {
            files.push(dep_manifest);
        }
    }
    Ok(files)
}

fn toml_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn is_source_extension(path: &Path) -> bool {
    let ext = path.extension().and_then(|ext| ext.to_str());
    matches!(ext, Some("enk") | Some("en") | Some("enkai"))
}

fn find_manifest(root: &Path) -> Option<PathBuf> {
    let enkai = root.join("enkai.toml");
    if enkai.is_file() {
        return Some(enkai);
    }
    None
}

fn find_entry_file(root: &Path) -> Option<PathBuf> {
    let candidates = if root.join("src").is_dir() {
        root.join("src")
    } else {
        root.to_path_buf()
    };
    let enk = candidates.join("main.enk");
    if enk.is_file() {
        return Some(enk);
    }
    let en = candidates.join("main.en");
    if en.is_file() {
        return Some(en);
    }
    let legacy = candidates.join("main.enkai");
    if legacy.is_file() {
        return Some(legacy);
    }
    None
}
