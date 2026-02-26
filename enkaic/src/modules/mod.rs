use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::ast::{ImportDecl, Item, Module};
use crate::parser::{parse_module_named, ParseError};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModuleId(pub Vec<String>);

#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub id: ModuleId,
    pub file: PathBuf,
    pub source: String,
    pub module: Module,
    pub imports: Vec<ImportDecl>,
    pub import_aliases: HashMap<String, ModuleId>,
    pub exports: HashSet<String>,
}

#[derive(Debug, Clone)]
pub struct Package {
    pub root: PathBuf,
    pub entry: ModuleId,
    pub modules: HashMap<ModuleId, ModuleInfo>,
    pub order: Vec<ModuleId>,
}

#[derive(Debug)]
pub enum ModuleError {
    Io {
        path: PathBuf,
        message: String,
    },
    Parse {
        path: PathBuf,
        error: Box<ParseError>,
    },
    MissingModule {
        path: Vec<String>,
    },
    Cycle {
        chain: Vec<ModuleId>,
    },
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::Io { path, message } => {
                write!(f, "Failed to read {}: {}", path.display(), message)
            }
            ModuleError::Parse { path, error } => {
                write!(f, "Parse error in {}: {}", path.display(), error)
            }
            ModuleError::MissingModule { path } => {
                write!(f, "Module not found: {}", path.join("::"))
            }
            ModuleError::Cycle { chain } => {
                let names = chain
                    .iter()
                    .map(|id| id.0.join("::"))
                    .collect::<Vec<_>>()
                    .join(" -> ");
                write!(f, "Circular import detected: {}", names)
            }
        }
    }
}

impl std::error::Error for ModuleError {}

pub fn load_package(entry: &Path) -> Result<Package, ModuleError> {
    let root = entry
        .parent()
        .ok_or_else(|| ModuleError::Io {
            path: entry.to_path_buf(),
            message: "Invalid entry path".to_string(),
        })?
        .to_path_buf();
    let entry_id = module_id_from_file(&root, entry);
    let mut loader = ModuleLoader::new(root.clone());
    loader.load_module(&entry_id)?;
    Ok(Package {
        root,
        entry: entry_id,
        modules: loader.modules,
        order: loader.order,
    })
}

struct ModuleLoader {
    root: PathBuf,
    modules: HashMap<ModuleId, ModuleInfo>,
    visiting: Vec<ModuleId>,
    order: Vec<ModuleId>,
}

impl ModuleLoader {
    fn new(root: PathBuf) -> Self {
        Self {
            root,
            modules: HashMap::new(),
            visiting: Vec::new(),
            order: Vec::new(),
        }
    }

    fn load_module(&mut self, id: &ModuleId) -> Result<(), ModuleError> {
        if let Some(pos) = self.visiting.iter().position(|m| m == id) {
            let mut chain = self.visiting[pos..].to_vec();
            chain.push(id.clone());
            return Err(ModuleError::Cycle { chain });
        }
        if self.modules.contains_key(id) {
            return Ok(());
        }
        self.visiting.push(id.clone());
        let file = resolve_module_file(&self.root, id)?;
        let source = fs::read_to_string(&file).map_err(|err| ModuleError::Io {
            path: file.clone(),
            message: err.to_string(),
        })?;
        let name = file.to_string_lossy().to_string();
        let module =
            parse_module_named(&source, Some(&name)).map_err(|err| ModuleError::Parse {
                path: file.clone(),
                error: Box::new(err),
            })?;
        let (imports, import_aliases) = collect_imports(&module);
        let exports = collect_exports(&module);
        let info = ModuleInfo {
            id: id.clone(),
            file: file.clone(),
            source,
            module,
            imports: imports.clone(),
            import_aliases,
            exports,
        };
        self.modules.insert(id.clone(), info);
        for import in imports {
            let dep_id = ModuleId(import.path.clone());
            self.load_module(&dep_id)?;
        }
        self.visiting.pop();
        self.order.push(id.clone());
        Ok(())
    }
}

fn resolve_module_file(root: &Path, id: &ModuleId) -> Result<PathBuf, ModuleError> {
    if id.0.is_empty() {
        return Err(ModuleError::MissingModule { path: id.0.clone() });
    }
    if matches!(id.0.first(), Some(first) if first == "std") {
        if let Some(path) = resolve_std_module(root, &id.0[1..]) {
            return Ok(path);
        }
        return Err(ModuleError::MissingModule { path: id.0.clone() });
    }
    let mut path = root.to_path_buf();
    for segment in &id.0 {
        path.push(segment);
    }
    if let Some(found) = with_source_extension(&path) {
        return Ok(found);
    }
    Err(ModuleError::MissingModule { path: id.0.clone() })
}

fn resolve_std_module(root: &Path, segments: &[String]) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(root.join("std"));
    if let Ok(env_root) = std::env::var("ENKAI_STD") {
        candidates.push(PathBuf::from(env_root));
    }
    if let Ok(env_root) = std::env::var("enkai_STD") {
        candidates.push(PathBuf::from(env_root));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("std"));
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join("std"));
            }
        }
    }
    for base in candidates {
        let mut path = base.clone();
        for segment in segments {
            path.push(segment);
        }
        if let Some(found) = with_source_extension(&path) {
            return Some(found);
        }
    }
    None
}

fn with_source_extension(base: &Path) -> Option<PathBuf> {
    let extensions = ["enk", "en", "enkai"];
    for ext in &extensions {
        let candidate = base.with_extension(ext);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn module_id_from_file(root: &Path, file: &Path) -> ModuleId {
    if let Ok(rel) = file.strip_prefix(root) {
        let parts: Vec<String> = rel
            .with_extension("")
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();
        if !parts.is_empty() {
            return ModuleId(parts);
        }
    }
    ModuleId(
        file.file_stem()
            .map(|s| vec![s.to_string_lossy().to_string()])
            .unwrap_or_default(),
    )
}

fn collect_imports(module: &Module) -> (Vec<ImportDecl>, HashMap<String, ModuleId>) {
    let mut imports = Vec::new();
    let mut aliases = HashMap::new();
    for item in &module.items {
        if let Item::Import(decl) = item {
            let alias = decl
                .alias
                .clone()
                .unwrap_or_else(|| decl.path.last().cloned().unwrap_or_default());
            aliases.insert(alias, ModuleId(decl.path.clone()));
            imports.push(decl.clone());
        }
    }
    (imports, aliases)
}

fn collect_exports(module: &Module) -> HashSet<String> {
    let mut exports = HashSet::new();
    for item in &module.items {
        if let Item::Fn(decl) = item {
            if decl.is_pub {
                exports.insert(decl.name.clone());
            }
        } else if let Item::Type(decl) = item {
            if decl.is_pub {
                exports.insert(decl.name.clone());
            }
        } else if let Item::Enum(decl) = item {
            if decl.is_pub {
                exports.insert(decl.name.clone());
            }
        }
    }
    exports
}
