use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::ast::{Item, Module};
use crate::parser::{parse_module_named, ParseError};

#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub path: Vec<String>,
    pub file: PathBuf,
    pub module: Module,
    pub uses: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Package {
    pub root: PathBuf,
    pub src_dir: PathBuf,
    pub entry: Vec<String>,
    pub modules: HashMap<Vec<String>, ModuleInfo>,
}

#[derive(Debug)]
pub enum LoadError {
    Io { path: PathBuf, message: String },
    Parse { path: PathBuf, error: ParseError },
    MissingModule { path: Vec<String> },
    Cycle { path: Vec<String> },
    EntryNotFound { path: PathBuf },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io { path, message } => {
                write!(f, "Failed to read {}: {}", path.display(), message)
            }
            LoadError::Parse { path, error } => {
                write!(f, "Parse error in {}: {}", path.display(), error)
            }
            LoadError::MissingModule { path } => {
                write!(f, "Module not found: {}", path.join("."))
            }
            LoadError::Cycle { path } => write!(f, "Cyclic module dependency: {}", path.join(".")),
            LoadError::EntryNotFound { path } => {
                write!(f, "Entry file not found: {}", path.display())
            }
        }
    }
}

impl std::error::Error for LoadError {}

pub fn load_package(entry: &Path, root: &Path) -> Result<Package, LoadError> {
    if !entry.is_file() {
        return Err(LoadError::EntryNotFound {
            path: entry.to_path_buf(),
        });
    }
    let src_dir = pick_source_dir(root);
    let entry_path = module_path_from_file(&src_dir, entry);
    let mut loader = ModuleLoader::new(root.to_path_buf(), src_dir);
    loader.load_entry(entry, entry_path.clone())?;
    Ok(Package {
        root: loader.root,
        src_dir: loader.src_dir,
        entry: entry_path,
        modules: loader.modules,
    })
}

struct ModuleLoader {
    root: PathBuf,
    src_dir: PathBuf,
    modules: HashMap<Vec<String>, ModuleInfo>,
    loading: HashSet<Vec<String>>,
}

impl ModuleLoader {
    fn new(root: PathBuf, src_dir: PathBuf) -> Self {
        Self {
            root,
            src_dir,
            modules: HashMap::new(),
            loading: HashSet::new(),
        }
    }

    fn load_entry(&mut self, entry: &Path, path: Vec<String>) -> Result<(), LoadError> {
        let info = self.load_file(entry, path.clone())?;
        let deps = info.uses.clone();
        self.modules.insert(path.clone(), info);
        for dep in deps {
            if is_std_path(&dep) {
                continue;
            }
            self.load_module_path(dep)?;
        }
        Ok(())
    }

    fn load_module_path(&mut self, path: Vec<String>) -> Result<(), LoadError> {
        if self.modules.contains_key(&path) {
            return Ok(());
        }
        if self.loading.contains(&path) {
            return Err(LoadError::Cycle { path });
        }
        self.loading.insert(path.clone());
        let file = self
            .resolve_module_file(&path)
            .ok_or_else(|| LoadError::MissingModule { path: path.clone() })?;
        let info = self.load_file(&file, path.clone())?;
        let deps = info.uses.clone();
        self.modules.insert(path.clone(), info);
        for dep in deps {
            if is_std_path(&dep) {
                continue;
            }
            self.load_module_path(dep)?;
        }
        self.loading.remove(&path);
        Ok(())
    }

    fn load_file(&self, path: &Path, module_path: Vec<String>) -> Result<ModuleInfo, LoadError> {
        let source = fs::read_to_string(path).map_err(|err| LoadError::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?;
        let name = path.to_string_lossy();
        let module =
            parse_module_named(&source, Some(name.as_ref())).map_err(|err| LoadError::Parse {
                path: path.to_path_buf(),
                error: err,
            })?;
        let uses = collect_uses(&module);
        Ok(ModuleInfo {
            path: module_path,
            file: path.to_path_buf(),
            module,
            uses,
        })
    }

    fn resolve_module_file(&self, path: &[String]) -> Option<PathBuf> {
        let mut base = self.src_dir.clone();
        for segment in path {
            base.push(segment);
        }
        let direct = base.with_extension("morph");
        if direct.is_file() {
            return Some(direct);
        }
        let index = base.join("index.morph");
        if index.is_file() {
            return Some(index);
        }
        None
    }
}

fn pick_source_dir(root: &Path) -> PathBuf {
    let src = root.join("src");
    if src.is_dir() {
        src
    } else {
        root.to_path_buf()
    }
}

fn module_path_from_file(src_dir: &Path, file: &Path) -> Vec<String> {
    if let Ok(relative) = file.strip_prefix(src_dir) {
        let mut parts: Vec<String> = relative
            .with_extension("")
            .components()
            .map(|comp| comp.as_os_str().to_string_lossy().to_string())
            .collect();
        if parts.len() > 1 && parts.last().map(|name| name == "index").unwrap_or(false) {
            parts.pop();
        }
        if !parts.is_empty() {
            return parts;
        }
    }
    file.file_stem()
        .map(|stem| vec![stem.to_string_lossy().to_string()])
        .unwrap_or_default()
}

fn collect_uses(module: &Module) -> Vec<Vec<String>> {
    let mut uses = Vec::new();
    let mut seen = HashSet::new();
    for item in &module.items {
        if let Item::Use(decl) = item {
            if decl.path.is_empty() {
                continue;
            }
            if seen.insert(decl.path.clone()) {
                uses.push(decl.path.clone());
            }
        }
    }
    uses
}

fn is_std_path(path: &[String]) -> bool {
    matches!(path.first(), Some(segment) if segment == "std")
}
