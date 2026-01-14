use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::ast::{Item, Module, UseDecl};
use crate::diagnostic::{Diagnostic, Span};
use crate::parser::{parse_module_named, ParseError};
use toml::Value;

#[derive(Debug, Default)]
struct Manifest {
    dependencies: HashMap<String, Dependency>,
}

#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub path: Vec<String>,
    pub file: PathBuf,
    pub source: String,
    pub module: Module,
    pub uses: Vec<UseDecl>,
    pub resolved_uses: Vec<Vec<ResolvedUse>>,
    pub exports: HashSet<String>,
}

#[derive(Debug, Clone)]
pub enum UseTarget {
    Module {
        path: Vec<String>,
    },
    Symbol {
        module_path: Vec<String>,
        symbol: String,
    },
}

#[derive(Debug, Clone)]
pub struct ResolvedUse {
    pub alias: String,
    pub target: UseTarget,
    pub is_pub: bool,
    pub spans: Vec<Span>,
}

#[derive(Debug, Clone)]
pub struct Dependency {
    pub name: String,
    pub root: PathBuf,
    pub src_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Package {
    pub root: PathBuf,
    pub src_dir: PathBuf,
    pub entry: Vec<String>,
    pub modules: HashMap<Vec<String>, ModuleInfo>,
    pub dependencies: HashMap<String, Dependency>,
}

#[derive(Debug)]
pub enum LoadError {
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
        path: Vec<String>,
    },
    EntryNotFound {
        path: PathBuf,
    },
    Manifest {
        path: PathBuf,
        message: String,
    },
    Diagnostic(Box<Diagnostic>),
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
            LoadError::Manifest { path, message } => {
                write!(f, "Failed to read {}: {}", path.display(), message)
            }
            LoadError::Diagnostic(diagnostic) => write!(f, "{}", diagnostic),
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
    let manifest = load_manifest(root)?;
    let src_dir = pick_source_dir(root);
    let entry_path = module_path_from_file(&src_dir, entry);
    let mut loader = ModuleLoader::new(root.to_path_buf(), src_dir, manifest.dependencies);
    loader.load_entry(entry, entry_path.clone())?;
    loader.resolve_exports()?;
    Ok(Package {
        root: loader.root,
        src_dir: loader.src_dir,
        entry: entry_path,
        modules: loader.modules,
        dependencies: loader.dependencies,
    })
}

struct ModuleLoader {
    root: PathBuf,
    src_dir: PathBuf,
    modules: HashMap<Vec<String>, ModuleInfo>,
    loading: HashSet<Vec<String>>,
    dependencies: HashMap<String, Dependency>,
    export_cache: HashMap<Vec<String>, HashSet<String>>,
    export_visiting: HashSet<Vec<String>>,
}

impl ModuleLoader {
    fn new(root: PathBuf, src_dir: PathBuf, dependencies: HashMap<String, Dependency>) -> Self {
        Self {
            root,
            src_dir,
            modules: HashMap::new(),
            loading: HashSet::new(),
            dependencies,
            export_cache: HashMap::new(),
            export_visiting: HashSet::new(),
        }
    }

    fn load_entry(&mut self, entry: &Path, path: Vec<String>) -> Result<(), LoadError> {
        let info = self.load_file(entry, path.clone())?;
        let uses = info.uses.clone();
        self.modules.insert(path.clone(), info);
        for decl in uses {
            let module_path = if !decl.symbols.is_empty() {
                if !is_std_path(&decl.path) && self.resolve_module_file(&decl.path).is_none() {
                    return Err(self.missing_module_error(&path, &decl, &decl.path));
                }
                decl.path.clone()
            } else {
                let target = self.resolve_use_target(&path, &decl)?;
                match target {
                    UseTarget::Module { path } => path,
                    UseTarget::Symbol { module_path, .. } => module_path,
                }
            };
            if is_std_path(&module_path) {
                continue;
            }
            self.load_module_path(module_path)?;
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
        let uses = info.uses.clone();
        self.modules.insert(path.clone(), info);
        for decl in uses {
            let module_path = if !decl.symbols.is_empty() {
                if !is_std_path(&decl.path) && self.resolve_module_file(&decl.path).is_none() {
                    return Err(self.missing_module_error(&path, &decl, &decl.path));
                }
                decl.path.clone()
            } else {
                let target = self.resolve_use_target(&path, &decl)?;
                match target {
                    UseTarget::Module { path } => path,
                    UseTarget::Symbol { module_path, .. } => module_path,
                }
            };
            if is_std_path(&module_path) {
                continue;
            }
            self.load_module_path(module_path)?;
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
                error: Box::new(err),
            })?;
        let uses = collect_uses(&module);
        Ok(ModuleInfo {
            path: module_path,
            file: path.to_path_buf(),
            source,
            module,
            uses,
            resolved_uses: Vec::new(),
            exports: HashSet::new(),
        })
    }

    fn resolve_exports(&mut self) -> Result<(), LoadError> {
        let module_paths: Vec<Vec<String>> = self.modules.keys().cloned().collect();
        for path in &module_paths {
            let exports = self.public_exports(path)?;
            if let Some(info) = self.modules.get_mut(path) {
                info.exports = exports;
            }
        }
        for path in &module_paths {
            let resolved = self.resolve_module_uses(path)?;
            if let Some(info) = self.modules.get_mut(path) {
                info.resolved_uses = resolved;
            }
        }
        Ok(())
    }

    fn resolve_module_uses(
        &mut self,
        module_path: &[String],
    ) -> Result<Vec<Vec<ResolvedUse>>, LoadError> {
        let uses = self
            .modules
            .get(module_path)
            .ok_or_else(|| LoadError::MissingModule {
                path: module_path.to_vec(),
            })?
            .uses
            .clone();
        let mut resolved = Vec::new();
        for decl in &uses {
            let mut group = Vec::new();
            if !decl.symbols.is_empty() {
                let target_path = decl.path.clone();
                if !is_std_path(&target_path) && self.resolve_module_file(&target_path).is_none() {
                    return Err(self.missing_module_error(module_path, decl, &target_path));
                }
                for symbol in &decl.symbols {
                    if !is_std_path(&target_path) {
                        let exports = self.public_exports(&target_path)?;
                        if !exports.contains(&symbol.name) {
                            return Err(self.visibility_error(
                                module_path,
                                decl,
                                Some(symbol.span.clone()),
                                "Symbol is private",
                            ));
                        }
                    }
                    group.push(ResolvedUse {
                        alias: symbol.name.clone(),
                        target: UseTarget::Symbol {
                            module_path: target_path.clone(),
                            symbol: symbol.name.clone(),
                        },
                        is_pub: decl.is_pub,
                        spans: decl.spans.clone(),
                    });
                }
                resolved.push(group);
                continue;
            }
            let target = self.resolve_use_target(module_path, decl)?;
            match &target {
                UseTarget::Module { path } => {
                    let name = decl
                        .alias
                        .clone()
                        .unwrap_or_else(|| path.last().cloned().unwrap_or_default());
                    group.push(ResolvedUse {
                        alias: name,
                        target,
                        is_pub: decl.is_pub,
                        spans: decl.spans.clone(),
                    });
                }
                UseTarget::Symbol {
                    module_path,
                    symbol,
                } => {
                    let exports = self.public_exports(module_path)?;
                    if !exports.contains(symbol) {
                        return Err(self.visibility_error(
                            module_path,
                            decl,
                            None,
                            "Symbol is private",
                        ));
                    }
                    let name = decl.alias.clone().unwrap_or_else(|| symbol.clone());
                    group.push(ResolvedUse {
                        alias: name,
                        target,
                        is_pub: decl.is_pub,
                        spans: decl.spans.clone(),
                    });
                }
            }
            resolved.push(group);
        }
        Ok(resolved)
    }

    fn public_exports(&mut self, path: &[String]) -> Result<HashSet<String>, LoadError> {
        if let Some(exports) = self.export_cache.get(path) {
            return Ok(exports.clone());
        }
        if self.export_visiting.contains(path) {
            return Err(LoadError::Cycle {
                path: path.to_vec(),
            });
        }
        self.export_visiting.insert(path.to_vec());
        let items = self
            .modules
            .get(path)
            .ok_or_else(|| LoadError::MissingModule {
                path: path.to_vec(),
            })?
            .module
            .items
            .clone();
        let mut exports = HashSet::new();
        for item in &items {
            match item {
                Item::Fn(decl) if decl.is_pub => {
                    exports.insert(decl.name.clone());
                }
                Item::Type(decl) if decl.is_pub => {
                    exports.insert(decl.name.clone());
                }
                Item::Enum(decl) if decl.is_pub => {
                    exports.insert(decl.name.clone());
                }
                _ => {}
            }
        }
        for item in &items {
            if let Item::Use(decl) = item {
                if !decl.is_pub {
                    continue;
                }
                if !decl.symbols.is_empty() {
                    let target_path = decl.path.clone();
                    if !is_std_path(&target_path)
                        && self.resolve_module_file(&target_path).is_none()
                    {
                        return Err(self.missing_module_error(path, decl, &target_path));
                    }
                    for symbol in &decl.symbols {
                        if !is_std_path(&target_path) {
                            let target_exports = self.public_exports(&target_path)?;
                            if !target_exports.contains(&symbol.name) {
                                return Err(self.visibility_error(
                                    path,
                                    decl,
                                    Some(symbol.span.clone()),
                                    "Symbol is private",
                                ));
                            }
                        }
                        exports.insert(symbol.name.clone());
                    }
                    continue;
                }
                let target = self.resolve_use_target(path, decl)?;
                match target {
                    UseTarget::Module { path: target_path } => {
                        let name = decl
                            .alias
                            .clone()
                            .unwrap_or_else(|| target_path.last().cloned().unwrap_or_default());
                        exports.insert(name);
                    }
                    UseTarget::Symbol {
                        module_path,
                        symbol,
                    } => {
                        let target_exports = self.public_exports(&module_path)?;
                        if !target_exports.contains(&symbol) {
                            return Err(self.visibility_error(
                                path,
                                decl,
                                None,
                                "Symbol is private",
                            ));
                        }
                        let name = decl.alias.clone().unwrap_or_else(|| symbol.clone());
                        exports.insert(name);
                    }
                }
            }
        }
        self.export_visiting.remove(path);
        self.export_cache.insert(path.to_vec(), exports.clone());
        Ok(exports)
    }

    fn resolve_use_target(
        &self,
        module_path: &[String],
        decl: &UseDecl,
    ) -> Result<UseTarget, LoadError> {
        if !decl.symbols.is_empty() {
            return Err(self.use_error(
                module_path,
                "Use lists are not supported here",
                &decl.spans,
            ));
        }
        if decl.path.is_empty() {
            return Err(self.use_error(module_path, "Empty use path", &decl.spans));
        }
        if is_std_path(&decl.path) {
            return Ok(UseTarget::Module {
                path: decl.path.clone(),
            });
        }
        if self.resolve_module_file(&decl.path).is_some() {
            return Ok(UseTarget::Module {
                path: decl.path.clone(),
            });
        }
        if decl.path.len() == 1 {
            return Err(self.missing_module_error(module_path, decl, &decl.path));
        }
        let module_part = &decl.path[..decl.path.len() - 1];
        if self.resolve_module_file(module_part).is_some() {
            let symbol = decl.path.last().cloned().unwrap_or_default();
            return Ok(UseTarget::Symbol {
                module_path: module_part.to_vec(),
                symbol,
            });
        }
        Err(self.missing_module_error(module_path, decl, module_part))
    }

    fn missing_module_error(
        &self,
        module_path: &[String],
        decl: &UseDecl,
        missing: &[String],
    ) -> LoadError {
        if !self.modules.contains_key(module_path) {
            return LoadError::MissingModule {
                path: missing.to_vec(),
            };
        }
        if let Some(dep) = missing.first().and_then(|name| self.dependencies.get(name)) {
            if !dep.root.is_dir() {
                return self.use_error(
                    module_path,
                    &format!("Dependency path not found: {}", dep.root.display()),
                    &decl.spans,
                );
            }
        }
        self.use_error(
            module_path,
            &format!("Module not found: {}", missing.join(".")),
            &decl.spans,
        )
    }

    fn use_error(&self, module_path: &[String], message: &str, spans: &[Span]) -> LoadError {
        let info = match self.modules.get(module_path) {
            Some(info) => info,
            None => {
                return LoadError::MissingModule {
                    path: module_path.to_vec(),
                }
            }
        };
        let mut diagnostic = Diagnostic::new(message, Some(info.file.to_string_lossy().as_ref()));
        if let Some(span) = join_spans(spans) {
            diagnostic = diagnostic.with_span("use path", span);
        }
        diagnostic = diagnostic.with_source(&info.source);
        LoadError::Diagnostic(Box::new(diagnostic))
    }

    fn visibility_error(
        &self,
        module_path: &[String],
        decl: &UseDecl,
        symbol_span: Option<Span>,
        message: &str,
    ) -> LoadError {
        let info = match self.modules.get(module_path) {
            Some(info) => info,
            None => {
                return LoadError::MissingModule {
                    path: module_path.to_vec(),
                }
            }
        };
        let mut diagnostic = Diagnostic::new(message, Some(info.file.to_string_lossy().as_ref()));
        if let Some(span) = symbol_span {
            if let Some(module_span) = join_spans(&decl.spans) {
                diagnostic = diagnostic.with_span("module", module_span);
            }
            diagnostic = diagnostic.with_span("symbol", span);
        } else if decl.spans.len() >= 2 {
            if let Some(module_span) = join_spans(&decl.spans[..decl.spans.len() - 1]) {
                diagnostic = diagnostic.with_span("module", module_span);
            }
            if let Some(symbol_span) = decl.spans.last().cloned() {
                diagnostic = diagnostic.with_span("symbol", symbol_span);
            }
        } else if let Some(span) = join_spans(&decl.spans) {
            diagnostic = diagnostic.with_span("use path", span);
        }
        diagnostic = diagnostic.with_source(&info.source);
        LoadError::Diagnostic(Box::new(diagnostic))
    }

    fn resolve_module_file(&self, path: &[String]) -> Option<PathBuf> {
        if path.is_empty() {
            return None;
        }
        if let Some(dep) = self.dependencies.get(&path[0]) {
            return resolve_in_src(&dep.src_dir, &path[1..]);
        }
        resolve_in_src(&self.src_dir, path)
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

fn resolve_in_src(src_dir: &Path, path: &[String]) -> Option<PathBuf> {
    if path.is_empty() {
        let lib = src_dir.join("lib.morph");
        if lib.is_file() {
            return Some(lib);
        }
        let index = src_dir.join("index.morph");
        if index.is_file() {
            return Some(index);
        }
        let main = src_dir.join("main.morph");
        if main.is_file() {
            return Some(main);
        }
        return None;
    }
    let mut base = src_dir.to_path_buf();
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

fn join_spans(spans: &[Span]) -> Option<Span> {
    let first = spans.first()?;
    let last = spans.last()?;
    if first.line != last.line {
        return Some(Span::single(first.line, first.col));
    }
    Some(Span {
        line: first.line,
        col: first.col,
        end_col: last.end_col,
    })
}

fn load_manifest(root: &Path) -> Result<Manifest, LoadError> {
    let manifest_path = root.join("morph.toml");
    if !manifest_path.is_file() {
        return Ok(Manifest::default());
    }
    let source = fs::read_to_string(&manifest_path).map_err(|err| LoadError::Manifest {
        path: manifest_path.clone(),
        message: err.to_string(),
    })?;
    let value = source.parse::<Value>().map_err(|err| LoadError::Manifest {
        path: manifest_path.clone(),
        message: err.to_string(),
    })?;
    let mut dependencies = HashMap::new();
    if let Some(table) = value.get("dependencies").and_then(|val| val.as_table()) {
        for (name, entry) in table {
            if let Some(path) = dependency_path(entry, root) {
                let src_dir = pick_source_dir(&path);
                dependencies.insert(
                    name.to_string(),
                    Dependency {
                        name: name.to_string(),
                        root: path,
                        src_dir,
                    },
                );
            }
        }
    }
    Ok(Manifest { dependencies })
}

fn dependency_path(value: &Value, root: &Path) -> Option<PathBuf> {
    if let Some(path_value) = value.get("path").and_then(|val| val.as_str()) {
        return Some(root.join(path_value));
    }
    if let Some(path_value) = value.as_str() {
        let candidate = root.join(path_value);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
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

fn collect_uses(module: &Module) -> Vec<UseDecl> {
    let mut uses = Vec::new();
    for item in &module.items {
        if let Item::Use(decl) = item {
            uses.push(decl.clone());
        }
    }
    uses
}

fn is_std_path(path: &[String]) -> bool {
    matches!(path.first(), Some(segment) if segment == "std")
}
