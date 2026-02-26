use std::collections::{HashMap, HashSet};

use bumpalo::Bump;

use crate::arena::CompilerArena;
use crate::ast::*;
use crate::bytecode::{ByteFunction, Chunk, Constant, FfiSignature, FfiType, Instruction, Program};
use crate::diagnostic::{Diagnostic, Span};
use crate::modules::{ModuleId, ModuleInfo, Package};
use crate::symbols::SymbolTable;

#[derive(Debug)]
pub struct CompileError {
    pub message: String,
    pub span: Option<Span>,
    pub source_name: Option<String>,
    pub source: Option<String>,
}

impl CompileError {
    fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
            span: None,
            source_name: None,
            source: None,
        }
    }

    fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    fn with_source(mut self, source_name: &str, source: &str) -> Self {
        self.source_name = Some(source_name.to_string());
        self.source = Some(source.to_string());
        self
    }

    pub fn diagnostic(&self) -> Option<Diagnostic> {
        let span = self.span.clone()?;
        let source = self.source.as_ref()?;
        let diagnostic = Diagnostic::new(&self.message, self.source_name.as_deref())
            .with_span("here", span)
            .with_source(source);
        Some(diagnostic)
    }
}

#[derive(Debug, Clone, Copy)]
enum ResolvedVar {
    Local(u16),
    Global(u16),
}

pub fn compile_module(module: &Module) -> Result<Program, CompileError> {
    let arena = CompilerArena::new();
    let mut ctx = ProgramBuilder::new(&arena.bump);
    ctx.define_builtin_globals();
    let info = ModuleInfo {
        id: ModuleId(vec!["main".to_string()]),
        file: std::path::PathBuf::new(),
        source: String::new(),
        module: module.clone(),
        imports: Vec::new(),
        import_aliases: std::collections::HashMap::new(),
        exports: std::collections::HashSet::new(),
    };
    let main_fn = ctx.compile_module_items(&info, None)?;
    Ok(Program {
        functions: ctx.functions,
        main: main_fn,
        globals: ctx.global_names,
        global_inits: ctx.global_inits,
    })
}

pub fn compile_package(package: &Package) -> Result<Program, CompileError> {
    let arena = CompilerArena::new();
    let mut ctx = ProgramBuilder::new(&arena.bump);
    ctx.define_builtin_globals();
    let mut module_inits: Vec<(ModuleId, u16)> = Vec::new();
    for id in &package.order {
        let info = package
            .modules
            .get(id)
            .ok_or_else(|| CompileError::new("Missing module info"))?;
        let init_idx = ctx.compile_module_items(info, Some(package))?;
        module_inits.push((id.clone(), init_idx));
    }
    let entry_id = package.entry.clone();
    let mut chunk = Chunk::new();
    for (id, init_idx) in &module_inits {
        let is_entry = *id == entry_id;
        let const_idx = chunk.add_constant(Constant::Function(*init_idx));
        chunk.write(Instruction::Const(const_idx), 0);
        chunk.write(Instruction::Call(0), 0);
        if !is_entry {
            chunk.write(Instruction::Pop, 0);
        }
    }
    chunk.write(Instruction::Return, 0);
    let bootstrap = ByteFunction {
        name: Some("<bootstrap>".to_string()),
        arity: 0,
        chunk,
        source_name: None,
    };
    let bootstrap_idx = ctx.functions.len() as u16;
    ctx.functions.push(bootstrap);
    Ok(Program {
        functions: ctx.functions,
        main: bootstrap_idx,
        globals: ctx.global_names,
        global_inits: ctx.global_inits,
    })
}

struct ProgramBuilder<'a> {
    _bump: &'a Bump,
    globals: HashMap<String, u16>,
    global_names: Vec<String>,
    global_inits: Vec<Option<Constant>>,
    functions: Vec<ByteFunction>,
    method_tables: HashSet<String>,
    tool_namespaces: HashSet<String>,
    _symbols: SymbolTable,
}

#[derive(Clone)]
struct ModuleContext {
    exports: Vec<String>,
    record_global: u16,
    prefix: String,
    import_aliases: HashMap<String, ModuleId>,
    import_exports: HashMap<String, HashSet<String>>,
    source_name: Option<String>,
}

impl<'a> ProgramBuilder<'a> {
    fn new(bump: &'a Bump) -> Self {
        Self {
            _bump: bump,
            globals: HashMap::new(),
            global_names: Vec::new(),
            global_inits: Vec::new(),
            functions: Vec::new(),
            method_tables: HashSet::new(),
            tool_namespaces: HashSet::new(),
            _symbols: SymbolTable::new(),
        }
    }

    fn define_builtin_globals(&mut self) {
        let _ = self.ensure_global("print");
        let _ = self.ensure_global("task");
        let _ = self.ensure_global("chan");
        let _ = self.ensure_global("net");
        let _ = self.ensure_global("http");
        let _ = self.ensure_global("policy");
        let _ = self.ensure_global("json");
        let _ = self.ensure_global("tokenizer");
        let _ = self.ensure_global("dataset");
        let _ = self.ensure_global("checkpoint");
    }

    fn ensure_global(&mut self, name: &str) -> u16 {
        if let Some(idx) = self.globals.get(name) {
            *idx
        } else {
            let idx = self.global_names.len() as u16;
            self.global_names.push(name.to_string());
            self.global_inits.push(None);
            self.globals.insert(name.to_string(), idx);
            idx
        }
    }

    fn compile_type_constructor(
        &mut self,
        module: &ModuleContext,
        decl: &TypeDecl,
        full_name: &str,
    ) -> u16 {
        let mut chunk = Chunk::new();
        let line = 0;
        let key_idx = chunk.add_constant(Constant::String("__type".to_string()));
        let val_idx = chunk.add_constant(Constant::String(full_name.to_string()));
        chunk.write(Instruction::Const(key_idx), line);
        chunk.write(Instruction::Const(val_idx), line);
        for (idx, field) in decl.fields.iter().enumerate() {
            let key_idx = chunk.add_constant(Constant::String(field.name.clone()));
            chunk.write(Instruction::Const(key_idx), line);
            chunk.write(Instruction::LoadLocal(idx as u16), line);
        }
        chunk.write(
            Instruction::MakeRecord((decl.fields.len() + 1) as u16),
            line,
        );
        chunk.write(Instruction::Return, line);
        let func = ByteFunction {
            name: Some(decl.name.clone()),
            arity: decl.fields.len() as u16,
            chunk,
            source_name: module.source_name.clone(),
        };
        let idx = self.functions.len() as u16;
        self.functions.push(func);
        idx
    }

    fn compile_tool_stub(&mut self, module: &ModuleContext, name: &str, params: &[Param]) -> u16 {
        let mut chunk = Chunk::new();
        let line = 0;
        let null_idx = chunk.add_constant(Constant::Null);
        chunk.write(Instruction::Const(null_idx), line);
        chunk.write(Instruction::TryUnwrap, line);
        chunk.write(Instruction::Return, line);
        let func = ByteFunction {
            name: Some(name.to_string()),
            arity: params.len() as u16,
            chunk,
            source_name: module.source_name.clone(),
        };
        let idx = self.functions.len() as u16;
        self.functions.push(func);
        idx
    }

    fn compile_module_items(
        &mut self,
        info: &ModuleInfo,
        package: Option<&Package>,
    ) -> Result<u16, CompileError> {
        let module_prefix = module_prefix(&info.id);
        let module_record_name = module_record_name(&info.id);
        let module_record_global = self.ensure_global(&module_record_name);
        let exports = info.exports.iter().cloned().collect::<Vec<_>>();
        let mut import_exports = HashMap::new();
        if let Some(package) = package {
            for (alias, module_id) in &info.import_aliases {
                let exports = package
                    .modules
                    .get(module_id)
                    .ok_or_else(|| CompileError::new("Missing module exports"))?
                    .exports
                    .clone();
                import_exports.insert(alias.clone(), exports);
            }
        }
        let module_ctx = ModuleContext {
            exports,
            record_global: module_record_global,
            prefix: module_prefix,
            import_aliases: info.import_aliases.clone(),
            import_exports,
            source_name: if info.file.as_os_str().is_empty() {
                None
            } else {
                Some(info.file.to_string_lossy().to_string())
            },
        };
        let mut f = FunctionBuilder::new("<module_init>", &[], self, true, module_ctx);
        if let Err(err) = f.compile_items(&info.module.items) {
            if info.file.as_os_str().is_empty() {
                return Err(err);
            }
            return Err(err.with_source(info.file.to_string_lossy().as_ref(), &info.source));
        }
        let function = f.finish();
        let func_index = self.functions.len() as u16;
        self.functions.push(function);
        Ok(func_index)
    }

    fn compile_function(
        &mut self,
        module: &ModuleContext,
        name: &str,
        params: &[Param],
        body_items: &[Item],
    ) -> Result<u16, CompileError> {
        let mut f = FunctionBuilder::new(name, params, self, false, module.clone());
        f.compile_items(body_items)?;
        let function = f.finish();
        let func_index = self.functions.len() as u16;
        self.functions.push(function);
        Ok(func_index)
    }
}

struct FunctionBuilder<'a, 'p> {
    name: String,
    params: Vec<Param>,
    chunk: Chunk,
    enclosing: &'p mut ProgramBuilder<'a>,
    is_root: bool,
    scopes: Vec<std::collections::HashMap<String, u16>>,
    next_local: Vec<u16>,
    module: ModuleContext,
}

impl<'a, 'p> FunctionBuilder<'a, 'p> {
    fn new(
        name: &str,
        params: &[Param],
        enclosing: &'p mut ProgramBuilder<'a>,
        is_root: bool,
        module: ModuleContext,
    ) -> Self {
        let mut scopes = Vec::new();
        let mut next_local = Vec::new();
        // function scope
        scopes.push(std::collections::HashMap::new());
        next_local.push(0);
        Self {
            name: name.to_string(),
            params: params.to_vec(),
            chunk: Chunk::new(),
            enclosing,
            is_root,
            scopes,
            next_local,
            module,
        }
    }

    fn compile_items(&mut self, items: &[Item]) -> Result<(), CompileError> {
        // define params as locals
        let param_names: Vec<String> = self.params.iter().map(|p| p.name.clone()).collect();
        for name in param_names {
            self.define_local(&name);
        }
        if self.is_root {
            self.emit_imports()?;
        }
        let mut return_slot: Option<u16> = None;
        for (i, item) in items.iter().enumerate() {
            let is_last = i + 1 == items.len();
            match item {
                Item::Stmt(Stmt::Expr(expr)) if is_last => {
                    if self.is_root && !self.module.exports.is_empty() {
                        let slot = self.add_local("__last");
                        self.compile_expr(expr)?;
                        self.chunk
                            .write(Instruction::StoreLocal(slot), line_of_expr(expr));
                        return_slot = Some(slot);
                    } else {
                        self.compile_expr(expr)?;
                        self.chunk.write(Instruction::Return, line_of_expr(expr));
                        return Ok(());
                    }
                }
                Item::Stmt(stmt) => self.compile_stmt(stmt)?,
                Item::Import(_) => {
                    // imports handled in emit_imports
                }
                Item::NativeImport(decl) => {
                    for func in &decl.functions {
                        let mut params = Vec::new();
                        for param in &func.params {
                            let ffi_ty = ffi_type_from_ref(&param.type_ann).ok_or_else(|| {
                                CompileError::new("Unsupported FFI parameter type")
                                    .with_span(param.name_span.clone())
                            })?;
                            if matches!(ffi_ty, FfiType::Void) {
                                return Err(CompileError::new("FFI parameter cannot be Void")
                                    .with_span(param.name_span.clone()));
                            }
                            params.push(ffi_ty);
                        }
                        let ret = ffi_type_from_ref(&func.return_type).ok_or_else(|| {
                            CompileError::new("Unsupported FFI return type")
                                .with_span(func.name_span.clone())
                        })?;
                        let global_name = mangle_symbol(&self.module.prefix, &func.name);
                        let global = self.enclosing.ensure_global(&global_name);
                        let signature = FfiSignature { params, ret };
                        let native_decl = crate::bytecode::NativeFunctionDecl {
                            library: decl.library.clone(),
                            name: func.name.clone(),
                            signature,
                        };
                        let const_idx = self
                            .chunk
                            .add_constant(Constant::NativeFunction(native_decl));
                        self.chunk
                            .write(Instruction::Const(const_idx), func.name_span.line);
                        self.chunk
                            .write(Instruction::StoreGlobal(global), func.name_span.line);
                    }
                }
                Item::Use(_) => {
                    // handled by loader; no runtime effect
                }
                Item::Fn(decl) => {
                    let global_name = mangle_symbol(&self.module.prefix, &decl.name);
                    let global = self.enclosing.ensure_global(&global_name);
                    // wrap statements as items for compiler
                    let body_items: Vec<Item> =
                        decl.body.stmts.iter().cloned().map(Item::Stmt).collect();
                    let func_idx = self.enclosing.compile_function(
                        &self.module,
                        &decl.name,
                        &decl.params,
                        &body_items,
                    )?;
                    if let Some(slot) = self.enclosing.global_inits.get_mut(global as usize) {
                        *slot = Some(Constant::Function(func_idx));
                    }
                    let const_idx = self.chunk.add_constant(Constant::Function(func_idx));
                    self.chunk
                        .write(Instruction::Const(const_idx), decl.name_span.line);
                    self.chunk
                        .write(Instruction::StoreGlobal(global), decl.name_span.line);
                }
                Item::Type(decl) => {
                    let full_name = mangle_symbol(&self.module.prefix, &decl.name);
                    let global = self.enclosing.ensure_global(&full_name);
                    let func_idx =
                        self.enclosing
                            .compile_type_constructor(&self.module, decl, &full_name);
                    if let Some(slot) = self.enclosing.global_inits.get_mut(global as usize) {
                        *slot = Some(Constant::Function(func_idx));
                    }
                    let const_idx = self.chunk.add_constant(Constant::Function(func_idx));
                    self.chunk.write(Instruction::Const(const_idx), 0);
                    self.chunk.write(Instruction::StoreGlobal(global), 0);
                }
                Item::Enum(decl) => {
                    let full_name = mangle_symbol(&self.module.prefix, &decl.name);
                    let global = self.enclosing.ensure_global(&full_name);
                    for variant in &decl.variants {
                        let key_idx = self.chunk.add_constant(Constant::String(variant.clone()));
                        self.chunk.write(Instruction::Const(key_idx), 0);
                        let type_key = self
                            .chunk
                            .add_constant(Constant::String("__type".to_string()));
                        let type_val = self.chunk.add_constant(Constant::String(full_name.clone()));
                        self.chunk.write(Instruction::Const(type_key), 0);
                        self.chunk.write(Instruction::Const(type_val), 0);
                        let var_key = self
                            .chunk
                            .add_constant(Constant::String("__variant".to_string()));
                        let var_val = self.chunk.add_constant(Constant::String(variant.clone()));
                        self.chunk.write(Instruction::Const(var_key), 0);
                        self.chunk.write(Instruction::Const(var_val), 0);
                        self.chunk.write(Instruction::MakeRecord(2), 0);
                    }
                    self.chunk
                        .write(Instruction::MakeRecord(decl.variants.len() as u16), 0);
                    self.chunk.write(Instruction::StoreGlobal(global), 0);
                }
                Item::Impl(decl) => {
                    let full_name = mangle_symbol(&self.module.prefix, &decl.name);
                    let table_idx = self.ensure_method_table(&full_name, 0);
                    for method in &decl.methods {
                        let mut params = Vec::with_capacity(method.params.len() + 1);
                        params.push(Param {
                            name: "self".to_string(),
                            type_ann: Some(TypeRef::Named {
                                path: vec![decl.name.clone()],
                                args: Vec::new(),
                                optional: false,
                            }),
                            default: None,
                        });
                        params.extend(method.params.clone());
                        let body_items: Vec<Item> =
                            method.body.stmts.iter().cloned().map(Item::Stmt).collect();
                        let method_name = format!("{}::{}", decl.name, method.name);
                        let func_idx = self.enclosing.compile_function(
                            &self.module,
                            &method_name,
                            &params,
                            &body_items,
                        )?;
                        self.chunk.write(Instruction::LoadGlobal(table_idx), 0);
                        let const_idx = self.chunk.add_constant(Constant::Function(func_idx));
                        self.chunk.write(Instruction::Const(const_idx), 0);
                        let key_idx = self
                            .chunk
                            .add_constant(Constant::String(method.name.clone()));
                        self.chunk.write(Instruction::SetField(key_idx), 0);
                    }
                }
                Item::Tool(decl) => {
                    if decl.path.is_empty() {
                        return Err(CompileError::new("Tool path cannot be empty"));
                    }
                    let stub_name = format!("tool.{}", decl.path.join("."));
                    let func_idx =
                        self.enclosing
                            .compile_tool_stub(&self.module, &stub_name, &decl.params);
                    if decl.path.len() == 1 {
                        let global = self
                            .enclosing
                            .ensure_global(&mangle_symbol(&self.module.prefix, &decl.path[0]));
                        let const_idx = self.chunk.add_constant(Constant::Function(func_idx));
                        self.chunk.write(Instruction::Const(const_idx), 0);
                        self.chunk.write(Instruction::StoreGlobal(global), 0);
                    } else {
                        let parent = &decl.path[..decl.path.len() - 1];
                        self.ensure_tool_namespace(parent, 0)?;
                        self.emit_load_path_record(parent, 0)?;
                        let const_idx = self.chunk.add_constant(Constant::Function(func_idx));
                        self.chunk.write(Instruction::Const(const_idx), 0);
                        let key_idx = self
                            .chunk
                            .add_constant(Constant::String(decl.path.last().unwrap().clone()));
                        self.chunk.write(Instruction::SetField(key_idx), 0);
                    }
                }
                Item::Prompt(decl) => {
                    let global = self
                        .enclosing
                        .ensure_global(&mangle_symbol(&self.module.prefix, &decl.name));
                    let key_kind = self
                        .chunk
                        .add_constant(Constant::String("__kind".to_string()));
                    let val_kind = self
                        .chunk
                        .add_constant(Constant::String("prompt".to_string()));
                    self.chunk.write(Instruction::Const(key_kind), 0);
                    self.chunk.write(Instruction::Const(val_kind), 0);
                    let key_name = self
                        .chunk
                        .add_constant(Constant::String("name".to_string()));
                    let val_name = self.chunk.add_constant(Constant::String(decl.name.clone()));
                    self.chunk.write(Instruction::Const(key_name), 0);
                    self.chunk.write(Instruction::Const(val_name), 0);
                    let key_template = self
                        .chunk
                        .add_constant(Constant::String("template".to_string()));
                    self.chunk.write(Instruction::Const(key_template), 0);
                    if let Some(text) = &decl.template {
                        let val_template = self.chunk.add_constant(Constant::String(text.clone()));
                        self.chunk.write(Instruction::Const(val_template), 0);
                    } else {
                        let null_idx = self.chunk.add_constant(Constant::Null);
                        self.chunk.write(Instruction::Const(null_idx), 0);
                    }
                    let key_inputs = self
                        .chunk
                        .add_constant(Constant::String("inputs".to_string()));
                    self.chunk.write(Instruction::Const(key_inputs), 0);
                    for field in &decl.input_fields {
                        let name_idx = self
                            .chunk
                            .add_constant(Constant::String(field.name.clone()));
                        self.chunk.write(Instruction::Const(name_idx), 0);
                    }
                    self.chunk
                        .write(Instruction::MakeList(decl.input_fields.len() as u16), 0);
                    self.chunk.write(Instruction::MakeRecord(4), 0);
                    self.chunk.write(Instruction::StoreGlobal(global), 0);
                }
                Item::Model(decl) => {
                    let global = self
                        .enclosing
                        .ensure_global(&mangle_symbol(&self.module.prefix, &decl.name));
                    let key_kind = self
                        .chunk
                        .add_constant(Constant::String("__kind".to_string()));
                    let val_kind = self
                        .chunk
                        .add_constant(Constant::String("model".to_string()));
                    self.chunk.write(Instruction::Const(key_kind), 0);
                    self.chunk.write(Instruction::Const(val_kind), 0);
                    let key_value = self
                        .chunk
                        .add_constant(Constant::String("value".to_string()));
                    self.chunk.write(Instruction::Const(key_value), 0);
                    self.compile_expr(&decl.expr)?;
                    self.chunk.write(Instruction::MakeRecord(2), 0);
                    self.chunk.write(Instruction::StoreGlobal(global), 0);
                }
                Item::Agent(decl) => {
                    let global = self
                        .enclosing
                        .ensure_global(&mangle_symbol(&self.module.prefix, &decl.name));
                    let key_kind = self
                        .chunk
                        .add_constant(Constant::String("__kind".to_string()));
                    let val_kind = self
                        .chunk
                        .add_constant(Constant::String("agent".to_string()));
                    self.chunk.write(Instruction::Const(key_kind), 0);
                    self.chunk.write(Instruction::Const(val_kind), 0);
                    self.chunk.write(Instruction::MakeRecord(1), 0);
                    self.chunk.write(Instruction::StoreGlobal(global), 0);
                    for item in &decl.items {
                        match item {
                            AgentItem::PolicyUse(name) => {
                                self.chunk.write(Instruction::LoadGlobal(global), 0);
                                let val = self.chunk.add_constant(Constant::String(name.clone()));
                                self.chunk.write(Instruction::Const(val), 0);
                                let key = self
                                    .chunk
                                    .add_constant(Constant::String("policy_name".to_string()));
                                self.chunk.write(Instruction::SetField(key), 0);
                            }
                            AgentItem::Memory(mem) => {
                                self.chunk.write(Instruction::LoadGlobal(global), 0);
                                let key = self.chunk.add_constant(Constant::String(match mem {
                                    MemoryDecl::Path { name, .. }
                                    | MemoryDecl::Expr { name, .. } => name.clone(),
                                }));
                                let mk_key = self
                                    .chunk
                                    .add_constant(Constant::String("__kind".to_string()));
                                let mk_val = self
                                    .chunk
                                    .add_constant(Constant::String("memory".to_string()));
                                self.chunk.write(Instruction::Const(mk_key), 0);
                                self.chunk.write(Instruction::Const(mk_val), 0);
                                match mem {
                                    MemoryDecl::Path { path, .. } => {
                                        let pkey = self
                                            .chunk
                                            .add_constant(Constant::String("path".to_string()));
                                        let pval =
                                            self.chunk.add_constant(Constant::String(path.clone()));
                                        self.chunk.write(Instruction::Const(pkey), 0);
                                        self.chunk.write(Instruction::Const(pval), 0);
                                        self.chunk.write(Instruction::MakeRecord(2), 0);
                                    }
                                    MemoryDecl::Expr { expr, .. } => {
                                        let ekey = self
                                            .chunk
                                            .add_constant(Constant::String("expr".to_string()));
                                        self.chunk.write(Instruction::Const(ekey), 0);
                                        self.compile_expr(expr)?;
                                        self.chunk.write(Instruction::MakeRecord(2), 0);
                                    }
                                }
                                self.chunk.write(Instruction::SetField(key), 0);
                            }
                            AgentItem::Fn(func) => {
                                let body_items: Vec<Item> =
                                    func.body.stmts.iter().cloned().map(Item::Stmt).collect();
                                let func_idx = self.enclosing.compile_function(
                                    &self.module,
                                    &func.name,
                                    &func.params,
                                    &body_items,
                                )?;
                                self.chunk.write(Instruction::LoadGlobal(global), 0);
                                let const_idx =
                                    self.chunk.add_constant(Constant::Function(func_idx));
                                self.chunk.write(Instruction::Const(const_idx), 0);
                                let key_idx =
                                    self.chunk.add_constant(Constant::String(func.name.clone()));
                                self.chunk.write(Instruction::SetField(key_idx), 0);
                            }
                            AgentItem::Stmt(_) => {
                                // Agent bodies are not executed in v1.1 runtime.
                            }
                        }
                    }
                }
                Item::Policy(decl) => {
                    self.compile_policy_decl(decl);
                }
            }
        }
        if self.is_root {
            self.emit_exports()?;
        }
        if let Some(slot) = return_slot {
            self.chunk.write(Instruction::LoadLocal(slot), 0);
            self.chunk.write(Instruction::Return, 0);
            return Ok(());
        }
        // implicit return null
        let null_idx = self.chunk.add_constant(Constant::Null);
        self.chunk.write(
            Instruction::Const(null_idx),
            items.last().map(line_for_item).unwrap_or(0),
        );
        self.chunk.write(Instruction::Return, 0);
        Ok(())
    }

    fn compile_policy_decl(&mut self, decl: &PolicyDecl) {
        let line = 0;
        let policy_global = self.enclosing.ensure_global("policy");
        self.chunk
            .write(Instruction::LoadGlobal(policy_global), line);
        let key_idx = self
            .chunk
            .add_constant(Constant::String("register".to_string()));
        self.chunk.write(Instruction::GetField(key_idx), line);
        let name_idx = self.chunk.add_constant(Constant::String(decl.name.clone()));
        self.chunk.write(Instruction::Const(name_idx), line);
        for rule in &decl.rules {
            let key_allow = self
                .chunk
                .add_constant(Constant::String("allow".to_string()));
            self.chunk.write(Instruction::Const(key_allow), line);
            let allow_idx = self.chunk.add_constant(Constant::Bool(rule.allow));
            self.chunk.write(Instruction::Const(allow_idx), line);
            let key_cap = self
                .chunk
                .add_constant(Constant::String("capability".to_string()));
            self.chunk.write(Instruction::Const(key_cap), line);
            for seg in &rule.capability {
                let seg_idx = self.chunk.add_constant(Constant::String(seg.clone()));
                self.chunk.write(Instruction::Const(seg_idx), line);
            }
            self.chunk
                .write(Instruction::MakeList(rule.capability.len() as u16), line);
            let key_filters = self
                .chunk
                .add_constant(Constant::String("filters".to_string()));
            self.chunk.write(Instruction::Const(key_filters), line);
            for filter in &rule.filters {
                let key_name = self
                    .chunk
                    .add_constant(Constant::String("name".to_string()));
                self.chunk.write(Instruction::Const(key_name), line);
                let name_idx = self
                    .chunk
                    .add_constant(Constant::String(filter.name.clone()));
                self.chunk.write(Instruction::Const(name_idx), line);
                let key_values = self
                    .chunk
                    .add_constant(Constant::String("values".to_string()));
                self.chunk.write(Instruction::Const(key_values), line);
                let values = policy_filter_values(filter);
                for value in &values {
                    let v_idx = self.chunk.add_constant(Constant::String(value.clone()));
                    self.chunk.write(Instruction::Const(v_idx), line);
                }
                self.chunk
                    .write(Instruction::MakeList(values.len() as u16), line);
                self.chunk.write(Instruction::MakeRecord(2), line);
            }
            self.chunk
                .write(Instruction::MakeList(rule.filters.len() as u16), line);
            self.chunk.write(Instruction::MakeRecord(3), line);
        }
        self.chunk
            .write(Instruction::MakeList(decl.rules.len() as u16), line);
        let is_default = decl.name == "default";
        let default_idx = self.chunk.add_constant(Constant::Bool(is_default));
        self.chunk.write(Instruction::Const(default_idx), line);
        self.chunk.write(Instruction::Call(3), line);
        self.chunk.write(Instruction::Pop, line);
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), CompileError> {
        match stmt {
            Stmt::Let { name, expr, .. } => {
                if self.is_root {
                    let g = self
                        .enclosing
                        .ensure_global(&mangle_symbol(&self.module.prefix, name));
                    self.compile_expr(expr)?;
                    self.chunk
                        .write(Instruction::DefineGlobal(g), line_of_expr(expr));
                } else {
                    let idx = self.add_local(name);
                    self.compile_expr(expr)?;
                    self.chunk
                        .write(Instruction::StoreLocal(idx), line_of_expr(expr));
                }
            }
            Stmt::Assign { target, expr } => {
                if target.accesses.is_empty() {
                    self.compile_expr(expr)?;
                    let resolved = self.resolve_var(&target.base)?;
                    match resolved {
                        ResolvedVar::Local(i) => self
                            .chunk
                            .write(Instruction::StoreLocal(i), target.base_span.line),
                        ResolvedVar::Global(i) => self
                            .chunk
                            .write(Instruction::StoreGlobal(i), target.base_span.line),
                    }
                } else {
                    let resolved = self.resolve_var(&target.base)?;
                    match resolved {
                        ResolvedVar::Local(i) => self
                            .chunk
                            .write(Instruction::LoadLocal(i), target.base_span.line),
                        ResolvedVar::Global(i) => self
                            .chunk
                            .write(Instruction::LoadGlobal(i), target.base_span.line),
                    }
                    let last = target.accesses.len() - 1;
                    for access in &target.accesses[..last] {
                        match access {
                            LValueAccess::Field(name) => {
                                let key_idx =
                                    self.chunk.add_constant(Constant::String(name.clone()));
                                self.chunk
                                    .write(Instruction::GetField(key_idx), target.base_span.line);
                            }
                            LValueAccess::Index(index) => {
                                self.compile_expr(index)?;
                                self.chunk.write(Instruction::GetIndex, line_of_expr(index));
                            }
                        }
                    }
                    match &target.accesses[last] {
                        LValueAccess::Field(name) => {
                            self.compile_expr(expr)?;
                            let key_idx = self.chunk.add_constant(Constant::String(name.clone()));
                            self.chunk
                                .write(Instruction::SetField(key_idx), target.base_span.line);
                        }
                        LValueAccess::Index(index) => {
                            self.compile_expr(index)?;
                            self.compile_expr(expr)?;
                            self.chunk.write(Instruction::SetIndex, line_of_expr(index));
                        }
                    }
                }
            }
            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
                self.chunk.write(Instruction::Pop, line_of_expr(expr));
            }
            Stmt::If {
                cond,
                then_block,
                else_branch,
            } => {
                self.compile_expr(cond)?;
                let jump_if_false_pos = self.chunk.code.len();
                self.chunk
                    .write(Instruction::JumpIfFalse(0), line_of_expr(cond));
                self.push_scope();
                self.compile_block(then_block)?;
                self.pop_scope();
                let jump_over_else_pos = self.chunk.code.len();
                self.chunk
                    .write(Instruction::Jump(0), line_for_block(then_block));
                // patch jump_if_false to start of else
                let else_start = self.chunk.code.len();
                if let Some(else_branch) = else_branch {
                    self.chunk.code[jump_if_false_pos] = Instruction::JumpIfFalse(else_start);
                    match else_branch {
                        ElseBranch::Block(b) => {
                            self.push_scope();
                            self.compile_block(b)?;
                            self.pop_scope();
                        }
                        ElseBranch::If(stmt) => self.compile_stmt(stmt)?,
                    }
                } else {
                    self.chunk.code[jump_if_false_pos] = Instruction::JumpIfFalse(else_start);
                }
                let after_else = self.chunk.code.len();
                self.chunk.code[jump_over_else_pos] = Instruction::Jump(after_else);
            }
            Stmt::While { cond, body } => {
                let loop_start = self.chunk.code.len();
                self.compile_expr(cond)?;
                let jump_if_false_pos = self.chunk.code.len();
                self.chunk
                    .write(Instruction::JumpIfFalse(0), line_of_expr(cond));
                self.push_scope();
                self.compile_block(body)?;
                self.pop_scope();
                self.chunk
                    .write(Instruction::Jump(loop_start), line_for_block(body));
                let after = self.chunk.code.len();
                self.chunk.code[jump_if_false_pos] = Instruction::JumpIfFalse(after);
            }
            Stmt::Return { expr } => {
                if let Some(expr) = expr {
                    self.compile_expr(expr)?;
                } else {
                    let idx = self.chunk.add_constant(Constant::Null);
                    self.chunk
                        .write(Instruction::Const(idx), line_of_expr_opt(expr));
                }
                self.chunk
                    .write(Instruction::Return, line_of_expr_opt(expr));
            }
            _ => return Err(CompileError::new("Unsupported statement")),
        }
        Ok(())
    }

    fn compile_block(&mut self, block: &Block) -> Result<(), CompileError> {
        for stmt in &block.stmts {
            self.compile_stmt(stmt)?;
        }
        Ok(())
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<(), CompileError> {
        match expr {
            Expr::Literal { lit, span } => {
                let c = match lit {
                    Literal::Int(v) => Constant::Int(*v),
                    Literal::Float(v) => Constant::Float(*v),
                    Literal::Bool(v) => Constant::Bool(*v),
                    Literal::String(s) => Constant::String(s.clone()),
                    Literal::None => Constant::Null,
                };
                let idx = self.chunk.add_constant(c);
                self.chunk.write(Instruction::Const(idx), span.line);
            }
            Expr::Ident { name, span } => match self.resolve_var(name)? {
                ResolvedVar::Local(i) => self.chunk.write(Instruction::LoadLocal(i), span.line),
                ResolvedVar::Global(i) => self.chunk.write(Instruction::LoadGlobal(i), span.line),
            },
            Expr::Unary { op, expr, span } => match op {
                UnaryOp::Negate => {
                    self.compile_expr(expr)?;
                    self.chunk.write(Instruction::Neg, span.line);
                }
                UnaryOp::Not => {
                    self.compile_expr(expr)?;
                    self.chunk.write(Instruction::Not, span.line);
                }
                UnaryOp::Await | UnaryOp::Spawn => {
                    let task_idx = self.enclosing.ensure_global("task");
                    self.chunk
                        .write(Instruction::LoadGlobal(task_idx), span.line);
                    let name = match op {
                        UnaryOp::Await => "join",
                        UnaryOp::Spawn => "spawn",
                        _ => unreachable!(),
                    };
                    let key_idx = self.chunk.add_constant(Constant::String(name.to_string()));
                    self.chunk.write(Instruction::GetField(key_idx), span.line);
                    self.compile_expr(expr)?;
                    self.chunk.write(Instruction::Call(1), span.line);
                }
            },
            Expr::Binary {
                left,
                op,
                right,
                span,
            } => {
                match op {
                    BinaryOp::And => {
                        self.compile_expr(left)?;
                        let jump_false = self.chunk.code.len();
                        self.chunk.write(Instruction::JumpIfFalse(0), span.line);
                        self.chunk.write(Instruction::Pop, span.line); // drop lhs when true path
                        self.compile_expr(right)?;
                        let end = self.chunk.code.len();
                        self.chunk.code[jump_false] = Instruction::JumpIfFalse(end);
                    }
                    BinaryOp::Or => {
                        self.compile_expr(left)?;
                        let jump_false = self.chunk.code.len();
                        self.chunk.write(Instruction::JumpIfFalse(0), span.line);
                        let jump_end = self.chunk.code.len();
                        self.chunk.write(Instruction::Jump(0), span.line);
                        let else_start = self.chunk.code.len();
                        self.chunk.write(Instruction::Pop, span.line); // drop lhs to eval rhs
                        self.compile_expr(right)?;
                        let end = self.chunk.code.len();
                        self.chunk.code[jump_false] = Instruction::JumpIfFalse(else_start);
                        self.chunk.code[jump_end] = Instruction::Jump(end);
                    }
                    _ => {
                        self.compile_expr(left)?;
                        self.compile_expr(right)?;
                        let instr = match op {
                            BinaryOp::Add => Instruction::Add,
                            BinaryOp::Subtract => Instruction::Sub,
                            BinaryOp::Multiply => Instruction::Mul,
                            BinaryOp::Divide => Instruction::Div,
                            BinaryOp::Modulo => Instruction::Mod,
                            BinaryOp::Equal => Instruction::Eq,
                            BinaryOp::NotEqual => Instruction::Neq,
                            BinaryOp::Less => Instruction::Lt,
                            BinaryOp::Greater => Instruction::Gt,
                            BinaryOp::LessEqual => Instruction::Le,
                            BinaryOp::GreaterEqual => Instruction::Ge,
                            _ => unreachable!(),
                        };
                        self.chunk.write(instr, span.line);
                    }
                }
            }
            Expr::Call { callee, args, span } => {
                self.compile_expr(callee)?;
                for arg in args {
                    match arg {
                        Arg::Positional(expr) => self.compile_expr(expr)?,
                        Arg::Named(_, _) => {
                            return Err(CompileError::new("Named arguments not supported yet"))
                        }
                    }
                }
                self.chunk
                    .write(Instruction::Call(args.len() as u16), span.line);
            }
            Expr::Index {
                target,
                index,
                span,
            } => {
                self.compile_expr(target)?;
                self.compile_expr(index)?;
                self.chunk.write(Instruction::GetIndex, span.line);
            }
            Expr::Field { target, name, span } => {
                if let Expr::Ident { name: alias, .. } = &**target {
                    if !self.is_local_name(alias) {
                        if let Some(exports) = self.module.import_exports.get(alias) {
                            if !exports.contains(name) {
                                let module_name = self
                                    .module
                                    .import_aliases
                                    .get(alias)
                                    .map(|id| id.0.join("::"))
                                    .unwrap_or_else(|| alias.clone());
                                return Err(CompileError::new(&format!(
                                    "Symbol '{}' is private to module {}",
                                    name, module_name
                                ))
                                .with_span(span.clone()));
                            }
                        }
                    }
                }
                self.compile_expr(target)?;
                let key_idx = self.chunk.add_constant(Constant::String(name.clone()));
                self.chunk.write(Instruction::GetField(key_idx), span.line);
            }
            Expr::List { items, span } => {
                for item in items {
                    self.compile_expr(item)?;
                }
                self.chunk
                    .write(Instruction::MakeList(items.len() as u16), span.line);
            }
            Expr::Try { expr, span } => {
                self.compile_expr(expr)?;
                self.chunk.write(Instruction::TryUnwrap, span.line);
            }
            _ => return Err(CompileError::new("Unsupported expression")),
        }
        Ok(())
    }

    fn ensure_method_table(&mut self, type_name: &str, line: usize) -> u16 {
        let table_name = type_method_table_name(type_name);
        let idx = self.enclosing.ensure_global(&table_name);
        if self.enclosing.method_tables.insert(table_name) {
            self.chunk.write(Instruction::MakeRecord(0), line);
            self.chunk.write(Instruction::StoreGlobal(idx), line);
        }
        idx
    }

    fn ensure_tool_namespace(
        &mut self,
        segments: &[String],
        line: usize,
    ) -> Result<(), CompileError> {
        if segments.is_empty() {
            return Ok(());
        }
        for i in 0..segments.len() {
            let key = format!("{}::{}", self.module.prefix, segments[..=i].join("."));
            if !self.enclosing.tool_namespaces.insert(key) {
                continue;
            }
            if i == 0 {
                let global = self
                    .enclosing
                    .ensure_global(&mangle_symbol(&self.module.prefix, &segments[0]));
                self.chunk.write(Instruction::MakeRecord(0), line);
                self.chunk.write(Instruction::StoreGlobal(global), line);
            } else {
                self.emit_load_path_record(&segments[..i], line)?;
                self.chunk.write(Instruction::MakeRecord(0), line);
                let key_idx = self
                    .chunk
                    .add_constant(Constant::String(segments[i].clone()));
                self.chunk.write(Instruction::SetField(key_idx), line);
            }
        }
        Ok(())
    }

    fn emit_load_path_record(
        &mut self,
        segments: &[String],
        line: usize,
    ) -> Result<(), CompileError> {
        if segments.is_empty() {
            return Err(CompileError::new("Empty path"));
        }
        let root = self
            .enclosing
            .ensure_global(&mangle_symbol(&self.module.prefix, &segments[0]));
        self.chunk.write(Instruction::LoadGlobal(root), line);
        for segment in &segments[1..] {
            let key_idx = self.chunk.add_constant(Constant::String(segment.clone()));
            self.chunk.write(Instruction::GetField(key_idx), line);
        }
        Ok(())
    }

    fn emit_imports(&mut self) -> Result<(), CompileError> {
        for (alias, module_id) in &self.module.import_aliases {
            let module_record = module_record_name(module_id);
            let record_idx = self.enclosing.ensure_global(&module_record);
            let alias_name = mangle_symbol(&self.module.prefix, alias);
            let alias_idx = self.enclosing.ensure_global(&alias_name);
            self.chunk.write(Instruction::LoadGlobal(record_idx), 0);
            self.chunk.write(Instruction::DefineGlobal(alias_idx), 0);
        }
        Ok(())
    }

    fn emit_exports(&mut self) -> Result<(), CompileError> {
        let count = self.module.exports.len() as u16;
        for name in &self.module.exports {
            let key_idx = self.chunk.add_constant(Constant::String(name.clone()));
            self.chunk.write(Instruction::Const(key_idx), 0);
            let global_name = mangle_symbol(&self.module.prefix, name);
            let global_idx = self.enclosing.ensure_global(&global_name);
            self.chunk.write(Instruction::LoadGlobal(global_idx), 0);
        }
        self.chunk.write(Instruction::MakeRecord(count), 0);
        self.chunk
            .write(Instruction::DefineGlobal(self.module.record_global), 0);
        Ok(())
    }

    fn push_scope(&mut self) {
        self.scopes.push(std::collections::HashMap::new());
        let current = *self.next_local.last().unwrap_or(&0);
        self.next_local.push(current);
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.next_local.pop();
    }

    fn define_local(&mut self, name: &str) -> u16 {
        let next = self.next_local.last_mut().unwrap();
        let slot = *next;
        *next += 1;
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name.to_string(), slot);
        slot
    }

    fn add_local(&mut self, name: &str) -> u16 {
        self.define_local(name)
    }

    fn is_local_name(&self, name: &str) -> bool {
        for scope in self.scopes.iter().rev() {
            if scope.contains_key(name) {
                return true;
            }
        }
        false
    }

    fn resolve_var(&self, name: &str) -> Result<ResolvedVar, CompileError> {
        for scope in self.scopes.iter().rev() {
            if let Some(idx) = scope.get(name) {
                return Ok(ResolvedVar::Local(*idx));
            }
        }
        let key = mangle_symbol(&self.module.prefix, name);
        if let Some(idx) = self.enclosing.globals.get(&key) {
            return Ok(ResolvedVar::Global(*idx));
        }
        if let Some(idx) = self.enclosing.globals.get(name) {
            return Ok(ResolvedVar::Global(*idx));
        }
        Err(CompileError::new("Unknown variable"))
    }

    fn finish(self) -> ByteFunction {
        ByteFunction {
            name: if self.name == "<main>" {
                None
            } else {
                Some(self.name)
            },
            arity: self.params.len() as u16,
            chunk: self.chunk,
            source_name: self.module.source_name.clone(),
        }
    }
}

fn line_of_expr(expr: &Expr) -> usize {
    match expr {
        Expr::Literal { span, .. }
        | Expr::Ident { span, .. }
        | Expr::Unary { span, .. }
        | Expr::Binary { span, .. }
        | Expr::Call { span, .. }
        | Expr::Index { span, .. }
        | Expr::Field { span, .. }
        | Expr::List { span, .. }
        | Expr::Lambda { span, .. }
        | Expr::Match { span, .. }
        | Expr::Try { span, .. } => span.line,
    }
}

fn line_of_expr_opt(expr: &Option<Expr>) -> usize {
    expr.as_ref().map(line_of_expr).unwrap_or(0)
}

fn line_for_block(block: &Block) -> usize {
    block
        .stmts
        .first()
        .map(|stmt| match stmt {
            Stmt::Let { name_span, .. } => name_span.line,
            Stmt::Assign { target, .. } => target.base_span.line,
            Stmt::Expr(expr) => line_of_expr(expr),
            Stmt::If { cond, .. } => line_of_expr(cond),
            Stmt::While { cond, .. } => line_of_expr(cond),
            Stmt::For { var_span, .. } => var_span.line,
            Stmt::Match { expr, .. } => line_of_expr(expr),
            Stmt::Try { body, .. } => line_for_block(body),
            Stmt::Return { expr } => line_of_expr_opt(expr),
            Stmt::Break | Stmt::Continue => 0,
        })
        .unwrap_or(0)
}

fn line_for_item(item: &Item) -> usize {
    match item {
        Item::Stmt(stmt) => match stmt {
            Stmt::Let { name_span, .. } => name_span.line,
            Stmt::Assign { target, .. } => target.base_span.line,
            Stmt::Expr(expr) => line_of_expr(expr),
            Stmt::If { cond, .. } => line_of_expr(cond),
            Stmt::While { cond, .. } => line_of_expr(cond),
            Stmt::For { var_span, .. } => var_span.line,
            Stmt::Match { expr, .. } => line_of_expr(expr),
            Stmt::Try { body, .. } => line_for_block(body),
            Stmt::Return { expr } => line_of_expr_opt(expr),
            Stmt::Break | Stmt::Continue => 0,
        },
        Item::Fn(decl) => decl.name_span.line,
        _ => 0,
    }
}

fn ffi_type_from_ref(ty: &TypeRef) -> Option<FfiType> {
    match ty {
        TypeRef::Named { path, optional, .. } => {
            if *optional {
                return match path.last().map(|s| s.as_str()) {
                    Some("String") => Some(FfiType::Optional(Box::new(FfiType::String))),
                    Some("Buffer") => Some(FfiType::Optional(Box::new(FfiType::Buffer))),
                    _ => None,
                };
            }
            match path.last().map(|s| s.as_str()) {
                Some("Int") => Some(FfiType::Int),
                Some("Float") => Some(FfiType::Float),
                Some("Bool") => Some(FfiType::Bool),
                Some("String") => Some(FfiType::String),
                Some("Buffer") => Some(FfiType::Buffer),
                Some("Void") => Some(FfiType::Void),
                _ => None,
            }
        }
        TypeRef::Function { .. } => None,
    }
}

fn policy_filter_values(filter: &PolicyFilter) -> Vec<String> {
    let mut out = Vec::new();
    match &filter.value {
        LiteralOrList::Literal(lit) => {
            if let Literal::String(value) = lit {
                out.push(value.clone());
            }
        }
        LiteralOrList::List(list) => {
            for lit in list {
                if let Literal::String(value) = lit {
                    out.push(value.clone());
                }
            }
        }
    }
    out
}

fn module_prefix(id: &ModuleId) -> String {
    id.0.join("::")
}

fn module_record_name(id: &ModuleId) -> String {
    format!("__module::{}", module_prefix(id))
}

fn mangle_symbol(module_prefix: &str, name: &str) -> String {
    format!("{}::{}", module_prefix, name)
}

fn type_method_table_name(type_name: &str) -> String {
    format!("__type_methods::{}", type_name)
}
