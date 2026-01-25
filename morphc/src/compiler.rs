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
            _symbols: SymbolTable::new(),
        }
    }

    fn define_builtin_globals(&mut self) {
        let _ = self.ensure_global("print");
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
                _ => {
                    return Err(CompileError::new(
                        "Only functions and statements are supported in this compiler pass",
                    ))
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
            Expr::Unary { op, expr, span } => {
                self.compile_expr(expr)?;
                match op {
                    UnaryOp::Negate => self.chunk.write(Instruction::Neg, span.line),
                    UnaryOp::Not => {
                        self.chunk.write(Instruction::Not, span.line);
                    }
                    _ => return Err(CompileError::new("Unsupported unary op")),
                }
            }
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
            _ => return Err(CompileError::new("Unsupported expression")),
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
            let base = match path.last().map(|s| s.as_str()) {
                Some("Int") => FfiType::Int,
                Some("Float") => FfiType::Float,
                Some("Bool") => FfiType::Bool,
                Some("String") => FfiType::String,
                Some("Buffer") => FfiType::Buffer,
                Some("Void") => FfiType::Void,
                _ => return None,
            };
            if *optional {
                Some(FfiType::Optional(Box::new(base)))
            } else {
                Some(base)
            }
        }
        TypeRef::Function { .. } => None,
    }
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
