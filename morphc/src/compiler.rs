use std::collections::HashMap;

use bumpalo::Bump;

use crate::arena::CompilerArena;
use crate::ast::*;
use crate::bytecode::{ByteFunction, Chunk, Constant, Instruction, Program};
use crate::diagnostic::Span;
use crate::symbols::SymbolTable;

#[derive(Debug)]
pub struct CompileError {
    pub message: String,
    pub span: Option<Span>,
}

impl CompileError {
    fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
            span: None,
        }
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
    let main_fn = ctx.compile_module_items(&module.items)?;
    Ok(Program {
        functions: ctx.functions,
        main: main_fn,
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
        // placeholder for future builtins; ensures stable slots
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

    fn compile_module_items(&mut self, items: &[Item]) -> Result<u16, CompileError> {
        // top-level scope
        let mut f = FunctionBuilder::new("<main>", &[], self, true);
        f.compile_items(items)?;
        let function = f.finish();
        let func_index = self.functions.len() as u16;
        self.functions.push(function);
        Ok(func_index)
    }

    fn compile_function(
        &mut self,
        name: &str,
        params: &[Param],
        body_items: &[Item],
    ) -> Result<u16, CompileError> {
        let mut f = FunctionBuilder::new(name, params, self, false);
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
}

impl<'a, 'p> FunctionBuilder<'a, 'p> {
    fn new(
        name: &str,
        params: &[Param],
        enclosing: &'p mut ProgramBuilder<'a>,
        is_root: bool,
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
        }
    }

    fn compile_items(&mut self, items: &[Item]) -> Result<(), CompileError> {
        // define params as locals
        let param_names: Vec<String> = self.params.iter().map(|p| p.name.clone()).collect();
        for name in param_names {
            self.define_local(&name);
        }
        for (i, item) in items.iter().enumerate() {
            let is_last = i + 1 == items.len();
            match item {
                Item::Stmt(Stmt::Expr(expr)) if is_last => {
                    self.compile_expr(expr)?;
                    self.chunk.write(Instruction::Return, line_of_expr(expr));
                    return Ok(());
                }
                Item::Stmt(stmt) => self.compile_stmt(stmt)?,
                Item::Fn(decl) => {
                    let global = self.enclosing.ensure_global(&decl.name);
                    // wrap statements as items for compiler
                    let body_items: Vec<Item> =
                        decl.body.stmts.iter().cloned().map(Item::Stmt).collect();
                    let func_idx =
                        self.enclosing
                            .compile_function(&decl.name, &decl.params, &body_items)?;
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
                    let g = self.enclosing.ensure_global(name);
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
            _ => return Err(CompileError::new("Unsupported expression")),
        }
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

    fn resolve_var(&self, name: &str) -> Result<ResolvedVar, CompileError> {
        for scope in self.scopes.iter().rev() {
            if let Some(idx) = scope.get(name) {
                return Ok(ResolvedVar::Local(*idx));
            }
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
