use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read};
use std::rc::Rc;

use morphc::ast::{
    AgentDecl, AgentItem, Arg, ArmBody, Block, Expr, Item, LValue, LValueAccess, Literal, MatchArm,
    MemoryDecl, Module, Pattern, Stmt,
};

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    None,
    List(Vec<Value>),
    Record(HashMap<String, Value>),
    Function(Rc<Function>),
    NativeFunction(Rc<NativeFunction>),
    Stub(String),
}

impl Value {
    fn display(&self) -> String {
        match self {
            Value::Int(value) => value.to_string(),
            Value::Float(value) => value.to_string(),
            Value::Bool(value) => value.to_string(),
            Value::String(value) => value.clone(),
            Value::None => "none".to_string(),
            Value::List(values) => {
                let inner = values
                    .iter()
                    .map(|v| v.display())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{}]", inner)
            }
            Value::Record(values) => {
                let inner = values
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.display()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{{}}}", inner)
            }
            Value::Function(func) => match &func.name {
                Some(name) => format!("<fn {}>", name),
                None => "<fn>".to_string(),
            },
            Value::NativeFunction(func) => format!("<native {}>", func.name),
            Value::Stub(name) => format!("<stub {}>", name),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::None, Value::None) => true,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Record(a), Value::Record(b)) => a == b,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct RuntimeError {
    pub message: String,
}

impl RuntimeError {
    fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RuntimeError {}

#[derive(Debug, Clone)]
pub struct Function {
    name: Option<String>,
    params: Vec<morphc::ast::Param>,
    body: Block,
    env: EnvRef,
}

pub struct NativeFunction {
    name: String,
    func: Box<dyn Fn(Vec<Value>) -> Result<Value, RuntimeError>>,
}

impl fmt::Debug for NativeFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeFunction")
            .field("name", &self.name)
            .finish()
    }
}

type EnvRef = Rc<RefCell<Env>>;

#[derive(Debug)]
struct Env {
    parent: Option<EnvRef>,
    values: HashMap<String, Value>,
}

impl Env {
    fn new(parent: Option<EnvRef>) -> EnvRef {
        Rc::new(RefCell::new(Self {
            parent,
            values: HashMap::new(),
        }))
    }

    fn define(env: &EnvRef, name: &str, value: Value) {
        env.borrow_mut().values.insert(name.to_string(), value);
    }

    fn get(env: &EnvRef, name: &str) -> Option<Value> {
        let mut current = Some(env.clone());
        while let Some(scope) = current {
            if let Some(value) = scope.borrow().values.get(name) {
                return Some(value.clone());
            }
            current = scope.borrow().parent.clone();
        }
        None
    }

    fn resolve_env(env: &EnvRef, name: &str) -> Option<EnvRef> {
        let mut current = Some(env.clone());
        while let Some(scope) = current {
            if scope.borrow().values.contains_key(name) {
                return Some(scope);
            }
            current = scope.borrow().parent.clone();
        }
        None
    }
}

#[derive(Clone, Copy)]
struct EvalContext {
    in_function: bool,
    in_loop: bool,
}

enum Flow {
    Value(Value),
    Return(Value),
    Break,
    Continue,
}

pub struct Interpreter {
    globals: EnvRef,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    pub fn new() -> Self {
        let globals = Env::new(None);
        let mut interpreter = Self { globals };
        interpreter.install_builtins();
        interpreter
    }

    pub fn eval_module(&mut self, module: &Module) -> Result<Value, RuntimeError> {
        let ctx = EvalContext {
            in_function: false,
            in_loop: false,
        };
        for item in &module.items {
            match item {
                Item::Use(decl) => {
                    self.eval_use_decl(&decl.path, decl.alias.as_deref())?;
                }
                Item::Fn(decl) => {
                    let func = Function {
                        name: Some(decl.name.clone()),
                        params: decl.params.clone(),
                        body: decl.body.clone(),
                        env: self.globals.clone(),
                    };
                    Env::define(&self.globals, &decl.name, Value::Function(Rc::new(func)));
                }
                Item::Type(_) | Item::Enum(_) | Item::Impl(_) | Item::Policy(_) => {
                    // Declarations without runtime behavior in v0.1.
                }
                Item::Tool(decl) => {
                    self.define_tool_stub(&decl.path)?;
                }
                Item::Prompt(decl) => {
                    Env::define(
                        &self.globals,
                        &decl.name,
                        Value::Stub(format!("prompt {}", decl.name)),
                    );
                }
                Item::Model(decl) => {
                    Env::define(
                        &self.globals,
                        &decl.name,
                        Value::Stub(format!("model {}", decl.name)),
                    );
                }
                Item::Agent(decl) => {
                    self.eval_agent_decl(decl)?;
                }
                Item::Stmt(stmt) => {
                    let flow = self.eval_stmt(stmt, self.globals.clone(), ctx)?;
                    if let Flow::Return(_) = flow {
                        return Err(RuntimeError::new("Return outside of function"));
                    }
                }
            }
        }
        Ok(Value::None)
    }

    pub fn call_main(&mut self) -> Result<Option<Value>, RuntimeError> {
        let main_value = Env::get(&self.globals, "main");
        match main_value {
            None => Ok(None),
            Some(value) => {
                let result = self.eval_call(value, Vec::new())?;
                Ok(Some(result))
            }
        }
    }

    pub fn get_global(&self, name: &str) -> Option<Value> {
        Env::get(&self.globals, name)
    }

    fn install_builtins(&mut self) {
        let print_fn = Value::NativeFunction(Rc::new(NativeFunction {
            name: "print".to_string(),
            func: Box::new(|args| {
                let output = args
                    .iter()
                    .map(|v| v.display())
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("{}", output);
                Ok(Value::None)
            }),
        }));

        let readline_fn = Value::NativeFunction(Rc::new(NativeFunction {
            name: "readline".to_string(),
            func: Box::new(|_| {
                let mut input = String::new();
                io::stdin()
                    .read_to_string(&mut input)
                    .map_err(|err| RuntimeError::new(&format!("Failed to read stdin: {}", err)))?;
                Ok(Value::String(input))
            }),
        }));

        Env::define(&self.globals, "print", print_fn.clone());
        Env::define(&self.globals, "readline", readline_fn.clone());

        let mut std_core = HashMap::new();
        std_core.insert("print".to_string(), print_fn.clone());
        let mut std_io = HashMap::new();
        std_io.insert("print".to_string(), print_fn);
        std_io.insert("readline".to_string(), readline_fn);
        let mut std_collections = HashMap::new();
        std_collections.insert(
            "len".to_string(),
            Value::NativeFunction(Rc::new(NativeFunction {
                name: "len".to_string(),
                func: Box::new(|args| {
                    if args.len() != 1 {
                        return Err(RuntimeError::new("len expects one argument"));
                    }
                    match &args[0] {
                        Value::String(value) => Ok(Value::Int(value.len() as i64)),
                        Value::List(values) => Ok(Value::Int(values.len() as i64)),
                        _ => Err(RuntimeError::new("len expects a string or list")),
                    }
                }),
            })),
        );
        let mut std = HashMap::new();
        std.insert("core".to_string(), Value::Record(std_core));
        std.insert("io".to_string(), Value::Record(std_io));
        std.insert("collections".to_string(), Value::Record(std_collections));
        Env::define(&self.globals, "std", Value::Record(std));
    }

    fn eval_use_decl(&self, path: &[String], alias: Option<&str>) -> Result<(), RuntimeError> {
        if path.is_empty() {
            return Err(RuntimeError::new("Empty module path"));
        }
        let mut current = Env::get(&self.globals, &path[0])
            .ok_or_else(|| RuntimeError::new("Module not found"))?;
        for segment in &path[1..] {
            current = match current {
                Value::Record(map) => map
                    .get(segment)
                    .cloned()
                    .ok_or_else(|| RuntimeError::new("Module not found"))?,
                _ => return Err(RuntimeError::new("Module not found")),
            };
        }
        let name = alias.unwrap_or_else(|| path.last().unwrap().as_str());
        Env::define(&self.globals, name, current);
        Ok(())
    }

    fn define_tool_stub(&self, path: &[String]) -> Result<(), RuntimeError> {
        if path.is_empty() {
            return Err(RuntimeError::new("Tool path cannot be empty"));
        }
        let stub_name = path.join(".");
        let stub_value = Value::NativeFunction(Rc::new(NativeFunction {
            name: stub_name.clone(),
            func: Box::new(move |_| {
                Err(RuntimeError::new(&format!(
                    "Tool not implemented: {}",
                    stub_name
                )))
            }),
        }));
        define_path(&self.globals, path, stub_value)?;
        Ok(())
    }

    fn eval_agent_decl(&mut self, decl: &AgentDecl) -> Result<(), RuntimeError> {
        let mut agent_record = HashMap::new();
        for item in &decl.items {
            match item {
                AgentItem::PolicyUse(name) => {
                    agent_record.insert(
                        "policy".to_string(),
                        Value::Stub(format!("policy {}", name)),
                    );
                }
                AgentItem::Memory(mem) => match mem {
                    MemoryDecl::Path { name, .. } => {
                        agent_record.insert(name.clone(), Value::Stub("memory".to_string()));
                    }
                    MemoryDecl::Expr { name, .. } => {
                        agent_record.insert(name.clone(), Value::Stub("memory".to_string()));
                    }
                },
                AgentItem::Fn(func) => {
                    let function = Function {
                        name: Some(func.name.clone()),
                        params: func.params.clone(),
                        body: func.body.clone(),
                        env: self.globals.clone(),
                    };
                    agent_record.insert(func.name.clone(), Value::Function(Rc::new(function)));
                }
                AgentItem::Stmt(_) => {
                    // Agent bodies are not executed in v0.1.
                }
            }
        }
        Env::define(&self.globals, &decl.name, Value::Record(agent_record));
        Ok(())
    }

    fn eval_stmt(
        &mut self,
        stmt: &Stmt,
        env: EnvRef,
        ctx: EvalContext,
    ) -> Result<Flow, RuntimeError> {
        match stmt {
            Stmt::Let { name, expr, .. } => {
                let value = self.eval_expr(expr, env.clone(), ctx)?;
                Env::define(&env, name, value);
                Ok(Flow::Value(Value::None))
            }
            Stmt::Assign { target, expr } => {
                let value = self.eval_expr(expr, env.clone(), ctx)?;
                self.assign_lvalue(env, target, value, ctx)?;
                Ok(Flow::Value(Value::None))
            }
            Stmt::Expr(expr) => {
                let value = self.eval_expr(expr, env, ctx)?;
                Ok(Flow::Value(value))
            }
            Stmt::If {
                cond,
                then_block,
                else_branch,
            } => {
                if truthy(&self.eval_expr(cond, env.clone(), ctx)?) {
                    let flow = self.eval_block(then_block, env, ctx)?;
                    return Ok(flow);
                }
                if let Some(branch) = else_branch {
                    match branch {
                        morphc::ast::ElseBranch::Block(block) => {
                            let flow = self.eval_block(block, env, ctx)?;
                            return Ok(flow);
                        }
                        morphc::ast::ElseBranch::If(stmt) => {
                            return self.eval_stmt(stmt, env, ctx);
                        }
                    }
                }
                Ok(Flow::Value(Value::None))
            }
            Stmt::While { cond, body } => {
                loop {
                    if !truthy(&self.eval_expr(cond, env.clone(), ctx)?) {
                        break;
                    }
                    let loop_ctx = EvalContext {
                        in_function: ctx.in_function,
                        in_loop: true,
                    };
                    match self.eval_block(body, env.clone(), loop_ctx)? {
                        Flow::Value(_) => {}
                        Flow::Break => break,
                        Flow::Continue => continue,
                        Flow::Return(value) => return Ok(Flow::Return(value)),
                    }
                }
                Ok(Flow::Value(Value::None))
            }
            Stmt::For { var, iter, body } => {
                let values = self.eval_expr(iter, env.clone(), ctx)?;
                let list = match values {
                    Value::List(items) => items,
                    _ => return Err(RuntimeError::new("for-in expects a list")),
                };
                let loop_env = Env::new(Some(env.clone()));
                let loop_ctx = EvalContext {
                    in_function: ctx.in_function,
                    in_loop: true,
                };
                for value in list {
                    Env::define(&loop_env, var, value);
                    match self.eval_block(body, loop_env.clone(), loop_ctx)? {
                        Flow::Value(_) => {}
                        Flow::Break => break,
                        Flow::Continue => continue,
                        Flow::Return(value) => return Ok(Flow::Return(value)),
                    }
                }
                Ok(Flow::Value(Value::None))
            }
            Stmt::Match { expr, arms } => {
                let value = self.eval_expr(expr, env.clone(), ctx)?;
                for arm in arms {
                    if let Some(bind) = match_pattern(&arm.pattern, &value) {
                        let arm_env = Env::new(Some(env.clone()));
                        if let Some((name, bound)) = bind {
                            Env::define(&arm_env, &name, bound);
                        }
                        let flow = match &arm.body {
                            ArmBody::Expr(expr) => Flow::Value(self.eval_expr(expr, arm_env, ctx)?),
                            ArmBody::Block(block) => self.eval_block(block, arm_env, ctx)?,
                        };
                        return Ok(flow);
                    }
                }
                Err(RuntimeError::new("No match arm satisfied"))
            }
            Stmt::Try {
                body,
                catch_name,
                catch_body,
            } => match self.eval_block(body, env.clone(), ctx) {
                Ok(flow) => Ok(flow),
                Err(err) => {
                    let catch_env = Env::new(Some(env));
                    Env::define(&catch_env, catch_name, Value::String(err.message));
                    self.eval_block(catch_body, catch_env, ctx)
                }
            },
            Stmt::Return { expr } => {
                if !ctx.in_function {
                    return Err(RuntimeError::new("Return outside of function"));
                }
                let value = if let Some(expr) = expr {
                    self.eval_expr(expr, env, ctx)?
                } else {
                    Value::None
                };
                Ok(Flow::Return(value))
            }
            Stmt::Break => {
                if !ctx.in_loop {
                    return Err(RuntimeError::new("Break outside of loop"));
                }
                Ok(Flow::Break)
            }
            Stmt::Continue => {
                if !ctx.in_loop {
                    return Err(RuntimeError::new("Continue outside of loop"));
                }
                Ok(Flow::Continue)
            }
        }
    }

    fn eval_block(
        &mut self,
        block: &Block,
        env: EnvRef,
        ctx: EvalContext,
    ) -> Result<Flow, RuntimeError> {
        let local = Env::new(Some(env));
        let mut last = Value::None;
        for stmt in &block.stmts {
            match self.eval_stmt(stmt, local.clone(), ctx)? {
                Flow::Value(value) => last = value,
                flow @ Flow::Return(_) => return Ok(flow),
                Flow::Break => return Ok(Flow::Break),
                Flow::Continue => return Ok(Flow::Continue),
            }
        }
        Ok(Flow::Value(last))
    }

    fn eval_expr(
        &mut self,
        expr: &Expr,
        env: EnvRef,
        ctx: EvalContext,
    ) -> Result<Value, RuntimeError> {
        match expr {
            Expr::Literal(lit) => Ok(eval_literal(lit)),
            Expr::Ident(name) => Env::get(&env, name)
                .ok_or_else(|| RuntimeError::new(&format!("Undefined variable '{}'", name))),
            Expr::Binary { left, op, right } => {
                use morphc::ast::BinaryOp;
                if matches!(op, BinaryOp::And) {
                    let left_val = self.eval_expr(left, env.clone(), ctx)?;
                    if !truthy(&left_val) {
                        return Ok(Value::Bool(false));
                    }
                    let right_val = self.eval_expr(right, env, ctx)?;
                    return Ok(Value::Bool(truthy(&right_val)));
                }
                if matches!(op, BinaryOp::Or) {
                    let left_val = self.eval_expr(left, env.clone(), ctx)?;
                    if truthy(&left_val) {
                        return Ok(Value::Bool(true));
                    }
                    let right_val = self.eval_expr(right, env, ctx)?;
                    return Ok(Value::Bool(truthy(&right_val)));
                }
                let left_val = self.eval_expr(left, env.clone(), ctx)?;
                let right_val = self.eval_expr(right, env, ctx)?;
                eval_binary(op, left_val, right_val)
            }
            Expr::Unary { op, expr } => {
                let value = self.eval_expr(expr, env, ctx)?;
                eval_unary(op, value)
            }
            Expr::Call { callee, args } => {
                let callee_value = self.eval_expr(callee, env.clone(), ctx)?;
                let values = self.eval_args(args, env, ctx)?;
                self.eval_call(callee_value, values)
            }
            Expr::Index { target, index } => {
                let target_value = self.eval_expr(target, env.clone(), ctx)?;
                let index_value = self.eval_expr(index, env, ctx)?;
                eval_index(target_value, index_value)
            }
            Expr::Field { target, name } => {
                let target_value = self.eval_expr(target, env, ctx)?;
                eval_field(target_value, name)
            }
            Expr::List(values) => {
                let mut items = Vec::new();
                for value in values {
                    items.push(self.eval_expr(value, env.clone(), ctx)?);
                }
                Ok(Value::List(items))
            }
            Expr::Lambda { params, body, .. } => {
                let function = Function {
                    name: None,
                    params: params.clone(),
                    body: Block {
                        stmts: vec![Stmt::Expr(*body.clone())],
                    },
                    env,
                };
                Ok(Value::Function(Rc::new(function)))
            }
            Expr::Match { expr, arms } => self.eval_match_expr(expr, arms, env, ctx),
            Expr::Try(inner) => self.eval_expr(inner, env, ctx),
        }
    }

    fn eval_args(
        &mut self,
        args: &[Arg],
        env: EnvRef,
        ctx: EvalContext,
    ) -> Result<Vec<ArgValue>, RuntimeError> {
        let mut values = Vec::new();
        for arg in args {
            match arg {
                Arg::Positional(expr) => {
                    values.push(ArgValue::Positional(self.eval_expr(
                        expr,
                        env.clone(),
                        ctx,
                    )?));
                }
                Arg::Named(name, expr) => {
                    values.push(ArgValue::Named(
                        name.clone(),
                        self.eval_expr(expr, env.clone(), ctx)?,
                    ));
                }
            }
        }
        Ok(values)
    }

    fn eval_call(&mut self, callee: Value, args: Vec<ArgValue>) -> Result<Value, RuntimeError> {
        match callee {
            Value::Function(func) => self.call_user_function(&func, args),
            Value::NativeFunction(func) => {
                if args.iter().any(|arg| matches!(arg, ArgValue::Named(_, _))) {
                    return Err(RuntimeError::new(
                        "Named args are not supported for native functions",
                    ));
                }
                let mut values = Vec::new();
                for arg in args {
                    if let ArgValue::Positional(value) = arg {
                        values.push(value);
                    }
                }
                (func.func)(values)
            }
            Value::Stub(name) => Err(RuntimeError::new(&format!("{} is not implemented", name))),
            _ => Err(RuntimeError::new("Value is not callable")),
        }
    }

    fn call_user_function(
        &mut self,
        func: &Function,
        args: Vec<ArgValue>,
    ) -> Result<Value, RuntimeError> {
        let mut positional = Vec::new();
        let mut named = HashMap::new();
        for arg in args {
            match arg {
                ArgValue::Positional(value) => positional.push(value),
                ArgValue::Named(name, value) => {
                    if named.contains_key(&name) {
                        return Err(RuntimeError::new("Duplicate named argument"));
                    }
                    named.insert(name, value);
                }
            }
        }

        if positional.len() > func.params.len() {
            return Err(RuntimeError::new("Too many arguments"));
        }

        let call_env = Env::new(Some(func.env.clone()));
        for (idx, param) in func.params.iter().enumerate() {
            let value = if idx < positional.len() {
                positional[idx].clone()
            } else if let Some(value) = named.remove(&param.name) {
                value
            } else if let Some(default) = &param.default {
                self.eval_expr(
                    default,
                    call_env.clone(),
                    EvalContext {
                        in_function: false,
                        in_loop: false,
                    },
                )?
            } else {
                return Err(RuntimeError::new("Missing argument"));
            };
            Env::define(&call_env, &param.name, value);
        }

        if !named.is_empty() {
            return Err(RuntimeError::new("Unknown named argument"));
        }

        let ctx = EvalContext {
            in_function: true,
            in_loop: false,
        };
        match self.eval_block(&func.body, call_env, ctx)? {
            Flow::Return(value) => Ok(value),
            Flow::Value(value) => Ok(value),
            Flow::Break | Flow::Continue => {
                Err(RuntimeError::new("Invalid control flow in function"))
            }
        }
    }

    fn eval_match_expr(
        &mut self,
        expr: &Expr,
        arms: &[MatchArm],
        env: EnvRef,
        ctx: EvalContext,
    ) -> Result<Value, RuntimeError> {
        let value = self.eval_expr(expr, env.clone(), ctx)?;
        for arm in arms {
            if let Some(bind) = match_pattern(&arm.pattern, &value) {
                let arm_env = Env::new(Some(env.clone()));
                if let Some((name, bound)) = bind {
                    Env::define(&arm_env, &name, bound);
                }
                return match &arm.body {
                    ArmBody::Expr(expr) => self.eval_expr(expr, arm_env, ctx),
                    ArmBody::Block(_) => Err(RuntimeError::new(
                        "Block arms are not allowed in match expressions",
                    )),
                };
            }
        }
        Err(RuntimeError::new("No match arm satisfied"))
    }

    fn assign_lvalue(
        &mut self,
        env: EnvRef,
        target: &LValue,
        value: Value,
        ctx: EvalContext,
    ) -> Result<(), RuntimeError> {
        let accesses = self.eval_lvalue_accesses(&target.accesses, env.clone(), ctx)?;
        let scope = Env::resolve_env(&env, &target.base)
            .ok_or_else(|| RuntimeError::new("Undefined variable"))?;
        let mut scope_ref = scope.borrow_mut();
        let entry = scope_ref
            .values
            .get_mut(&target.base)
            .ok_or_else(|| RuntimeError::new("Undefined variable"))?;
        assign_into_value(entry, &accesses, value)
    }

    fn eval_lvalue_accesses(
        &mut self,
        accesses: &[LValueAccess],
        env: EnvRef,
        ctx: EvalContext,
    ) -> Result<Vec<AccessValue>, RuntimeError> {
        let mut resolved = Vec::new();
        for access in accesses {
            match access {
                LValueAccess::Field(name) => resolved.push(AccessValue::Field(name.clone())),
                LValueAccess::Index(expr) => {
                    let value = self.eval_expr(expr, env.clone(), ctx)?;
                    match value {
                        Value::Int(idx) => {
                            if idx < 0 {
                                return Err(RuntimeError::new("Index out of range"));
                            }
                            resolved.push(AccessValue::Index(idx as usize));
                        }
                        _ => {
                            return Err(RuntimeError::new("Index assignment expects integer index"))
                        }
                    }
                }
            }
        }
        Ok(resolved)
    }
}

enum ArgValue {
    Positional(Value),
    Named(String, Value),
}

enum AccessValue {
    Field(String),
    Index(usize),
}

fn eval_literal(lit: &Literal) -> Value {
    match lit {
        Literal::Int(value) => Value::Int(*value),
        Literal::Float(value) => Value::Float(*value),
        Literal::Bool(value) => Value::Bool(*value),
        Literal::String(value) => Value::String(value.clone()),
        Literal::None => Value::None,
    }
}

fn eval_binary(
    op: &morphc::ast::BinaryOp,
    left: Value,
    right: Value,
) -> Result<Value, RuntimeError> {
    use morphc::ast::BinaryOp;
    match op {
        BinaryOp::Add => add_values(left, right),
        BinaryOp::Subtract => number_op(left, right, |a, b| a - b, |a, b| a - b),
        BinaryOp::Multiply => number_op(left, right, |a, b| a * b, |a, b| a * b),
        BinaryOp::Divide => divide_values(left, right),
        BinaryOp::Modulo => number_op(left, right, |a, b| a % b, |a, b| a % b),
        BinaryOp::Equal => Ok(Value::Bool(left == right)),
        BinaryOp::NotEqual => Ok(Value::Bool(left != right)),
        BinaryOp::Less => compare_values(left, right, |a, b| a < b),
        BinaryOp::LessEqual => compare_values(left, right, |a, b| a <= b),
        BinaryOp::Greater => compare_values(left, right, |a, b| a > b),
        BinaryOp::GreaterEqual => compare_values(left, right, |a, b| a >= b),
        BinaryOp::And => Ok(Value::Bool(truthy(&left) && truthy(&right))),
        BinaryOp::Or => Ok(Value::Bool(truthy(&left) || truthy(&right))),
    }
}

fn eval_unary(op: &morphc::ast::UnaryOp, value: Value) -> Result<Value, RuntimeError> {
    use morphc::ast::UnaryOp;
    match op {
        UnaryOp::Negate => match value {
            Value::Int(value) => Ok(Value::Int(-value)),
            Value::Float(value) => Ok(Value::Float(-value)),
            _ => Err(RuntimeError::new("Unary '-' expects a number")),
        },
        UnaryOp::Not => Ok(Value::Bool(!truthy(&value))),
        UnaryOp::Await | UnaryOp::Spawn => {
            Err(RuntimeError::new("async/await is not implemented in v0.1"))
        }
    }
}

fn eval_index(target: Value, index: Value) -> Result<Value, RuntimeError> {
    match (target, index) {
        (Value::List(values), Value::Int(idx)) => {
            let idx = idx as isize;
            if idx < 0 || idx as usize >= values.len() {
                return Err(RuntimeError::new("Index out of range"));
            }
            Ok(values[idx as usize].clone())
        }
        _ => Err(RuntimeError::new(
            "Indexing expects a list and integer index",
        )),
    }
}

fn eval_field(target: Value, name: &str) -> Result<Value, RuntimeError> {
    match target {
        Value::Record(map) => map
            .get(name)
            .cloned()
            .ok_or_else(|| RuntimeError::new("Unknown field")),
        _ => Err(RuntimeError::new("Field access expects a record")),
    }
}

fn match_pattern(pattern: &Pattern, value: &Value) -> Option<Option<(String, Value)>> {
    match pattern {
        Pattern::Wildcard => Some(None),
        Pattern::Literal(lit) => {
            let lit_value = eval_literal(lit);
            if &lit_value == value {
                Some(None)
            } else {
                None
            }
        }
        Pattern::Ident(name) => Some(Some((name.clone(), value.clone()))),
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::None => false,
        Value::Int(value) => *value != 0,
        Value::Float(value) => *value != 0.0,
        Value::String(value) => !value.is_empty(),
        Value::List(values) => !values.is_empty(),
        Value::Record(map) => !map.is_empty(),
        _ => true,
    }
}

fn add_values(left: Value, right: Value) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
        (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
        _ => Err(RuntimeError::new("Unsupported operands for '+'")),
    }
}

fn divide_values(left: Value, right: Value) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Int(_), Value::Int(0)) => Err(RuntimeError::new("Division by zero")),
        (Value::Float(_), Value::Float(0.0)) => Err(RuntimeError::new("Division by zero")),
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 / b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / b as f64)),
        _ => Err(RuntimeError::new("Unsupported operands for '/'")),
    }
}

fn number_op(
    left: Value,
    right: Value,
    int_op: fn(i64, i64) -> i64,
    float_op: fn(f64, f64) -> f64,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(a, b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(a, b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(float_op(a as f64, b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(float_op(a, b as f64))),
        _ => Err(RuntimeError::new("Expected numeric operands")),
    }
}

fn compare_values(
    left: Value,
    right: Value,
    cmp: fn(f64, f64) -> bool,
) -> Result<Value, RuntimeError> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(cmp(a as f64, b as f64))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(cmp(a, b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Bool(cmp(a as f64, b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(cmp(a, b as f64))),
        _ => Err(RuntimeError::new("Expected numeric operands")),
    }
}

fn assign_into_value(
    target: &mut Value,
    accesses: &[AccessValue],
    value: Value,
) -> Result<(), RuntimeError> {
    if accesses.is_empty() {
        *target = value;
        return Ok(());
    }
    match &accesses[0] {
        AccessValue::Field(name) => match target {
            Value::Record(map) => {
                if accesses.len() == 1 {
                    map.insert(name.clone(), value);
                    Ok(())
                } else {
                    let next = map
                        .get_mut(name)
                        .ok_or_else(|| RuntimeError::new("Unknown field"))?;
                    assign_into_value(next, &accesses[1..], value)
                }
            }
            _ => Err(RuntimeError::new("Field assignment expects a record")),
        },
        AccessValue::Index(index) => match target {
            Value::List(items) => {
                if *index >= items.len() {
                    return Err(RuntimeError::new("Index out of range"));
                }
                if accesses.len() == 1 {
                    items[*index] = value;
                    Ok(())
                } else {
                    assign_into_value(&mut items[*index], &accesses[1..], value)
                }
            }
            _ => Err(RuntimeError::new("Index assignment expects list")),
        },
    }
}

fn define_path(env: &EnvRef, path: &[String], value: Value) -> Result<(), RuntimeError> {
    if path.is_empty() {
        return Err(RuntimeError::new("Empty path"));
    }
    if path.len() == 1 {
        Env::define(env, &path[0], value);
        return Ok(());
    }
    let head = &path[0];
    let current = Env::get(env, head).unwrap_or_else(|| Value::Record(HashMap::new()));
    let mut record = match current {
        Value::Record(map) => map,
        _ => return Err(RuntimeError::new("Path segment is not a record")),
    };
    let mut ptr = &mut record;
    for segment in &path[1..path.len() - 1] {
        let entry = ptr
            .entry(segment.clone())
            .or_insert_with(|| Value::Record(HashMap::new()));
        if let Value::Record(map) = entry {
            ptr = map;
        } else {
            return Err(RuntimeError::new("Path segment is not a record"));
        }
    }
    ptr.insert(path.last().unwrap().clone(), value);
    Env::define(env, head, Value::Record(record));
    Ok(())
}
