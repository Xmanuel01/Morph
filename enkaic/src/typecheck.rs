use crate::ast::*;
use crate::diagnostic::{Diagnostic, Span};
use crate::modules::{ModuleId, Package};
use crate::types::Type;

#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
    pub source_name: Option<String>,
    pub source: Option<String>,
}

impl TypeError {
    pub fn diagnostic(&self) -> Option<Diagnostic> {
        let source = self.source.as_ref()?;
        let diagnostic = Diagnostic::new(&self.message, self.source_name.as_deref())
            .with_span("here", self.span.clone())
            .with_source(source);
        Some(diagnostic)
    }
}

pub struct TypeChecker {
    scopes: Vec<std::collections::HashMap<String, Type>>,
    imports: std::collections::HashMap<String, ModuleId>,
    exports: std::collections::HashMap<ModuleId, std::collections::HashMap<String, Type>>,
    source_name: Option<String>,
    source: Option<String>,
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self {
            scopes: vec![std::collections::HashMap::new()],
            imports: std::collections::HashMap::new(),
            exports: std::collections::HashMap::new(),
            source_name: None,
            source: None,
        }
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        Self::new_with_imports(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
        )
    }

    pub fn new_with_imports(
        imports: std::collections::HashMap<String, ModuleId>,
        exports: std::collections::HashMap<ModuleId, std::collections::HashMap<String, Type>>,
    ) -> Self {
        let (imports, exports) = inject_builtins(imports, exports);
        Self {
            scopes: vec![std::collections::HashMap::new()],
            imports,
            exports,
            source_name: None,
            source: None,
        }
    }

    fn with_source(mut self, source_name: &str, source: &str) -> Self {
        self.source_name = Some(source_name.to_string());
        self.source = Some(source.to_string());
        self
    }

    fn push(&mut self) {
        self.scopes.push(std::collections::HashMap::new());
    }
    fn pop(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, name: &str, ty: Type) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), ty);
    }

    fn resolve(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(t) = scope.get(name) {
                return Some(t.clone());
            }
        }
        None
    }

    fn error(&self, message: impl Into<String>, span: Span) -> TypeError {
        TypeError {
            message: message.into(),
            span,
            source_name: self.source_name.clone(),
            source: self.source.clone(),
        }
    }

    fn check_native_import(&self, decl: &NativeImportDecl) -> Result<(), TypeError> {
        for func in &decl.functions {
            for param in &func.params {
                let ty = type_from_ref(param.type_ann.clone());
                if !ffi_param_type_allowed(&ty) {
                    return Err(self.error(
                        format!("Invalid FFI parameter type {}", ty.display()),
                        param.name_span.clone(),
                    ));
                }
            }
            let ret = type_from_ref(func.return_type.clone());
            if !ffi_return_type_allowed(&ret) {
                return Err(self.error(
                    format!("Invalid FFI return type {}", ret.display()),
                    func.name_span.clone(),
                ));
            }
        }
        Ok(())
    }

    pub fn check_module(&mut self, module: &Module) -> Result<(), TypeError> {
        for item in &module.items {
            if let Item::Import(decl) = item {
                let alias = decl
                    .alias
                    .clone()
                    .unwrap_or_else(|| decl.path.last().cloned().unwrap_or_default());
                self.imports
                    .entry(alias)
                    .or_insert(ModuleId(decl.path.clone()));
            }
        }
        for alias in self.imports.keys().cloned().collect::<Vec<_>>() {
            self.define(&alias, Type::Unknown);
        }
        for item in &module.items {
            if let Item::Fn(decl) = item {
                let fn_type = Type::Function(
                    decl.params
                        .iter()
                        .map(|p| type_from_ref_opt(p.type_ann.clone()))
                        .collect(),
                    Box::new(type_from_ref_opt(decl.return_type.clone())),
                );
                self.define(&decl.name, fn_type);
            } else if let Item::NativeImport(decl) = item {
                for func in &decl.functions {
                    let params = func
                        .params
                        .iter()
                        .map(|p| type_from_ref(p.type_ann.clone()))
                        .collect();
                    let ret = type_from_ref(func.return_type.clone());
                    self.define(&func.name, Type::Function(params, Box::new(ret)));
                }
            } else if let Item::Type(decl) = item {
                self.define(&decl.name, Type::Unknown);
            } else if let Item::Enum(decl) = item {
                self.define(&decl.name, Type::Unknown);
            } else if let Item::Model(decl) = item {
                self.define(&decl.name, Type::Unknown);
            } else if let Item::Prompt(decl) = item {
                self.define(&decl.name, Type::Unknown);
            } else if let Item::Agent(decl) = item {
                self.define(&decl.name, Type::Unknown);
            } else if let Item::Tool(decl) = item {
                if let Some(root) = decl.path.first() {
                    self.define(root, Type::Unknown);
                }
            }
        }
        for item in &module.items {
            self.check_item(item)?;
        }
        Ok(())
    }

    pub fn check_package(package: &Package) -> Result<(), TypeError> {
        let mut exports = std::collections::HashMap::new();
        for (id, info) in &package.modules {
            exports.insert(id.clone(), collect_export_types(&info.module));
        }
        for id in &package.order {
            let info = package.modules.get(id).expect("module");
            let mut checker =
                TypeChecker::new_with_imports(info.import_aliases.clone(), exports.clone());
            if !info.file.as_os_str().is_empty() {
                checker = checker.with_source(&info.file.to_string_lossy(), &info.source);
            }
            checker.check_module(&info.module)?;
        }
        Ok(())
    }

    fn check_item(&mut self, item: &Item) -> Result<(), TypeError> {
        match item {
            Item::Fn(decl) => {
                self.push();
                for param in &decl.params {
                    self.define(&param.name, type_from_ref_opt(param.type_ann.clone()));
                }
                for stmt in &decl.body.stmts {
                    self.check_stmt(stmt, type_from_ref_opt(decl.return_type.clone()))?;
                }
                self.pop();
            }
            Item::NativeImport(decl) => {
                self.check_native_import(decl)?;
            }
            Item::Stmt(stmt) => {
                self.check_stmt(stmt, Type::Void)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn check_stmt(&mut self, stmt: &Stmt, expected_return: Type) -> Result<(), TypeError> {
        match stmt {
            Stmt::Let {
                name,
                type_ann,
                expr,
                name_span,
            } => {
                let t = self.check_expr(expr)?;
                if let Some(ann) = type_ann {
                    let ann_t = type_from_ref(ann.clone());
                    if !compatible(&ann_t, &t) {
                        return Err(self.error(
                            format!(
                                "Type mismatch: expected {}, found {}",
                                ann_t.display(),
                                t.display()
                            ),
                            name_span.clone(),
                        ));
                    }
                }
                self.define(name, t);
            }
            Stmt::Assign { target, expr } => {
                let t_expr = self.check_expr(expr)?;
                let var_t = self
                    .resolve(&target.base)
                    .ok_or_else(|| self.error("Undefined variable", target.base_span.clone()))?;
                if var_t != Type::Unknown && !compatible(&var_t, &t_expr) {
                    return Err(self.error(
                        format!(
                            "Type mismatch: variable {} is {}, assigned {}",
                            target.base,
                            var_t.display(),
                            t_expr.display()
                        ),
                        target.base_span.clone(),
                    ));
                }
            }
            Stmt::If {
                cond,
                then_block,
                else_branch,
            } => {
                let t_cond = self.check_expr(cond)?;
                if t_cond != Type::Bool && t_cond != Type::Unknown {
                    return Err(self.error("Condition must be Bool", line_span(cond)));
                }
                self.push();
                for s in &then_block.stmts {
                    self.check_stmt(s, expected_return.clone())?;
                }
                self.pop();
                if let Some(else_b) = else_branch {
                    self.push();
                    match else_b {
                        ElseBranch::Block(b) => {
                            for s in &b.stmts {
                                self.check_stmt(s, expected_return.clone())?;
                            }
                        }
                        ElseBranch::If(s) => self.check_stmt(s, expected_return.clone())?,
                    }
                    self.pop();
                }
            }
            Stmt::While { cond, body } => {
                let t_cond = self.check_expr(cond)?;
                if t_cond != Type::Bool && t_cond != Type::Unknown {
                    return Err(self.error("Condition must be Bool", line_span(cond)));
                }
                self.push();
                for s in &body.stmts {
                    self.check_stmt(s, expected_return.clone())?;
                }
                self.pop();
            }
            Stmt::Return { expr } => {
                let t_val = if let Some(e) = expr {
                    self.check_expr(e)?
                } else {
                    Type::Void
                };
                if expected_return != Type::Unknown && t_val != expected_return {
                    return Err(self.error(
                        format!(
                            "Return type mismatch: expected {}, found {}",
                            expected_return.display(),
                            t_val.display()
                        ),
                        expr.as_ref()
                            .map(line_span)
                            .unwrap_or_else(|| Span::single(0, 0)),
                    ));
                }
            }
            Stmt::Expr(expr) => {
                self.check_expr(expr)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn check_expr(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        match expr {
            Expr::Literal { lit, .. } => Ok(match lit {
                Literal::Int(_) => Type::Int,
                Literal::Float(_) => Type::Float,
                Literal::Bool(_) => Type::Bool,
                Literal::String(_) => Type::String,
                Literal::None => Type::Optional(Box::new(Type::Unknown)),
            }),
            Expr::Ident { name, span } => self
                .resolve(name)
                .ok_or_else(|| self.error(format!("Undefined variable {}", name), span.clone())),
            Expr::Unary { op, expr, span } => {
                let t = self.check_expr(expr)?;
                match op {
                    UnaryOp::Negate => match t {
                        Type::Int | Type::Float | Type::Unknown => Ok(t),
                        _ => Err(self.error("Unary '-' expects number", span.clone())),
                    },
                    UnaryOp::Not => match t {
                        Type::Bool | Type::Unknown => Ok(Type::Bool),
                        _ => Err(self.error("Unary 'not' expects Bool", span.clone())),
                    },
                    _ => Ok(Type::Unknown),
                }
            }
            Expr::Binary {
                left,
                op,
                right,
                span,
            } => {
                let lt = self.check_expr(left)?;
                let rt = self.check_expr(right)?;
                use BinaryOp::*;
                match op {
                    Add | Subtract | Multiply | Divide | Modulo => {
                        if lt == Type::Int && rt == Type::Int {
                            Ok(Type::Int)
                        } else if matches!(lt, Type::Float | Type::Int)
                            && matches!(rt, Type::Float | Type::Int)
                        {
                            Ok(Type::Float)
                        } else {
                            Err(self.error("Arithmetic expects numbers", span.clone()))
                        }
                    }
                    And | Or => {
                        if lt == Type::Bool && rt == Type::Bool {
                            Ok(Type::Bool)
                        } else {
                            Err(self.error("Logical ops expect Bool", span.clone()))
                        }
                    }
                    Equal | NotEqual | Less | LessEqual | Greater | GreaterEqual => Ok(Type::Bool),
                }
            }
            Expr::Call { callee, args, span } => {
                let ct = self.check_expr(callee)?;
                if let Type::Function(params, ret) = ct {
                    if params.len() != args.len() {
                        return Err(self.error(
                            format!(
                                "Arity mismatch: expected {}, got {}",
                                params.len(),
                                args.len()
                            ),
                            span.clone(),
                        ));
                    }
                    for (arg, pt) in args.iter().zip(params.iter()) {
                        let at = self.check_expr(match arg {
                            Arg::Positional(e) => e,
                            Arg::Named(_, e) => e,
                        })?;
                        if *pt != Type::Unknown && !compatible(pt, &at) {
                            return Err(self.error(
                                format!(
                                    "Argument type mismatch: expected {}, found {}",
                                    pt.display(),
                                    at.display()
                                ),
                                line_span(match arg {
                                    Arg::Positional(e) => e,
                                    Arg::Named(_, e) => e,
                                }),
                            ));
                        }
                    }
                    Ok(*ret)
                } else {
                    Err(self.error("Callee is not a function", span.clone()))
                }
            }
            Expr::Field { target, name, span } => {
                if let Expr::Ident { name: alias, .. } = &**target {
                    if let Some(module_id) = self.imports.get(alias) {
                        if let Some(map) = self.exports.get(module_id) {
                            if let Some(t) = map.get(name) {
                                return Ok(t.clone());
                            }
                        }
                        return Err(self.error(
                            format!(
                                "Symbol '{}' is private to module {}",
                                name,
                                module_id.0.join("::")
                            ),
                            span.clone(),
                        ));
                    }
                }
                self.check_expr(target)?;
                Ok(Type::Unknown)
            }
            Expr::Index { target, index, .. } => {
                self.check_expr(target)?;
                self.check_expr(index)?;
                Ok(Type::Unknown)
            }
            Expr::List { items, .. } => {
                for item in items {
                    self.check_expr(item)?;
                }
                Ok(Type::Unknown)
            }
            Expr::Try { expr, .. } => match self.check_expr(expr)? {
                Type::Optional(inner) => Ok(*inner),
                _ => Ok(Type::Unknown),
            },
            _ => Ok(Type::Unknown),
        }
    }
}

fn line_span(expr: &Expr) -> Span {
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
        | Expr::Try { span, .. } => span.clone(),
    }
}

fn type_from_ref_opt(r: Option<TypeRef>) -> Type {
    r.map(type_from_ref).unwrap_or(Type::Unknown)
}

fn type_from_ref(r: TypeRef) -> Type {
    match r {
        TypeRef::Named { path, optional, .. } => {
            let mut t = match path.first().map(|s| s.as_str()) {
                Some("Int") => Type::Int,
                Some("Float") => Type::Float,
                Some("Bool") => Type::Bool,
                Some("String") => Type::String,
                Some("Buffer") => Type::Buffer,
                Some("Tokenizer") => Type::Tokenizer,
                Some("DataStream") => Type::DataStream,
                Some("Batch") => Type::Batch,
                Some("Tensor") => Type::Tensor,
                Some("Device") => Type::Device,
                Some("DType") => Type::DType,
                Some("Shape") => Type::Shape,
                Some("OptimizerState") => Type::OptimizerState,
                Some("TcpListener") => Type::TcpListener,
                Some("TcpConnection") => Type::TcpConnection,
                Some("Request") => Type::HttpRequest,
                Some("Response") => Type::HttpResponse,
                Some("Void") => Type::Void,
                _ => Type::Unknown,
            };
            if optional {
                t = Type::Optional(Box::new(t));
            }
            t
        }
        TypeRef::Function { params, ret } => {
            let p = params.into_iter().map(type_from_ref).collect();
            let r = type_from_ref(*ret);
            Type::Function(p, Box::new(r))
        }
    }
}

fn compatible(expected: &Type, found: &Type) -> bool {
    if expected == found || *expected == Type::Unknown || *found == Type::Unknown {
        return true;
    }
    match (expected, found) {
        (Type::Optional(inner), Type::Optional(f)) => compatible(inner, f),
        (Type::Optional(inner), f) => compatible(inner, f), // allow T to satisfy Optional<T>
        (Type::Device, Type::String) => true,
        (Type::DType, Type::String) => true,
        (Type::Shape, Type::String) => true,
        (Type::String, Type::Device) => true,
        (Type::String, Type::DType) => true,
        (Type::String, Type::Shape) => true,
        (Type::Tensor, Type::Int) => true,
        (Type::Device, Type::Int) => true,
        (Type::OptimizerState, Type::Int) => true,
        (Type::Int, Type::Tensor) => true,
        (Type::Int, Type::Device) => true,
        (Type::Int, Type::OptimizerState) => true,
        _ => false,
    }
}

fn inject_builtins(
    mut imports: std::collections::HashMap<String, ModuleId>,
    mut exports: std::collections::HashMap<ModuleId, std::collections::HashMap<String, Type>>,
) -> (
    std::collections::HashMap<String, ModuleId>,
    std::collections::HashMap<ModuleId, std::collections::HashMap<String, Type>>,
) {
    let task_id = ModuleId(vec!["task".to_string()]);
    imports.entry("task".to_string()).or_insert(task_id.clone());
    let mut task_exports = std::collections::HashMap::new();
    task_exports.insert(
        "spawn".to_string(),
        Type::Function(
            vec![Type::Function(Vec::new(), Box::new(Type::Unknown))],
            Box::new(Type::Unknown),
        ),
    );
    task_exports.insert(
        "join".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::Unknown)),
    );
    task_exports.insert(
        "sleep".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::Void)),
    );
    exports.entry(task_id).or_insert(task_exports);
    let chan_id = ModuleId(vec!["chan".to_string()]);
    imports.entry("chan".to_string()).or_insert(chan_id.clone());
    let mut chan_exports = std::collections::HashMap::new();
    chan_exports.insert(
        "make".to_string(),
        Type::Function(Vec::new(), Box::new(Type::Channel)),
    );
    chan_exports.insert(
        "send".to_string(),
        Type::Function(vec![Type::Channel, Type::Unknown], Box::new(Type::Void)),
    );
    chan_exports.insert(
        "recv".to_string(),
        Type::Function(vec![Type::Channel], Box::new(Type::Unknown)),
    );
    exports.entry(chan_id).or_insert(chan_exports);
    let net_id = ModuleId(vec!["net".to_string()]);
    imports.entry("net".to_string()).or_insert(net_id.clone());
    let mut net_exports = std::collections::HashMap::new();
    net_exports.insert(
        "bind".to_string(),
        Type::Function(vec![Type::String, Type::Int], Box::new(Type::TcpListener)),
    );
    exports.entry(net_id).or_insert(net_exports);
    let http_id = ModuleId(vec!["http".to_string()]);
    imports.entry("http".to_string()).or_insert(http_id.clone());
    let mut http_exports = std::collections::HashMap::new();
    http_exports.insert(
        "serve".to_string(),
        Type::Function(
            vec![
                Type::String,
                Type::Int,
                Type::Function(vec![Type::HttpRequest], Box::new(Type::HttpResponse)),
            ],
            Box::new(Type::Void),
        ),
    );
    http_exports.insert(
        "get".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::HttpResponse)),
    );
    http_exports.insert(
        "post".to_string(),
        Type::Function(
            vec![Type::String, Type::Unknown],
            Box::new(Type::HttpResponse),
        ),
    );
    http_exports.insert(
        "response".to_string(),
        Type::Function(vec![Type::Int, Type::Unknown], Box::new(Type::HttpResponse)),
    );
    http_exports.insert(
        "ok".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::HttpResponse)),
    );
    http_exports.insert(
        "bad_request".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::HttpResponse)),
    );
    http_exports.insert(
        "not_found".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::HttpResponse)),
    );
    exports.entry(http_id).or_insert(http_exports);
    let json_id = ModuleId(vec!["json".to_string()]);
    imports.entry("json".to_string()).or_insert(json_id.clone());
    let mut json_exports = std::collections::HashMap::new();
    json_exports.insert(
        "parse".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    json_exports.insert(
        "stringify".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::String)),
    );
    exports.entry(json_id).or_insert(json_exports);
    let tokenizer_id = ModuleId(vec!["tokenizer".to_string()]);
    imports
        .entry("tokenizer".to_string())
        .or_insert(tokenizer_id.clone());
    let mut tokenizer_exports = std::collections::HashMap::new();
    tokenizer_exports.insert(
        "train".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::Tokenizer)),
    );
    tokenizer_exports.insert(
        "load".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Tokenizer)),
    );
    exports.entry(tokenizer_id).or_insert(tokenizer_exports);
    let dataset_id = ModuleId(vec!["dataset".to_string()]);
    imports
        .entry("dataset".to_string())
        .or_insert(dataset_id.clone());
    let mut dataset_exports = std::collections::HashMap::new();
    dataset_exports.insert(
        "open".to_string(),
        Type::Function(
            vec![Type::String, Type::Tokenizer, Type::Unknown],
            Box::new(Type::DataStream),
        ),
    );
    exports.entry(dataset_id).or_insert(dataset_exports);
    let checkpoint_id = ModuleId(vec!["checkpoint".to_string()]);
    imports
        .entry("checkpoint".to_string())
        .or_insert(checkpoint_id.clone());
    let mut checkpoint_exports = std::collections::HashMap::new();
    checkpoint_exports.insert(
        "save".to_string(),
        Type::Function(vec![Type::String, Type::Unknown], Box::new(Type::Void)),
    );
    checkpoint_exports.insert(
        "load".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    checkpoint_exports.insert(
        "latest".to_string(),
        Type::Function(
            vec![Type::String],
            Box::new(Type::Optional(Box::new(Type::String))),
        ),
    );
    checkpoint_exports.insert(
        "rotate".to_string(),
        Type::Function(vec![Type::String, Type::Int], Box::new(Type::Void)),
    );
    exports.entry(checkpoint_id).or_insert(checkpoint_exports);
    let std_nn_id = ModuleId(vec!["std".to_string(), "nn".to_string()]);
    let mut nn_exports = std::collections::HashMap::new();
    nn_exports.insert(
        "embedding".to_string(),
        Type::Function(vec![Type::Tensor, Type::Tensor], Box::new(Type::Tensor)),
    );
    nn_exports.insert(
        "linear".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Tensor, Type::Tensor],
            Box::new(Type::Tensor),
        ),
    );
    nn_exports.insert(
        "layernorm".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Tensor, Type::Tensor, Type::Float],
            Box::new(Type::Tensor),
        ),
    );
    nn_exports.insert(
        "gelu".to_string(),
        Type::Function(vec![Type::Tensor], Box::new(Type::Tensor)),
    );
    nn_exports.insert(
        "relu".to_string(),
        Type::Function(vec![Type::Tensor], Box::new(Type::Tensor)),
    );
    nn_exports.insert(
        "dropout".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Float, Type::Bool],
            Box::new(Type::Tensor),
        ),
    );
    nn_exports.insert(
        "embedding_params".to_string(),
        Type::Function(
            vec![Type::Int, Type::Int, Type::DType, Type::Device],
            Box::new(Type::Unknown),
        ),
    );
    nn_exports.insert(
        "linear_params".to_string(),
        Type::Function(
            vec![Type::Int, Type::Int, Type::DType, Type::Device],
            Box::new(Type::Unknown),
        ),
    );
    nn_exports.insert(
        "layernorm_params".to_string(),
        Type::Function(
            vec![Type::Int, Type::DType, Type::Device],
            Box::new(Type::Unknown),
        ),
    );
    exports.entry(std_nn_id).or_insert(nn_exports);
    let std_loss_id = ModuleId(vec!["std".to_string(), "loss".to_string()]);
    let mut loss_exports = std::collections::HashMap::new();
    loss_exports.insert(
        "cross_entropy".to_string(),
        Type::Function(vec![Type::Tensor, Type::Tensor], Box::new(Type::Tensor)),
    );
    exports.entry(std_loss_id).or_insert(loss_exports);
    let std_optim_id = ModuleId(vec!["std".to_string(), "optim".to_string()]);
    let mut optim_exports = std::collections::HashMap::new();
    optim_exports.insert(
        "adamw_create".to_string(),
        Type::Function(
            vec![
                Type::Unknown,
                Type::Float,
                Type::Float,
                Type::Float,
                Type::Float,
                Type::Float,
            ],
            Box::new(Type::OptimizerState),
        ),
    );
    optim_exports.insert(
        "adamw_step".to_string(),
        Type::Function(vec![Type::OptimizerState], Box::new(Type::Int)),
    );
    optim_exports.insert(
        "retain".to_string(),
        Type::Function(vec![Type::OptimizerState], Box::new(Type::Int)),
    );
    optim_exports.insert(
        "free".to_string(),
        Type::Function(vec![Type::OptimizerState], Box::new(Type::Int)),
    );
    exports.entry(std_optim_id).or_insert(optim_exports);
    let std_env_id = ModuleId(vec!["std".to_string(), "env".to_string()]);
    let mut env_exports = std::collections::HashMap::new();
    env_exports.insert(
        "get".to_string(),
        Type::Function(
            vec![Type::String],
            Box::new(Type::Optional(Box::new(Type::String))),
        ),
    );
    env_exports.insert(
        "set".to_string(),
        Type::Function(vec![Type::String, Type::String], Box::new(Type::Bool)),
    );
    env_exports.insert(
        "remove".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Bool)),
    );
    env_exports.insert(
        "cwd".to_string(),
        Type::Function(Vec::new(), Box::new(Type::Optional(Box::new(Type::String)))),
    );
    env_exports.insert(
        "set_cwd".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Bool)),
    );
    exports.entry(std_env_id).or_insert(env_exports);
    let std_path_id = ModuleId(vec!["std".to_string(), "path".to_string()]);
    let mut path_exports = std::collections::HashMap::new();
    path_exports.insert(
        "join".to_string(),
        Type::Function(vec![Type::String, Type::String], Box::new(Type::String)),
    );
    path_exports.insert(
        "dirname".to_string(),
        Type::Function(
            vec![Type::String],
            Box::new(Type::Optional(Box::new(Type::String))),
        ),
    );
    path_exports.insert(
        "basename".to_string(),
        Type::Function(
            vec![Type::String],
            Box::new(Type::Optional(Box::new(Type::String))),
        ),
    );
    path_exports.insert(
        "extname".to_string(),
        Type::Function(
            vec![Type::String],
            Box::new(Type::Optional(Box::new(Type::String))),
        ),
    );
    path_exports.insert(
        "normalize".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::String)),
    );
    exports.entry(std_path_id).or_insert(path_exports);
    let std_time_id = ModuleId(vec!["std".to_string(), "time".to_string()]);
    let mut time_exports = std::collections::HashMap::new();
    time_exports.insert(
        "now_ms".to_string(),
        Type::Function(Vec::new(), Box::new(Type::Int)),
    );
    time_exports.insert(
        "sleep_ms".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::Void)),
    );
    exports.entry(std_time_id).or_insert(time_exports);
    let std_log_id = ModuleId(vec!["std".to_string(), "log".to_string()]);
    let mut log_exports = std::collections::HashMap::new();
    log_exports.insert(
        "info".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Void)),
    );
    log_exports.insert(
        "warn".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Void)),
    );
    log_exports.insert(
        "error".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Void)),
    );
    exports.entry(std_log_id).or_insert(log_exports);
    let std_io_id = ModuleId(vec!["std".to_string(), "io".to_string()]);
    let mut io_exports = std::collections::HashMap::new();
    io_exports.insert(
        "read_bytes".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Buffer)),
    );
    io_exports.insert(
        "write_bytes".to_string(),
        Type::Function(vec![Type::String, Type::Buffer], Box::new(Type::Bool)),
    );
    io_exports.insert(
        "read_text".to_string(),
        Type::Function(
            vec![Type::String],
            Box::new(Type::Optional(Box::new(Type::String))),
        ),
    );
    io_exports.insert(
        "write_text".to_string(),
        Type::Function(vec![Type::String, Type::String], Box::new(Type::Bool)),
    );
    io_exports.insert(
        "stdin_read".to_string(),
        Type::Function(Vec::new(), Box::new(Type::Buffer)),
    );
    io_exports.insert(
        "stdout_write".to_string(),
        Type::Function(vec![Type::Buffer], Box::new(Type::Bool)),
    );
    io_exports.insert(
        "stderr_write".to_string(),
        Type::Function(vec![Type::Buffer], Box::new(Type::Bool)),
    );
    io_exports.insert(
        "stdout_write_text".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Bool)),
    );
    io_exports.insert(
        "stderr_write_text".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Bool)),
    );
    exports.entry(std_io_id).or_insert(io_exports);
    let std_process_id = ModuleId(vec!["std".to_string(), "process".to_string()]);
    let mut process_exports = std::collections::HashMap::new();
    process_exports.insert(
        "spawn".to_string(),
        Type::Function(
            vec![
                Type::String,
                Type::Unknown,
                Type::Optional(Box::new(Type::String)),
            ],
            Box::new(Type::Int),
        ),
    );
    process_exports.insert(
        "wait".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::Int)),
    );
    process_exports.insert(
        "kill".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::Bool)),
    );
    process_exports.insert(
        "run".to_string(),
        Type::Function(
            vec![
                Type::String,
                Type::Unknown,
                Type::Optional(Box::new(Type::String)),
            ],
            Box::new(Type::Unknown),
        ),
    );
    process_exports.insert(
        "exit".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::Void)),
    );
    exports.entry(std_process_id).or_insert(process_exports);
    (imports, exports)
}

fn ffi_param_type_allowed(ty: &Type) -> bool {
    match ty {
        Type::Int | Type::Float | Type::Bool | Type::String | Type::Buffer => true,
        Type::Optional(inner) => matches!(inner.as_ref(), Type::String | Type::Buffer),
        _ => false,
    }
}

fn ffi_return_type_allowed(ty: &Type) -> bool {
    match ty {
        Type::Void => true,
        _ => ffi_param_type_allowed(ty),
    }
}

fn collect_export_types(module: &Module) -> std::collections::HashMap<String, Type> {
    let mut out = std::collections::HashMap::new();
    for item in &module.items {
        if let Item::Fn(decl) = item {
            if decl.is_pub {
                let params = decl
                    .params
                    .iter()
                    .map(|p| type_from_ref_opt(p.type_ann.clone()))
                    .collect();
                let ret = type_from_ref_opt(decl.return_type.clone());
                out.insert(decl.name.clone(), Type::Function(params, Box::new(ret)));
            }
        } else if let Item::Type(decl) = item {
            if decl.is_pub {
                out.insert(decl.name.clone(), Type::Unknown);
            }
        } else if let Item::Enum(decl) = item {
            if decl.is_pub {
                out.insert(decl.name.clone(), Type::Unknown);
            }
        }
    }
    out
}
