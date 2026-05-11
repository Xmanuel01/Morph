use crate::ast::*;
use crate::diagnostic::{Diagnostic, Span};
use crate::modules::{ModuleId, Package};
use crate::types::Type;

#[derive(Debug, Clone)]
struct BindingInfo {
    ty: Type,
    mutable: bool,
    constant: bool,
}

#[derive(Debug, Clone)]
struct StaticPolicy {
    rules: Vec<StaticPolicyRule>,
}

#[derive(Debug, Clone)]
struct StaticPolicyRule {
    allow: bool,
    capability: Vec<String>,
    filters: Vec<StaticPolicyFilter>,
}

#[derive(Debug, Clone)]
struct StaticPolicyFilter {
    name: String,
    values: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StaticCapabilityContext {
    Path(String),
    Domain(String),
}

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
    scopes: Vec<std::collections::HashMap<String, BindingInfo>>,
    imports: std::collections::HashMap<String, ModuleId>,
    exports: std::collections::HashMap<ModuleId, std::collections::HashMap<String, Type>>,
    record_fields: std::collections::HashMap<String, Vec<(String, Type)>>,
    policies: std::collections::HashMap<String, StaticPolicy>,
    source_name: Option<String>,
    source: Option<String>,
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self {
            scopes: vec![std::collections::HashMap::new()],
            imports: std::collections::HashMap::new(),
            exports: std::collections::HashMap::new(),
            record_fields: std::collections::HashMap::new(),
            policies: std::collections::HashMap::new(),
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
            record_fields: std::collections::HashMap::new(),
            policies: std::collections::HashMap::new(),
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
        self.define_binding(name, ty, false);
    }

    fn define_binding(&mut self, name: &str, ty: Type, mutable: bool) {
        self.define_binding_with_const(name, ty, mutable, false);
    }

    fn define_binding_with_const(&mut self, name: &str, ty: Type, mutable: bool, constant: bool) {
        self.scopes.last_mut().unwrap().insert(
            name.to_string(),
            BindingInfo {
                ty,
                mutable,
                constant,
            },
        );
    }

    fn resolve(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(binding.ty.clone());
            }
        }
        None
    }

    fn resolve_binding(&self, name: &str) -> Option<BindingInfo> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(binding.clone());
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

    fn register_type_decl(&mut self, decl: &TypeDecl) {
        let mut fields = Vec::with_capacity(decl.fields.len());
        for field in &decl.fields {
            fields.push((field.name.clone(), type_from_ref(field.field_type.clone())));
        }
        self.record_fields.insert(decl.name.clone(), fields);
    }

    fn record_field_type(&self, type_name: &str, field_name: &str) -> Option<Type> {
        self.record_fields.get(type_name).and_then(|fields| {
            fields
                .iter()
                .find(|(name, _)| name == field_name)
                .map(|(_, field_type)| field_type.clone())
        })
    }

    fn check_lvalue_target_type(&mut self, target: &LValue) -> Result<Type, TypeError> {
        let mut current = self
            .resolve(&target.base)
            .ok_or_else(|| self.error("Undefined variable", target.base_span.clone()))?;
        for access in &target.accesses {
            current = match access {
                LValueAccess::Field(name) => match current {
                    Type::Named(type_name) => {
                        if let Some(field_type) = self.record_field_type(&type_name, name) {
                            field_type
                        } else {
                            return Err(self.error(
                                format!("Unknown field {}.{}", type_name, name),
                                target.base_span.clone(),
                            ));
                        }
                    }
                    Type::Unknown => Type::Unknown,
                    _ => Type::Unknown,
                },
                LValueAccess::Index(index) => {
                    let index_type = self.check_expr(index)?;
                    if index_type != Type::Int && index_type != Type::Unknown {
                        return Err(self.error("Index expects Int", line_span(index)));
                    }
                    match current {
                        Type::Array(inner) | Type::Vector(inner) => *inner,
                        Type::List(inner) => *inner,
                        Type::Unknown => Type::Unknown,
                        _ => Type::Unknown,
                    }
                }
            };
        }
        Ok(current)
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

    fn register_policy_decl(&mut self, decl: &PolicyDecl) -> Result<(), TypeError> {
        if self.policies.contains_key(&decl.name) {
            return Err(self.error(
                format!("PolicyError: duplicate policy `{}`", decl.name),
                Span::single(1, 1),
            ));
        }
        let mut rules = Vec::with_capacity(decl.rules.len());
        for rule in &decl.rules {
            if rule.capability.is_empty() {
                return Err(self.error(
                    "PolicyError: policy rule capability cannot be empty",
                    Span::single(1, 1),
                ));
            }
            for segment in &rule.capability {
                if segment.trim().is_empty() {
                    return Err(self.error(
                        "PolicyError: policy capability contains an empty segment",
                        Span::single(1, 1),
                    ));
                }
            }
            let mut filters = Vec::with_capacity(rule.filters.len());
            for filter in &rule.filters {
                if !matches!(filter.name.as_str(), "path_prefix" | "domain") {
                    return Err(self.error(
                        format!(
                            "PolicyError: unsupported policy filter `{}`; expected `path_prefix` or `domain`",
                            filter.name
                        ),
                        Span::single(1, 1),
                    ));
                }
                let values = policy_filter_static_values(filter).map_err(|message| {
                    self.error(format!("PolicyError: {}", message), Span::single(1, 1))
                })?;
                if values.is_empty() {
                    return Err(self.error(
                        format!(
                            "PolicyError: policy filter `{}` cannot be empty",
                            filter.name
                        ),
                        Span::single(1, 1),
                    ));
                }
                filters.push(StaticPolicyFilter {
                    name: filter.name.clone(),
                    values,
                });
            }
            rules.push(StaticPolicyRule {
                allow: rule.allow,
                capability: rule.capability.clone(),
                filters,
            });
        }
        self.policies
            .insert(decl.name.clone(), StaticPolicy { rules });
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
            } else if let Item::Policy(decl) = item {
                self.register_policy_decl(decl)?;
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
                self.register_type_decl(decl);
                self.define(&decl.name, Type::Named(decl.name.clone()));
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
            let mut imports = info.import_aliases.clone();
            if is_std_json_module(id) {
                imports.insert("json".to_string(), ModuleId(vec!["json".to_string()]));
            }
            let mut checker = TypeChecker::new_with_imports(imports, exports.clone());
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
                    self.define_binding(
                        &param.name,
                        type_from_ref_opt(param.type_ann.clone()),
                        param.mutable,
                    );
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
                mutable,
                constant,
            } => {
                let t = if let Expr::List { .. } = expr {
                    let inferred = self.infer_array_literal(expr)?;
                    if type_ann.is_none() && matches!(inferred, ArrayLiteralType::Empty) {
                        return Err(self.error(
                            "TypeError: cannot infer type of empty array; add an explicit type such as Array[String].",
                            line_span(expr),
                        ));
                    }
                    if type_ann.is_none() && matches!(inferred, ArrayLiteralType::Mixed) {
                        return Err(self.error(
                            "TypeError: mixed-type arrays require an explicit dynamic type such as Array[Any].",
                            line_span(expr),
                        ));
                    }
                    inferred.as_type()
                } else {
                    self.check_expr(expr)?
                };
                if *constant && !self.is_compile_time_const_expr(expr) {
                    return Err(self.error(
                        format!(
                            "TypeError: const binding `{}` requires a compile-time constant expression.",
                            name
                        ),
                        line_span(expr),
                    ));
                }
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
                    self.define_binding_with_const(name, ann_t, *mutable, *constant);
                    return Ok(());
                }
                self.define_binding_with_const(name, t, *mutable, *constant);
            }
            Stmt::Assign { target, expr } => {
                let t_expr = self.check_expr(expr)?;
                if target.accesses.is_empty() {
                    let binding = self.resolve_binding(&target.base).ok_or_else(|| {
                        self.error(
                            format!("Undefined variable {}", target.base),
                            target.base_span.clone(),
                        )
                    })?;
                    if !binding.mutable {
                        return Err(self.error(
                            format!(
                                "TypeError: cannot assign to immutable variable `{}`; declare it with `mut` if mutation is required.",
                                target.base
                            ),
                            target.base_span.clone(),
                        ));
                    }
                }
                let target_t = self.check_lvalue_target_type(target)?;
                if target_t != Type::Unknown && !compatible(&target_t, &t_expr) {
                    return Err(self.error(
                        format!(
                            "Type mismatch: variable {} is {}, assigned {}",
                            target.base,
                            target_t.display(),
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
                if expected_return != Type::Unknown && !compatible(&expected_return, &t_val) {
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
                if let Expr::Ident {
                    name,
                    span: callee_span,
                } = &**callee
                {
                    if let Some(Type::Named(type_name)) = self.resolve(name) {
                        if let Some(fields) = self.record_fields.get(&type_name).cloned() {
                            if fields.len() != args.len() {
                                return Err(self.error(
                                    format!(
                                        "Constructor arity mismatch: expected {}, got {}",
                                        fields.len(),
                                        args.len()
                                    ),
                                    span.clone(),
                                ));
                            }
                            for (index, arg) in args.iter().enumerate() {
                                let (field_name, expected) = match arg {
                                    Arg::Positional(_) => fields
                                        .get(index)
                                        .map(|(field_name, expected)| {
                                            (field_name.clone(), expected.clone())
                                        })
                                        .ok_or_else(|| {
                                            self.error(
                                                "Constructor arity mismatch",
                                                line_span(match arg {
                                                    Arg::Positional(expr) => expr,
                                                    Arg::Named(_, expr) => expr,
                                                }),
                                            )
                                        })?,
                                    Arg::Named(name, _) => {
                                        let expected = self
                                            .record_field_type(&type_name, name)
                                            .ok_or_else(|| {
                                                self.error(
                                                    format!(
                                                        "Unknown constructor field {}.{}",
                                                        type_name, name
                                                    ),
                                                    callee_span.clone(),
                                                )
                                            })?;
                                        (name.clone(), expected)
                                    }
                                };
                                let actual = self.check_expr(match arg {
                                    Arg::Positional(expr) => expr,
                                    Arg::Named(_, expr) => expr,
                                })?;
                                if expected != Type::Unknown && !compatible(&expected, &actual) {
                                    return Err(self.error(
                                        format!(
                                            "Constructor field type mismatch for {}.{}: expected {}, found {}",
                                            type_name,
                                            field_name,
                                            expected.display(),
                                            actual.display()
                                        ),
                                        line_span(match arg {
                                            Arg::Positional(expr) => expr,
                                            Arg::Named(_, expr) => expr,
                                        }),
                                    ));
                                }
                            }
                            return Ok(Type::Named(type_name));
                        }
                    }
                }
                if let Expr::Field { target, name, .. } = &**callee {
                    if let Expr::Ident { name: alias, .. } = &**target {
                        if alias == "json" && !self.imports.contains_key(alias) {
                            return Err(self.error(
                                format!("ImportError: `json.{}` requires `import std::json`", name),
                                span.clone(),
                            ));
                        }
                        if let Some(module_id) = self.imports.get(alias) {
                            if let Some(map) = self.exports.get(module_id) {
                                if let Some(Type::Function(params, ret)) = map.get(name).cloned() {
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
                                    self.check_static_policy_for_module_call(
                                        alias,
                                        name,
                                        args,
                                        span.clone(),
                                    )?;
                                    return Ok(*ret);
                                }
                            }
                        }
                    }
                    let target_type = self.check_expr(target)?;
                    if let Type::Named(type_name) = target_type {
                        if self.record_field_type(&type_name, name).is_none() {
                            for arg in args {
                                let expr = match arg {
                                    Arg::Positional(expr) => expr,
                                    Arg::Named(_, expr) => expr,
                                };
                                self.check_expr(expr)?;
                            }
                            return Ok(Type::Unknown);
                        }
                    }
                }
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
                } else if ct == Type::Unknown {
                    for arg in args {
                        let expr = match arg {
                            Arg::Positional(e) => e,
                            Arg::Named(_, e) => e,
                        };
                        self.check_expr(expr)?;
                    }
                    Ok(Type::Unknown)
                } else {
                    Err(self.error("Callee is not a function", span.clone()))
                }
            }
            Expr::Field { target, name, span } => {
                if let Expr::Ident { name: alias, .. } = &**target {
                    if alias == "json" && !self.imports.contains_key(alias) {
                        return Err(self.error(
                            format!("ImportError: `json.{}` requires `import std::json`", name),
                            span.clone(),
                        ));
                    }
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
                let target_type = self.check_expr(target)?;
                if let Type::Named(type_name) = target_type {
                    if let Some(field_type) = self.record_field_type(&type_name, name) {
                        return Ok(field_type);
                    }
                    return Err(self.error(
                        format!("Unknown field {}.{}", type_name, name),
                        span.clone(),
                    ));
                }
                Ok(Type::Unknown)
            }
            Expr::Index { target, index, .. } => {
                let target_type = self.check_expr(target)?;
                let index_type = self.check_expr(index)?;
                if index_type != Type::Int && index_type != Type::Unknown {
                    return Err(self.error("Index expects Int", line_span(index)));
                }
                match target_type {
                    Type::Array(inner) | Type::Vector(inner) => Ok(*inner),
                    Type::List(inner) => Ok(*inner),
                    Type::Unknown => Ok(Type::Unknown),
                    _ => Ok(Type::Unknown),
                }
            }
            Expr::List { items, .. } => Ok(self.infer_array_items(items)?.as_type()),
            Expr::Try { expr, .. } => match self.check_expr(expr)? {
                Type::Optional(inner) => Ok(*inner),
                _ => Ok(Type::Unknown),
            },
            Expr::Lambda {
                params,
                return_type,
                body,
                span,
            } => {
                let mut param_types = Vec::with_capacity(params.len());
                self.push();
                for param in params {
                    let declared = type_from_ref_opt(param.type_ann.clone());
                    if let Some(default_expr) = &param.default {
                        let default_ty = self.check_expr(default_expr)?;
                        if declared != Type::Unknown && !compatible(&declared, &default_ty) {
                            self.pop();
                            return Err(self.error(
                                format!(
                                    "Default value type mismatch for parameter {}: expected {}, found {}",
                                    param.name,
                                    declared.display(),
                                    default_ty.display()
                                ),
                                line_span(default_expr),
                            ));
                        }
                    }
                    self.define_binding(&param.name, declared.clone(), param.mutable);
                    param_types.push(declared);
                }
                let body_ty = match self.check_expr(body) {
                    Ok(ty) => ty,
                    Err(err) => {
                        self.pop();
                        return Err(err);
                    }
                };
                self.pop();
                let ret_ty = if let Some(ret) = return_type.clone() {
                    let declared_ret = type_from_ref(ret);
                    if declared_ret != Type::Unknown && !compatible(&declared_ret, &body_ty) {
                        return Err(self.error(
                            format!(
                                "Lambda return type mismatch: expected {}, found {}",
                                declared_ret.display(),
                                body_ty.display()
                            ),
                            span.clone(),
                        ));
                    }
                    declared_ret
                } else {
                    body_ty
                };
                Ok(Type::Function(param_types, Box::new(ret_ty)))
            }
            _ => Ok(Type::Unknown),
        }
    }

    fn infer_array_literal(&mut self, expr: &Expr) -> Result<ArrayLiteralType, TypeError> {
        match expr {
            Expr::List { items, .. } => self.infer_array_items(items),
            _ => Ok(ArrayLiteralType::Homogeneous(self.check_expr(expr)?)),
        }
    }

    fn infer_array_items(&mut self, items: &[Expr]) -> Result<ArrayLiteralType, TypeError> {
        if items.is_empty() {
            return Ok(ArrayLiteralType::Empty);
        }
        let mut item_type: Option<Type> = None;
        for item in items {
            let current = self.check_expr(item)?;
            item_type = Some(match item_type {
                None => current,
                Some(previous) => unify_array_item_type(previous, current).unwrap_or(Type::Unknown),
            });
        }
        let inferred = item_type.unwrap_or(Type::Unknown);
        if inferred == Type::Unknown {
            Ok(ArrayLiteralType::Mixed)
        } else {
            Ok(ArrayLiteralType::Homogeneous(inferred))
        }
    }

    fn is_compile_time_const_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal { .. } => true,
            Expr::Ident { name, .. } => self
                .resolve_binding(name)
                .map(|binding| binding.constant)
                .unwrap_or(false),
            Expr::Unary { op, expr, .. } => {
                matches!(op, UnaryOp::Negate | UnaryOp::Not)
                    && self.is_compile_time_const_expr(expr)
            }
            Expr::Binary { left, right, .. } => {
                self.is_compile_time_const_expr(left) && self.is_compile_time_const_expr(right)
            }
            Expr::List { items, .. } => items
                .iter()
                .all(|item| self.is_compile_time_const_expr(item)),
            _ => false,
        }
    }

    fn check_static_policy_for_module_call(
        &self,
        alias: &str,
        function: &str,
        args: &[Arg],
        span: Span,
    ) -> Result<(), TypeError> {
        let Some(module_id) = self.imports.get(alias) else {
            return Ok(());
        };
        let Some(capability) = static_capability_for_std_call(module_id, function) else {
            return Ok(());
        };
        let context = static_context_for_call(&capability, args);
        self.check_static_policy_capability(&capability, context.as_ref(), span)
    }

    fn check_static_policy_capability(
        &self,
        capability: &[String],
        context: Option<&StaticCapabilityContext>,
        span: Span,
    ) -> Result<(), TypeError> {
        let Some(policy) = self.policies.get("default") else {
            return Err(self.error(
                format!(
                    "PolicyError: `{}` requires `policy default` with an explicit `allow {}` rule.",
                    capability.join("."),
                    capability.join(".")
                ),
                span,
            ));
        };
        let mut allowed = false;
        let mut filtered_allow = false;
        for rule in &policy.rules {
            if !capability_matches_static(&rule.capability, capability) {
                continue;
            }
            let filter_result = static_filters_match(&rule.filters, context);
            if filter_result == FilterMatch::No {
                continue;
            }
            if !rule.allow {
                return Err(self.error(
                    format!(
                        "PolicyError: {} is denied by policy `default`.",
                        capability.join(".")
                    ),
                    span,
                ));
            }
            if filter_result == FilterMatch::Unknown {
                filtered_allow = true;
            } else {
                allowed = true;
            }
        }
        if allowed {
            return Ok(());
        }
        if filtered_allow {
            return Err(self.error(
                format!(
                    "PolicyError: {} requires a statically matching policy filter or an unfiltered `allow {}` rule.",
                    capability.join("."),
                    capability.join(".")
                ),
                span,
            ));
        }
        Err(self.error(
            format!(
                "PolicyError: {} is not allowed by policy `default`.",
                capability.join(".")
            ),
            span,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArrayLiteralType {
    Empty,
    Mixed,
    Homogeneous(Type),
}

impl ArrayLiteralType {
    fn as_type(&self) -> Type {
        match self {
            ArrayLiteralType::Empty | ArrayLiteralType::Mixed => {
                Type::Array(Box::new(Type::Unknown))
            }
            ArrayLiteralType::Homogeneous(item) => Type::Array(Box::new(item.clone())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterMatch {
    Yes,
    No,
    Unknown,
}

fn policy_filter_static_values(filter: &PolicyFilter) -> Result<Vec<String>, String> {
    match &filter.value {
        LiteralOrList::Literal(lit) => literal_to_policy_string(lit)
            .map(|value| vec![value])
            .ok_or_else(|| format!("policy filter `{}` requires string values", filter.name)),
        LiteralOrList::List(values) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                let Some(value) = literal_to_policy_string(value) else {
                    return Err(format!(
                        "policy filter `{}` requires string values",
                        filter.name
                    ));
                };
                out.push(value);
            }
            Ok(out)
        }
    }
}

fn literal_to_policy_string(lit: &Literal) -> Option<String> {
    match lit {
        Literal::String(value) => Some(value.clone()),
        _ => None,
    }
}

fn static_capability_for_std_call(module_id: &ModuleId, function: &str) -> Option<Vec<String>> {
    let path = module_id.0.iter().map(String::as_str).collect::<Vec<_>>();
    match path.as_slice() {
        ["std", "io"] => match function {
            "read_bytes" | "read_text" => Some(cap("fs.read")),
            "write_bytes" | "write_text" => Some(cap("fs.write")),
            "stdin_read" => Some(cap("io.read")),
            "stdout_write" | "stderr_write" | "stdout_write_text" | "stderr_write_text" => {
                Some(cap("io.write"))
            }
            _ => None,
        },
        ["std", "fsx"] => match function {
            "read_bytes" | "mmap_read" => Some(cap("fs.read")),
            "write_bytes" => Some(cap("fs.write")),
            _ => None,
        },
        ["std", "env"] => match function {
            "get" | "cwd" => Some(cap("env.read")),
            "set" | "remove" | "set_cwd" => Some(cap("env.write")),
            _ => None,
        },
        ["std", "process"] => match function {
            "start" | "run" => Some(cap("process.spawn")),
            "wait" | "kill" => Some(cap("process.control")),
            "exit" => Some(cap("process.exit")),
            _ => None,
        },
        ["std", "db"] => match function {
            "sqlite_exec"
            | "sqlite_exec_many"
            | "sqlite_transaction_begin"
            | "sqlite_transaction_commit"
            | "pg_exec"
            | "mysql_exec" => Some(cap("db.write")),
            "sqlite_open" | "sqlite_close" | "sqlite_query" | "pg_open" | "pg_close"
            | "pg_query" | "mysql_open" | "mysql_close" | "mysql_query" => Some(cap("db.read")),
            _ => None,
        },
        ["std", "tls"] => match function {
            "inspect" => Some(cap("net.tls")),
            _ => None,
        },
        ["std", "http"] => match function {
            "serve" => Some(cap("net.serve")),
            "sse_send" | "sse_close" | "ws_send" | "ws_recv" | "ws_recv_timeout" | "ws_close" => {
                Some(cap("net.http"))
            }
            _ => None,
        },
        ["std", "log"] => match function {
            "info" | "warn" | "error" => Some(cap("io.log")),
            _ => None,
        },
        ["std", "time"] => match function {
            "sleep_ms" => Some(cap("time.sleep")),
            _ => None,
        },
        _ => None,
    }
}

fn cap(path: &str) -> Vec<String> {
    path.split('.').map(|segment| segment.to_string()).collect()
}

fn static_context_for_call(capability: &[String], args: &[Arg]) -> Option<StaticCapabilityContext> {
    let first_arg = args.first().and_then(arg_expr)?;
    match capability.first().map(String::as_str) {
        Some("fs") => string_literal(first_arg).map(StaticCapabilityContext::Path),
        Some("net") => string_literal(first_arg).map(StaticCapabilityContext::Domain),
        _ => None,
    }
}

fn arg_expr(arg: &Arg) -> Option<&Expr> {
    match arg {
        Arg::Positional(expr) => Some(expr),
        Arg::Named(_, _) => None,
    }
}

fn string_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal {
            lit: Literal::String(value),
            ..
        } => Some(value.clone()),
        _ => None,
    }
}

fn capability_matches_static(rule: &[String], requested: &[String]) -> bool {
    if rule.is_empty() || rule.len() > requested.len() {
        return false;
    }
    rule.iter().zip(requested.iter()).all(|(a, b)| a == b)
}

fn static_filters_match(
    filters: &[StaticPolicyFilter],
    context: Option<&StaticCapabilityContext>,
) -> FilterMatch {
    if filters.is_empty() {
        return FilterMatch::Yes;
    }
    let Some(context) = context else {
        return FilterMatch::Unknown;
    };
    let mut any_unknown = false;
    for filter in filters {
        match static_filter_matches(filter, context) {
            FilterMatch::Yes => {}
            FilterMatch::No => return FilterMatch::No,
            FilterMatch::Unknown => any_unknown = true,
        }
    }
    if any_unknown {
        FilterMatch::Unknown
    } else {
        FilterMatch::Yes
    }
}

fn static_filter_matches(
    filter: &StaticPolicyFilter,
    context: &StaticCapabilityContext,
) -> FilterMatch {
    match (filter.name.as_str(), context) {
        ("path_prefix", StaticCapabilityContext::Path(path)) => {
            if filter.values.iter().any(|prefix| path.starts_with(prefix)) {
                FilterMatch::Yes
            } else {
                FilterMatch::No
            }
        }
        ("domain", StaticCapabilityContext::Domain(domain)) => {
            if filter.values.iter().any(|allowed| {
                domain == allowed
                    || domain
                        .strip_suffix(allowed)
                        .map(|prefix| prefix.ends_with('.'))
                        .unwrap_or(false)
            }) {
                FilterMatch::Yes
            } else {
                FilterMatch::No
            }
        }
        _ => FilterMatch::Unknown,
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
        TypeRef::Named {
            path,
            args,
            optional,
        } => {
            let generic = |index: usize| {
                args.get(index)
                    .cloned()
                    .map(type_from_ref)
                    .unwrap_or(Type::Unknown)
            };
            let mut t = match path.first().map(|s| s.as_str()) {
                Some("Int") => Type::Int,
                Some("Float") => Type::Float,
                Some("Bool") => Type::Bool,
                Some("String") => Type::String,
                Some("Buffer") => Type::Buffer,
                Some("Handle") => Type::Handle,
                Some("Array") => Type::Array(Box::new(generic(0))),
                Some("Vector") => Type::Vector(Box::new(generic(0))),
                Some("SparseVector") if !args.is_empty() => {
                    Type::SparseVectorOf(Box::new(generic(0)))
                }
                Some("SparseVector") => Type::SparseVector,
                Some("SparseMatrix") => Type::SparseMatrix,
                Some("EventQueue") => Type::EventQueue,
                Some("Pool") => Type::Pool,
                Some("SimWorld") => Type::SimWorld,
                Some("SimCoroutine") => Type::SimCoroutine,
                Some("SpatialIndex") => Type::SpatialIndex,
                Some("SnnNetwork") => Type::SnnNetwork,
                Some("AgentEnv") => Type::AgentEnv,
                Some("RngStream") => Type::RngStream,
                Some("Tokenizer") => Type::Tokenizer,
                Some("DataStream") => Type::DataStream,
                Some("Batch") => Type::Batch,
                Some("Tensor") if !args.is_empty() => Type::TensorOf(
                    Box::new(generic(0)),
                    args.get(1).and_then(type_ref_const_usize),
                ),
                Some("Tensor") => Type::Tensor,
                Some("Device") => Type::Device,
                Some("DType") => Type::DType,
                Some("Shape") => Type::Shape,
                Some("OptimizerState") => Type::OptimizerState,
                Some("TcpListener") => Type::TcpListener,
                Some("TcpConnection") => Type::TcpConnection,
                Some("Request") => Type::HttpRequest,
                Some("Response") => Type::HttpResponse,
                Some("HttpStream") => Type::HttpStream,
                Some("Any") => Type::Unknown,
                Some("Record") => Type::Unknown,
                Some("List") if !args.is_empty() => Type::Array(Box::new(generic(0))),
                Some("List") => Type::Unknown,
                Some("Void") => Type::Void,
                Some(_) => Type::Named(path.join("::")),
                None => Type::Unknown,
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
        TypeRef::ConstInt(_) => Type::Unknown,
    }
}

fn compatible(expected: &Type, found: &Type) -> bool {
    if expected == found || *expected == Type::Unknown || *found == Type::Unknown {
        return true;
    }
    match (expected, found) {
        (Type::Optional(inner), Type::Optional(f)) => compatible(inner, f),
        (Type::Optional(inner), f) => compatible(inner, f), // allow T to satisfy Optional<T>
        (Type::List(expected_inner), Type::List(found_inner)) => {
            compatible(expected_inner, found_inner)
        }
        (Type::Array(expected_inner), Type::Array(found_inner))
        | (Type::Array(expected_inner), Type::List(found_inner))
        | (Type::List(expected_inner), Type::Array(found_inner))
        | (Type::Vector(expected_inner), Type::Vector(found_inner))
        | (Type::Vector(expected_inner), Type::Array(found_inner))
        | (Type::Array(expected_inner), Type::Vector(found_inner)) => {
            compatible(expected_inner, found_inner)
        }
        (Type::SparseVectorOf(expected_inner), Type::SparseVectorOf(found_inner)) => {
            compatible(expected_inner, found_inner)
        }
        (Type::SparseVectorOf(_), Type::SparseVector)
        | (Type::SparseVector, Type::SparseVectorOf(_)) => true,
        (
            Type::TensorOf(expected_inner, expected_rank),
            Type::TensorOf(found_inner, found_rank),
        ) => {
            compatible(expected_inner, found_inner)
                && (expected_rank.is_none() || found_rank.is_none() || expected_rank == found_rank)
        }
        (Type::TensorOf(_, _), Type::Tensor)
        | (Type::Tensor, Type::TensorOf(_, _))
        | (Type::TensorOf(_, _), Type::Int)
        | (Type::Int, Type::TensorOf(_, _)) => true,
        (
            Type::Function(expected_params, expected_ret),
            Type::Function(found_params, found_ret),
        ) => {
            expected_params.len() == found_params.len()
                && expected_params
                    .iter()
                    .zip(found_params.iter())
                    .all(|(expected, found)| compatible(expected, found))
                && compatible(expected_ret, found_ret)
        }
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

fn unify_array_item_type(left: Type, right: Type) -> Option<Type> {
    if left == right {
        return Some(left);
    }
    if left == Type::Unknown || right == Type::Unknown {
        return Some(Type::Unknown);
    }
    match (&left, &right) {
        (Type::Int, Type::Float) | (Type::Float, Type::Int) => Some(Type::Float),
        _ => None,
    }
}

fn type_ref_const_usize(value: &TypeRef) -> Option<usize> {
    match value {
        TypeRef::ConstInt(raw) if *raw >= 0 => Some(*raw as usize),
        _ => None,
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
    let array_id = ModuleId(vec!["array".to_string()]);
    imports
        .entry("array".to_string())
        .or_insert(array_id.clone());
    let mut array_exports = std::collections::HashMap::new();
    array_exports.insert(
        "len".to_string(),
        Type::Function(
            vec![Type::Array(Box::new(Type::Unknown))],
            Box::new(Type::Int),
        ),
    );
    array_exports.insert(
        "element_type".to_string(),
        Type::Function(
            vec![Type::Array(Box::new(Type::Unknown))],
            Box::new(Type::String),
        ),
    );
    array_exports.insert(
        "is_homogeneous".to_string(),
        Type::Function(
            vec![Type::Array(Box::new(Type::Unknown))],
            Box::new(Type::Bool),
        ),
    );
    exports.entry(array_id).or_insert(array_exports.clone());
    exports
        .entry(ModuleId(vec!["std".to_string(), "array".to_string()]))
        .or_insert(array_exports);
    let vector_id = ModuleId(vec!["vector".to_string()]);
    imports
        .entry("vector".to_string())
        .or_insert(vector_id.clone());
    let mut vector_exports = std::collections::HashMap::new();
    vector_exports.insert(
        "from_array".to_string(),
        Type::Function(
            vec![Type::Array(Box::new(Type::Float))],
            Box::new(Type::Vector(Box::new(Type::Float))),
        ),
    );
    vector_exports.insert(
        "len".to_string(),
        Type::Function(
            vec![Type::Vector(Box::new(Type::Float))],
            Box::new(Type::Int),
        ),
    );
    vector_exports.insert(
        "get".to_string(),
        Type::Function(
            vec![Type::Vector(Box::new(Type::Float)), Type::Int],
            Box::new(Type::Float),
        ),
    );
    vector_exports.insert(
        "dot".to_string(),
        Type::Function(
            vec![
                Type::Vector(Box::new(Type::Float)),
                Type::Vector(Box::new(Type::Float)),
            ],
            Box::new(Type::Float),
        ),
    );
    vector_exports.insert(
        "add".to_string(),
        Type::Function(
            vec![
                Type::Vector(Box::new(Type::Float)),
                Type::Vector(Box::new(Type::Float)),
            ],
            Box::new(Type::Vector(Box::new(Type::Float))),
        ),
    );
    vector_exports.insert(
        "scale".to_string(),
        Type::Function(
            vec![Type::Vector(Box::new(Type::Float)), Type::Float],
            Box::new(Type::Vector(Box::new(Type::Float))),
        ),
    );
    vector_exports.insert(
        "to_array".to_string(),
        Type::Function(
            vec![Type::Vector(Box::new(Type::Float))],
            Box::new(Type::Array(Box::new(Type::Float))),
        ),
    );
    exports.entry(vector_id).or_insert(vector_exports.clone());
    exports
        .entry(ModuleId(vec!["std".to_string(), "vector".to_string()]))
        .or_insert(vector_exports);
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
        "serve_with".to_string(),
        Type::Function(
            vec![Type::String, Type::Int, Type::Unknown, Type::Unknown],
            Box::new(Type::Void),
        ),
    );
    http_exports.insert(
        "route".to_string(),
        Type::Function(
            vec![
                Type::String,
                Type::String,
                Type::Function(vec![Type::HttpRequest], Box::new(Type::HttpResponse)),
            ],
            Box::new(Type::Unknown),
        ),
    );
    http_exports.insert(
        "middleware".to_string(),
        Type::Function(vec![Type::String, Type::Unknown], Box::new(Type::Unknown)),
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
        "request".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::HttpResponse)),
    );
    http_exports.insert(
        "header".to_string(),
        Type::Function(
            vec![Type::HttpRequest, Type::String],
            Box::new(Type::Optional(Box::new(Type::String))),
        ),
    );
    http_exports.insert(
        "query".to_string(),
        Type::Function(
            vec![Type::HttpRequest, Type::String],
            Box::new(Type::Optional(Box::new(Type::String))),
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
    http_exports.insert(
        "stream_open".to_string(),
        Type::Function(vec![Type::Int, Type::Unknown], Box::new(Type::HttpStream)),
    );
    http_exports.insert(
        "stream_send".to_string(),
        Type::Function(vec![Type::HttpStream, Type::Unknown], Box::new(Type::Void)),
    );
    http_exports.insert(
        "stream_close".to_string(),
        Type::Function(vec![Type::HttpStream], Box::new(Type::Void)),
    );
    http_exports.insert(
        "ws_open".to_string(),
        Type::Function(vec![Type::HttpRequest], Box::new(Type::Unknown)),
    );
    http_exports.insert(
        "ws_send".to_string(),
        Type::Function(vec![Type::Unknown, Type::Unknown], Box::new(Type::Void)),
    );
    http_exports.insert(
        "ws_recv".to_string(),
        Type::Function(
            vec![Type::Unknown, Type::Int],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
        ),
    );
    http_exports.insert(
        "ws_close".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::Void)),
    );
    exports.entry(http_id).or_insert(http_exports);
    let tool_id = ModuleId(vec!["tool".to_string()]);
    imports.entry("tool".to_string()).or_insert(tool_id.clone());
    let mut tool_exports = std::collections::HashMap::new();
    tool_exports.insert(
        "invoke".to_string(),
        Type::Function(vec![Type::String, Type::Unknown], Box::new(Type::Unknown)),
    );
    exports.entry(tool_id).or_insert(tool_exports);
    let json_id = ModuleId(vec!["json".to_string()]);
    let mut json_exports = std::collections::HashMap::new();
    json_exports.insert(
        "parse".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    json_exports.insert(
        "stringify".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::String)),
    );
    json_exports.insert(
        "enkai".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::String)),
    );
    json_exports.insert(
        "parse_many".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::Unknown)),
    );
    json_exports.insert(
        "stringify_many".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::Unknown)),
    );
    exports.entry(json_id).or_insert(json_exports.clone());
    exports
        .entry(ModuleId(vec!["std".to_string(), "json".to_string()]))
        .or_insert(json_exports);
    let bootstrap_id = ModuleId(vec!["bootstrap".to_string()]);
    imports
        .entry("bootstrap".to_string())
        .or_insert(bootstrap_id.clone());
    let mut bootstrap_exports = std::collections::HashMap::new();
    bootstrap_exports.insert(
        "format".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::String)),
    );
    bootstrap_exports.insert(
        "check".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Bool)),
    );
    bootstrap_exports.insert(
        "lint".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    bootstrap_exports.insert(
        "lint_count".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Int)),
    );
    bootstrap_exports.insert(
        "lint_json".to_string(),
        Type::Function(vec![Type::String, Type::String], Box::new(Type::String)),
    );
    exports.entry(bootstrap_id).or_insert(bootstrap_exports);
    let compiler_id = ModuleId(vec!["compiler".to_string()]);
    imports
        .entry("compiler".to_string())
        .or_insert(compiler_id.clone());
    let mut compiler_exports = std::collections::HashMap::new();
    compiler_exports.insert(
        "parse_subset".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    compiler_exports.insert(
        "check_subset".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Bool)),
    );
    compiler_exports.insert(
        "emit_subset".to_string(),
        Type::Function(vec![Type::String, Type::String], Box::new(Type::Bool)),
    );
    compiler_exports.insert(
        "parse_subset_file".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    compiler_exports.insert(
        "describe_subset".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    compiler_exports.insert(
        "describe_subset_file".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    compiler_exports.insert(
        "describe_subset_package_file".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    compiler_exports.insert(
        "describe_program_file".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Unknown)),
    );
    compiler_exports.insert(
        "check_subset_file".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Bool)),
    );
    compiler_exports.insert(
        "check_subset_raw".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Bool)),
    );
    compiler_exports.insert(
        "check_subset_raw_file".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Bool)),
    );
    compiler_exports.insert(
        "emit_subset_file".to_string(),
        Type::Function(vec![Type::String, Type::String], Box::new(Type::Bool)),
    );
    compiler_exports.insert(
        "emit_subset_raw".to_string(),
        Type::Function(vec![Type::String, Type::String], Box::new(Type::Bool)),
    );
    compiler_exports.insert(
        "emit_subset_raw_file".to_string(),
        Type::Function(vec![Type::String, Type::String], Box::new(Type::Bool)),
    );
    exports.entry(compiler_id).or_insert(compiler_exports);
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
    tokenizer_exports.insert(
        "save".to_string(),
        Type::Function(vec![Type::Tokenizer, Type::String], Box::new(Type::Void)),
    );
    tokenizer_exports.insert(
        "encode".to_string(),
        Type::Function(vec![Type::Tokenizer, Type::String], Box::new(Type::Buffer)),
    );
    tokenizer_exports.insert(
        "decode".to_string(),
        Type::Function(vec![Type::Tokenizer, Type::Unknown], Box::new(Type::String)),
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
    dataset_exports.insert(
        "next_batch".to_string(),
        Type::Function(
            vec![Type::DataStream],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
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
    let std_sparse_id = ModuleId(vec!["std".to_string(), "sparse".to_string()]);
    imports
        .entry("sparse".to_string())
        .or_insert(std_sparse_id.clone());
    let mut sparse_exports = std::collections::HashMap::new();
    sparse_exports.insert(
        "vector".to_string(),
        Type::Function(Vec::new(), Box::new(Type::SparseVector)),
    );
    sparse_exports.insert(
        "matrix".to_string(),
        Type::Function(Vec::new(), Box::new(Type::SparseMatrix)),
    );
    sparse_exports.insert(
        "get".to_string(),
        Type::Function(
            vec![Type::SparseMatrix, Type::Int, Type::Int],
            Box::new(Type::Optional(Box::new(Type::Float))),
        ),
    );
    sparse_exports.insert(
        "set".to_string(),
        Type::Function(
            vec![Type::SparseMatrix, Type::Int, Type::Int, Type::Float],
            Box::new(Type::Void),
        ),
    );
    sparse_exports.insert(
        "get_vector".to_string(),
        Type::Function(
            vec![Type::SparseVector, Type::Int],
            Box::new(Type::Optional(Box::new(Type::Float))),
        ),
    );
    sparse_exports.insert(
        "set_vector".to_string(),
        Type::Function(
            vec![Type::SparseVector, Type::Int, Type::Float],
            Box::new(Type::Void),
        ),
    );
    sparse_exports.insert(
        "nonzero".to_string(),
        Type::Function(vec![Type::SparseMatrix], Box::new(Type::Unknown)),
    );
    sparse_exports.insert(
        "nonzero_vector".to_string(),
        Type::Function(vec![Type::SparseVector], Box::new(Type::Unknown)),
    );
    sparse_exports.insert(
        "dot".to_string(),
        Type::Function(
            vec![Type::SparseVector, Type::Unknown],
            Box::new(Type::Float),
        ),
    );
    sparse_exports.insert(
        "matvec".to_string(),
        Type::Function(
            vec![Type::SparseMatrix, Type::Unknown],
            Box::new(Type::Unknown),
        ),
    );
    sparse_exports.insert(
        "nnz".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::Int)),
    );
    exports.entry(std_sparse_id).or_insert(sparse_exports);
    let std_event_id = ModuleId(vec!["std".to_string(), "event".to_string()]);
    imports
        .entry("event".to_string())
        .or_insert(std_event_id.clone());
    let mut event_exports = std::collections::HashMap::new();
    event_exports.insert(
        "make".to_string(),
        Type::Function(Vec::new(), Box::new(Type::EventQueue)),
    );
    event_exports.insert(
        "push".to_string(),
        Type::Function(
            vec![Type::EventQueue, Type::Float, Type::Unknown],
            Box::new(Type::Void),
        ),
    );
    event_exports.insert(
        "pop".to_string(),
        Type::Function(
            vec![Type::EventQueue],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
        ),
    );
    event_exports.insert(
        "peek".to_string(),
        Type::Function(
            vec![Type::EventQueue],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
        ),
    );
    event_exports.insert(
        "len".to_string(),
        Type::Function(vec![Type::EventQueue], Box::new(Type::Int)),
    );
    event_exports.insert(
        "is_empty".to_string(),
        Type::Function(vec![Type::EventQueue], Box::new(Type::Bool)),
    );
    exports.entry(std_event_id).or_insert(event_exports);
    let std_pool_id = ModuleId(vec!["std".to_string(), "pool".to_string()]);
    imports
        .entry("pool".to_string())
        .or_insert(std_pool_id.clone());
    let mut pool_exports = std::collections::HashMap::new();
    pool_exports.insert(
        "make".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::Pool)),
    );
    pool_exports.insert(
        "make_growable".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::Pool)),
    );
    pool_exports.insert(
        "acquire".to_string(),
        Type::Function(
            vec![Type::Pool],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
        ),
    );
    pool_exports.insert(
        "release".to_string(),
        Type::Function(vec![Type::Pool, Type::Unknown], Box::new(Type::Bool)),
    );
    pool_exports.insert(
        "reset".to_string(),
        Type::Function(vec![Type::Pool], Box::new(Type::Void)),
    );
    pool_exports.insert(
        "available".to_string(),
        Type::Function(vec![Type::Pool], Box::new(Type::Int)),
    );
    pool_exports.insert(
        "capacity".to_string(),
        Type::Function(vec![Type::Pool], Box::new(Type::Int)),
    );
    pool_exports.insert(
        "stats".to_string(),
        Type::Function(vec![Type::Pool], Box::new(Type::Unknown)),
    );
    exports.entry(std_pool_id).or_insert(pool_exports);
    let std_sim_id = ModuleId(vec!["std".to_string(), "sim".to_string()]);
    imports
        .entry("sim".to_string())
        .or_insert(std_sim_id.clone());
    let mut sim_exports = std::collections::HashMap::new();
    sim_exports.insert(
        "make".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::SimWorld)),
    );
    sim_exports.insert(
        "make_seeded".to_string(),
        Type::Function(vec![Type::Int, Type::Int], Box::new(Type::SimWorld)),
    );
    sim_exports.insert(
        "time".to_string(),
        Type::Function(vec![Type::SimWorld], Box::new(Type::Float)),
    );
    sim_exports.insert(
        "seed".to_string(),
        Type::Function(vec![Type::SimWorld], Box::new(Type::Int)),
    );
    sim_exports.insert(
        "pending".to_string(),
        Type::Function(vec![Type::SimWorld], Box::new(Type::Int)),
    );
    sim_exports.insert(
        "schedule".to_string(),
        Type::Function(
            vec![Type::SimWorld, Type::Float, Type::Unknown],
            Box::new(Type::Void),
        ),
    );
    sim_exports.insert(
        "step".to_string(),
        Type::Function(
            vec![Type::SimWorld],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
        ),
    );
    sim_exports.insert(
        "run".to_string(),
        Type::Function(vec![Type::SimWorld, Type::Int], Box::new(Type::Unknown)),
    );
    sim_exports.insert(
        "snapshot".to_string(),
        Type::Function(vec![Type::SimWorld], Box::new(Type::Unknown)),
    );
    sim_exports.insert(
        "restore".to_string(),
        Type::Function(vec![Type::Unknown], Box::new(Type::SimWorld)),
    );
    sim_exports.insert(
        "replay".to_string(),
        Type::Function(
            vec![Type::Unknown, Type::Int, Type::Int],
            Box::new(Type::SimWorld),
        ),
    );
    sim_exports.insert(
        "log".to_string(),
        Type::Function(vec![Type::SimWorld], Box::new(Type::Unknown)),
    );
    sim_exports.insert(
        "entity_set".to_string(),
        Type::Function(
            vec![Type::SimWorld, Type::Int, Type::Unknown],
            Box::new(Type::Void),
        ),
    );
    sim_exports.insert(
        "entity_get".to_string(),
        Type::Function(
            vec![Type::SimWorld, Type::Int],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
        ),
    );
    sim_exports.insert(
        "entity_remove".to_string(),
        Type::Function(vec![Type::SimWorld, Type::Int], Box::new(Type::Bool)),
    );
    sim_exports.insert(
        "entity_ids".to_string(),
        Type::Function(vec![Type::SimWorld], Box::new(Type::Unknown)),
    );
    sim_exports.insert(
        "coroutine".to_string(),
        Type::Function(
            vec![Type::SimWorld, Type::Unknown],
            Box::new(Type::SimCoroutine),
        ),
    );
    sim_exports.insert(
        "coroutine_with".to_string(),
        Type::Function(
            vec![Type::SimWorld, Type::Unknown, Type::Unknown],
            Box::new(Type::SimCoroutine),
        ),
    );
    sim_exports.insert(
        "coroutine_args".to_string(),
        Type::Function(
            vec![Type::SimWorld, Type::Unknown, Type::Unknown],
            Box::new(Type::SimCoroutine),
        ),
    );
    sim_exports.insert(
        "world".to_string(),
        Type::Function(vec![Type::SimCoroutine], Box::new(Type::SimWorld)),
    );
    sim_exports.insert(
        "state".to_string(),
        Type::Function(
            vec![Type::SimCoroutine],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
        ),
    );
    sim_exports.insert(
        "emit".to_string(),
        Type::Function(
            vec![Type::SimCoroutine, Type::Unknown],
            Box::new(Type::Void),
        ),
    );
    sim_exports.insert(
        "next".to_string(),
        Type::Function(
            vec![Type::SimCoroutine],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
        ),
    );
    sim_exports.insert(
        "join".to_string(),
        Type::Function(
            vec![Type::SimCoroutine],
            Box::new(Type::Optional(Box::new(Type::Unknown))),
        ),
    );
    sim_exports.insert(
        "done".to_string(),
        Type::Function(vec![Type::SimCoroutine], Box::new(Type::Bool)),
    );
    exports.entry(std_sim_id).or_insert(sim_exports);
    let std_spatial_id = ModuleId(vec!["std".to_string(), "spatial".to_string()]);
    imports
        .entry("spatial".to_string())
        .or_insert(std_spatial_id.clone());
    let mut spatial_exports = std::collections::HashMap::new();
    spatial_exports.insert(
        "make".to_string(),
        Type::Function(Vec::new(), Box::new(Type::SpatialIndex)),
    );
    spatial_exports.insert(
        "upsert".to_string(),
        Type::Function(
            vec![Type::SpatialIndex, Type::Int, Type::Float, Type::Float],
            Box::new(Type::Void),
        ),
    );
    spatial_exports.insert(
        "remove".to_string(),
        Type::Function(vec![Type::SpatialIndex, Type::Int], Box::new(Type::Bool)),
    );
    spatial_exports.insert(
        "radius".to_string(),
        Type::Function(
            vec![Type::SpatialIndex, Type::Float, Type::Float, Type::Float],
            Box::new(Type::Unknown),
        ),
    );
    spatial_exports.insert(
        "nearest".to_string(),
        Type::Function(
            vec![Type::SpatialIndex, Type::Float, Type::Float],
            Box::new(Type::Optional(Box::new(Type::Int))),
        ),
    );
    spatial_exports.insert(
        "occupancy".to_string(),
        Type::Function(
            vec![
                Type::SpatialIndex,
                Type::Float,
                Type::Float,
                Type::Float,
                Type::Float,
            ],
            Box::new(Type::Int),
        ),
    );
    exports.entry(std_spatial_id).or_insert(spatial_exports);
    let std_snn_id = ModuleId(vec!["std".to_string(), "snn".to_string()]);
    imports
        .entry("snn".to_string())
        .or_insert(std_snn_id.clone());
    let mut snn_exports = std::collections::HashMap::new();
    snn_exports.insert(
        "make".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::SnnNetwork)),
    );
    snn_exports.insert(
        "connect".to_string(),
        Type::Function(
            vec![Type::SnnNetwork, Type::Int, Type::Int, Type::Float],
            Box::new(Type::Void),
        ),
    );
    snn_exports.insert(
        "set_potential".to_string(),
        Type::Function(
            vec![Type::SnnNetwork, Type::Int, Type::Float],
            Box::new(Type::Void),
        ),
    );
    snn_exports.insert(
        "get_potential".to_string(),
        Type::Function(
            vec![Type::SnnNetwork, Type::Int],
            Box::new(Type::Optional(Box::new(Type::Float))),
        ),
    );
    snn_exports.insert(
        "set_threshold".to_string(),
        Type::Function(
            vec![Type::SnnNetwork, Type::Int, Type::Float],
            Box::new(Type::Void),
        ),
    );
    snn_exports.insert(
        "get_threshold".to_string(),
        Type::Function(
            vec![Type::SnnNetwork, Type::Int],
            Box::new(Type::Optional(Box::new(Type::Float))),
        ),
    );
    snn_exports.insert(
        "set_decay".to_string(),
        Type::Function(vec![Type::SnnNetwork, Type::Float], Box::new(Type::Void)),
    );
    snn_exports.insert(
        "get_decay".to_string(),
        Type::Function(vec![Type::SnnNetwork], Box::new(Type::Float)),
    );
    snn_exports.insert(
        "step".to_string(),
        Type::Function(
            vec![Type::SnnNetwork, Type::Unknown],
            Box::new(Type::Unknown),
        ),
    );
    snn_exports.insert(
        "spikes".to_string(),
        Type::Function(vec![Type::SnnNetwork], Box::new(Type::Unknown)),
    );
    snn_exports.insert(
        "potentials".to_string(),
        Type::Function(vec![Type::SnnNetwork], Box::new(Type::Unknown)),
    );
    snn_exports.insert(
        "synapses".to_string(),
        Type::Function(vec![Type::SnnNetwork], Box::new(Type::SparseMatrix)),
    );
    exports.entry(std_snn_id).or_insert(snn_exports);
    let std_agent_id = ModuleId(vec!["std".to_string(), "agent".to_string()]);
    imports
        .entry("agent".to_string())
        .or_insert(std_agent_id.clone());
    let mut agent_exports = std::collections::HashMap::new();
    agent_exports.insert(
        "make".to_string(),
        Type::Function(
            vec![Type::SimWorld, Type::SpatialIndex],
            Box::new(Type::AgentEnv),
        ),
    );
    agent_exports.insert(
        "register".to_string(),
        Type::Function(
            vec![
                Type::AgentEnv,
                Type::Int,
                Type::Unknown,
                Type::Unknown,
                Type::Float,
                Type::Float,
            ],
            Box::new(Type::Void),
        ),
    );
    agent_exports.insert(
        "state".to_string(),
        Type::Function(vec![Type::AgentEnv, Type::Int], Box::new(Type::Unknown)),
    );
    agent_exports.insert(
        "body".to_string(),
        Type::Function(vec![Type::AgentEnv, Type::Int], Box::new(Type::Unknown)),
    );
    agent_exports.insert(
        "memory".to_string(),
        Type::Function(vec![Type::AgentEnv, Type::Int], Box::new(Type::Unknown)),
    );
    agent_exports.insert(
        "set_body".to_string(),
        Type::Function(
            vec![Type::AgentEnv, Type::Int, Type::Unknown],
            Box::new(Type::Void),
        ),
    );
    agent_exports.insert(
        "set_memory".to_string(),
        Type::Function(
            vec![Type::AgentEnv, Type::Int, Type::Unknown],
            Box::new(Type::Void),
        ),
    );
    agent_exports.insert(
        "position".to_string(),
        Type::Function(vec![Type::AgentEnv, Type::Int], Box::new(Type::Unknown)),
    );
    agent_exports.insert(
        "set_position".to_string(),
        Type::Function(
            vec![Type::AgentEnv, Type::Int, Type::Float, Type::Float],
            Box::new(Type::Void),
        ),
    );
    agent_exports.insert(
        "neighbors".to_string(),
        Type::Function(
            vec![Type::AgentEnv, Type::Int, Type::Float],
            Box::new(Type::Unknown),
        ),
    );
    agent_exports.insert(
        "reward_add".to_string(),
        Type::Function(
            vec![Type::AgentEnv, Type::Int, Type::Float],
            Box::new(Type::Void),
        ),
    );
    agent_exports.insert(
        "reward_get".to_string(),
        Type::Function(vec![Type::AgentEnv, Type::Int], Box::new(Type::Float)),
    );
    agent_exports.insert(
        "reward_take".to_string(),
        Type::Function(vec![Type::AgentEnv, Type::Int], Box::new(Type::Float)),
    );
    agent_exports.insert(
        "sense_push".to_string(),
        Type::Function(
            vec![Type::AgentEnv, Type::Int, Type::Unknown],
            Box::new(Type::Void),
        ),
    );
    agent_exports.insert(
        "sense_take".to_string(),
        Type::Function(vec![Type::AgentEnv, Type::Int], Box::new(Type::Unknown)),
    );
    agent_exports.insert(
        "action_push".to_string(),
        Type::Function(
            vec![Type::AgentEnv, Type::Int, Type::Unknown],
            Box::new(Type::Void),
        ),
    );
    agent_exports.insert(
        "action_take".to_string(),
        Type::Function(vec![Type::AgentEnv, Type::Int], Box::new(Type::Unknown)),
    );
    agent_exports.insert(
        "stream".to_string(),
        Type::Function(
            vec![Type::AgentEnv, Type::Int, Type::String],
            Box::new(Type::RngStream),
        ),
    );
    agent_exports.insert(
        "next_float".to_string(),
        Type::Function(vec![Type::RngStream], Box::new(Type::Float)),
    );
    agent_exports.insert(
        "next_int".to_string(),
        Type::Function(vec![Type::RngStream, Type::Int], Box::new(Type::Int)),
    );
    exports.entry(std_agent_id).or_insert(agent_exports);
    let std_tensor_id = ModuleId(vec!["std".to_string(), "tensor".to_string()]);
    let mut tensor_exports = std::collections::HashMap::new();
    tensor_exports.insert(
        "device".to_string(),
        Type::Function(vec![Type::String], Box::new(Type::Device)),
    );
    tensor_exports.insert(
        "from_array".to_string(),
        Type::Function(
            vec![
                Type::Array(Box::new(Type::Float)),
                Type::Array(Box::new(Type::Int)),
            ],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "to_array".to_string(),
        Type::Function(
            vec![Type::Tensor],
            Box::new(Type::Array(Box::new(Type::Float))),
        ),
    );
    tensor_exports.insert(
        "rank".to_string(),
        Type::Function(vec![Type::Tensor], Box::new(Type::Int)),
    );
    tensor_exports.insert(
        "len".to_string(),
        Type::Function(vec![Type::Tensor], Box::new(Type::Int)),
    );
    tensor_exports.insert(
        "get_flat".to_string(),
        Type::Function(vec![Type::Tensor, Type::Int], Box::new(Type::Float)),
    );
    for name in ["randn", "zeros"] {
        tensor_exports.insert(
            name.to_string(),
            Type::Function(
                vec![Type::Unknown, Type::DType, Type::Device],
                Box::new(Type::Tensor),
            ),
        );
    }
    for name in ["matmul", "add", "sub", "mul", "div"] {
        tensor_exports.insert(
            name.to_string(),
            Type::Function(vec![Type::Tensor, Type::Tensor], Box::new(Type::Tensor)),
        );
    }
    tensor_exports.insert(
        "scale".to_string(),
        Type::Function(vec![Type::Tensor, Type::Float], Box::new(Type::Tensor)),
    );
    tensor_exports.insert(
        "broadcast_to".to_string(),
        Type::Function(vec![Type::Tensor, Type::Unknown], Box::new(Type::Tensor)),
    );
    tensor_exports.insert(
        "reshape".to_string(),
        Type::Function(vec![Type::Tensor, Type::Unknown], Box::new(Type::Tensor)),
    );
    tensor_exports.insert(
        "transpose".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Int, Type::Int],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "slice".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Int, Type::Int, Type::Int, Type::Int],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "concat".to_string(),
        Type::Function(
            vec![Type::Array(Box::new(Type::Tensor)), Type::Int],
            Box::new(Type::Tensor),
        ),
    );
    for name in ["sum", "mean"] {
        tensor_exports.insert(
            name.to_string(),
            Type::Function(
                vec![Type::Tensor, Type::Int, Type::Bool],
                Box::new(Type::Tensor),
            ),
        );
    }
    tensor_exports.insert(
        "softmax".to_string(),
        Type::Function(vec![Type::Tensor, Type::Int], Box::new(Type::Tensor)),
    );
    for name in ["relu", "sigmoid", "gelu", "exp", "log", "sqrt", "tanh"] {
        tensor_exports.insert(
            name.to_string(),
            Type::Function(vec![Type::Tensor], Box::new(Type::Tensor)),
        );
    }
    tensor_exports.insert(
        "dropout".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Float, Type::Bool],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "linear".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Tensor, Type::Tensor],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "layernorm".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Tensor, Type::Tensor, Type::Float],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "embedding".to_string(),
        Type::Function(vec![Type::Tensor, Type::Tensor], Box::new(Type::Tensor)),
    );
    tensor_exports.insert(
        "cross_entropy".to_string(),
        Type::Function(vec![Type::Tensor, Type::Tensor], Box::new(Type::Tensor)),
    );
    tensor_exports.insert(
        "to_dtype".to_string(),
        Type::Function(vec![Type::Tensor, Type::DType], Box::new(Type::Tensor)),
    );
    tensor_exports.insert(
        "to_device".to_string(),
        Type::Function(vec![Type::Tensor, Type::Device], Box::new(Type::Tensor)),
    );
    for name in ["require_grad", "requires_grad", "detach", "grad"] {
        tensor_exports.insert(
            name.to_string(),
            Type::Function(vec![Type::Tensor], Box::new(Type::Tensor)),
        );
    }
    tensor_exports.insert(
        "backward".to_string(),
        Type::Function(vec![Type::Tensor], Box::new(Type::Int)),
    );
    tensor_exports.insert(
        "zero_grad".to_string(),
        Type::Function(vec![Type::Tensor], Box::new(Type::Int)),
    );
    tensor_exports.insert(
        "sgd_step".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Tensor, Type::Float, Type::Float],
            Box::new(Type::Int),
        ),
    );
    tensor_exports.insert(
        "sgd_step_multi".to_string(),
        Type::Function(
            vec![
                Type::Array(Box::new(Type::Tensor)),
                Type::Array(Box::new(Type::Tensor)),
                Type::Float,
                Type::Float,
            ],
            Box::new(Type::Int),
        ),
    );
    tensor_exports.insert(
        "clip_grad_norm".to_string(),
        Type::Function(
            vec![
                Type::Array(Box::new(Type::Tensor)),
                Type::Float,
                Type::Float,
            ],
            Box::new(Type::Float),
        ),
    );
    tensor_exports.insert(
        "zero_grad_multi".to_string(),
        Type::Function(
            vec![Type::Array(Box::new(Type::Tensor))],
            Box::new(Type::Int),
        ),
    );
    for name in [
        "memory_current",
        "memory_peak",
        "memory_limit",
        "memory_clear_limit",
        "memory_reset_peak",
    ] {
        tensor_exports.insert(
            name.to_string(),
            Type::Function(Vec::new(), Box::new(Type::Int)),
        );
    }
    tensor_exports.insert(
        "memory_set_limit".to_string(),
        Type::Function(vec![Type::Int], Box::new(Type::Int)),
    );
    tensor_exports.insert(
        "adamw_state".to_string(),
        Type::Function(Vec::new(), Box::new(Type::OptimizerState)),
    );
    tensor_exports.insert(
        "adamw_step".to_string(),
        Type::Function(
            vec![
                Type::Tensor,
                Type::Tensor,
                Type::OptimizerState,
                Type::Float,
                Type::Float,
                Type::Float,
                Type::Float,
                Type::Float,
            ],
            Box::new(Type::OptimizerState),
        ),
    );
    tensor_exports.insert(
        "adamw_step_multi".to_string(),
        Type::Function(
            vec![
                Type::Array(Box::new(Type::Tensor)),
                Type::Array(Box::new(Type::Tensor)),
                Type::OptimizerState,
                Type::Float,
                Type::Float,
                Type::Float,
                Type::Float,
                Type::Float,
            ],
            Box::new(Type::OptimizerState),
        ),
    );
    tensor_exports.insert(
        "no_grad".to_string(),
        Type::Function(vec![Type::Bool], Box::new(Type::Bool)),
    );
    tensor_exports.insert(
        "grad_check".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Tensor, Type::Float],
            Box::new(Type::Bool),
        ),
    );
    tensor_exports.insert(
        "where".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Tensor, Type::Tensor],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "clip".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Float, Type::Float],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "argmax".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Int, Type::Bool],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "sort".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Int, Type::Bool],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "topk".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Int, Type::Int],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "gather".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Int, Type::Tensor],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "scatter".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Int, Type::Tensor, Type::Tensor],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "masked_fill".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Tensor, Type::Float],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "einsum".to_string(),
        Type::Function(
            vec![Type::String, Type::Tensor, Type::Tensor],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "conv2d".to_string(),
        Type::Function(
            vec![
                Type::Tensor,
                Type::Tensor,
                Type::Tensor,
                Type::Int,
                Type::Int,
            ],
            Box::new(Type::Tensor),
        ),
    );
    for name in ["max_pool2d", "avg_pool2d"] {
        tensor_exports.insert(
            name.to_string(),
            Type::Function(
                vec![Type::Tensor, Type::Int, Type::Int],
                Box::new(Type::Tensor),
            ),
        );
    }
    tensor_exports.insert(
        "batchnorm1d".to_string(),
        Type::Function(
            vec![
                Type::Tensor,
                Type::Tensor,
                Type::Tensor,
                Type::Float,
                Type::Bool,
            ],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "attention".to_string(),
        Type::Function(
            vec![Type::Tensor, Type::Tensor, Type::Tensor],
            Box::new(Type::Tensor),
        ),
    );
    tensor_exports.insert(
        "shape".to_string(),
        Type::Function(
            vec![Type::Tensor],
            Box::new(Type::Array(Box::new(Type::Int))),
        ),
    );
    exports.entry(std_tensor_id).or_insert(tensor_exports);
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
        Type::Int | Type::Float | Type::Bool | Type::String | Type::Buffer | Type::Handle => true,
        Type::Optional(inner) => {
            matches!(inner.as_ref(), Type::String | Type::Buffer | Type::Handle)
        }
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
                out.insert(decl.name.clone(), Type::Named(decl.name.clone()));
            }
        } else if let Item::Enum(decl) = item {
            if decl.is_pub {
                out.insert(decl.name.clone(), Type::Unknown);
            }
        }
    }
    out
}

fn is_std_json_module(id: &ModuleId) -> bool {
    id.0.len() == 2 && id.0[0] == "std" && id.0[1] == "json"
}
