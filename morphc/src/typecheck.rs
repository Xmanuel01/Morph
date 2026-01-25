use crate::ast::*;
use crate::diagnostic::Span;
use crate::types::Type;

#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
}

pub struct TypeChecker {
    scopes: Vec<std::collections::HashMap<String, Type>>,
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self {
            scopes: vec![std::collections::HashMap::new()],
        }
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            scopes: vec![std::collections::HashMap::new()],
        }
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

    pub fn check_module(&mut self, module: &Module) -> Result<(), TypeError> {
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
            }
        }
        for item in &module.items {
            self.check_item(item)?;
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
                        return Err(TypeError {
                            message: format!(
                                "Type mismatch: expected {}, found {}",
                                ann_t.display(),
                                t.display()
                            ),
                            span: name_span.clone(),
                        });
                    }
                }
                self.define(name, t);
            }
            Stmt::Assign { target, expr } => {
                let t_expr = self.check_expr(expr)?;
                let var_t = self.resolve(&target.base).ok_or(TypeError {
                    message: "Undefined variable".into(),
                    span: target.base_span.clone(),
                })?;
                if var_t != Type::Unknown && !compatible(&var_t, &t_expr) {
                    return Err(TypeError {
                        message: format!(
                            "Type mismatch: variable {} is {}, assigned {}",
                            target.base,
                            var_t.display(),
                            t_expr.display()
                        ),
                        span: target.base_span.clone(),
                    });
                }
            }
            Stmt::If {
                cond,
                then_block,
                else_branch,
            } => {
                let t_cond = self.check_expr(cond)?;
                if t_cond != Type::Bool && t_cond != Type::Unknown {
                    return Err(TypeError {
                        message: "Condition must be Bool".into(),
                        span: line_span(cond),
                    });
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
                    return Err(TypeError {
                        message: "Condition must be Bool".into(),
                        span: line_span(cond),
                    });
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
                    return Err(TypeError {
                        message: format!(
                            "Return type mismatch: expected {}, found {}",
                            expected_return.display(),
                            t_val.display()
                        ),
                        span: expr
                            .as_ref()
                            .map(line_span)
                            .unwrap_or_else(|| Span::single(0, 0)),
                    });
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
            Expr::Ident { name, span } => self.resolve(name).ok_or(TypeError {
                message: format!("Undefined variable {}", name),
                span: span.clone(),
            }),
            Expr::Unary { op, expr, span } => {
                let t = self.check_expr(expr)?;
                match op {
                    UnaryOp::Negate => match t {
                        Type::Int | Type::Float | Type::Unknown => Ok(t),
                        _ => Err(TypeError {
                            message: "Unary '-' expects number".into(),
                            span: span.clone(),
                        }),
                    },
                    UnaryOp::Not => match t {
                        Type::Bool | Type::Unknown => Ok(Type::Bool),
                        _ => Err(TypeError {
                            message: "Unary 'not' expects Bool".into(),
                            span: span.clone(),
                        }),
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
                            Err(TypeError {
                                message: "Arithmetic expects numbers".into(),
                                span: span.clone(),
                            })
                        }
                    }
                    And | Or => {
                        if lt == Type::Bool && rt == Type::Bool {
                            Ok(Type::Bool)
                        } else {
                            Err(TypeError {
                                message: "Logical ops expect Bool".into(),
                                span: span.clone(),
                            })
                        }
                    }
                    Equal | NotEqual | Less | LessEqual | Greater | GreaterEqual => Ok(Type::Bool),
                }
            }
            Expr::Call { callee, args, span } => {
                let ct = self.check_expr(callee)?;
                if let Type::Function(params, ret) = ct {
                    if params.len() != args.len() {
                        return Err(TypeError {
                            message: format!(
                                "Arity mismatch: expected {}, got {}",
                                params.len(),
                                args.len()
                            ),
                            span: span.clone(),
                        });
                    }
                    for (arg, pt) in args.iter().zip(params.iter()) {
                        let at = self.check_expr(match arg {
                            Arg::Positional(e) => e,
                            Arg::Named(_, e) => e,
                        })?;
                        if *pt != Type::Unknown && !compatible(pt, &at) {
                            return Err(TypeError {
                                message: format!(
                                    "Argument type mismatch: expected {}, found {}",
                                    pt.display(),
                                    at.display()
                                ),
                                span: line_span(match arg {
                                    Arg::Positional(e) => e,
                                    Arg::Named(_, e) => e,
                                }),
                            });
                        }
                    }
                    Ok(*ret)
                } else {
                    Err(TypeError {
                        message: "Callee is not a function".into(),
                        span: span.clone(),
                    })
                }
            }
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
        _ => false,
    }
}
