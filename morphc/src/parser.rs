use std::fmt;

use crate::ast::*;
use crate::lexer::{tokenize, LexerError, Token, TokenKind};

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub col: usize,
    pub source_name: Option<String>,
    pub snippet: Option<String>,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.source_name {
            write!(f, "{}:{}:{}: {}", name, self.line, self.col, self.message)?;
        } else {
            write!(f, "{} at {}:{}", self.message, self.line, self.col)?;
        }
        if let Some(snippet) = &self.snippet {
            write!(f, "\n{}", snippet)?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

impl From<LexerError> for ParseError {
    fn from(err: LexerError) -> Self {
        Self {
            message: err.message,
            line: err.line,
            col: err.col,
            source_name: None,
            snippet: None,
        }
    }
}

impl ParseError {
    fn from_lexer(err: LexerError, source: &str, source_name: Option<&str>) -> Self {
        let snippet = format_snippet(source, err.line, err.col);
        Self {
            message: err.message,
            line: err.line,
            col: err.col,
            source_name: source_name.map(|name| name.to_string()),
            snippet,
        }
    }
}

pub fn parse_module(source: &str) -> Result<Module, ParseError> {
    parse_module_named(source, None)
}

pub fn parse_module_named(source: &str, source_name: Option<&str>) -> Result<Module, ParseError> {
    let tokens =
        tokenize(source).map_err(|err| ParseError::from_lexer(err, source, source_name))?;
    let mut parser = Parser::new(tokens, source, source_name);
    parser.parse_module()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    source_lines: Vec<String>,
    source_name: Option<String>,
}

impl Parser {
    fn new(tokens: Vec<Token>, source: &str, source_name: Option<&str>) -> Self {
        Self {
            tokens,
            pos: 0,
            source_lines: source.lines().map(|line| line.to_string()).collect(),
            source_name: source_name.map(|name| name.to_string()),
        }
    }

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        let mut items = Vec::new();
        self.consume_newlines();
        while !self.at_end() {
            if self.check(TokenKind::BlockEnd) {
                let token = self.current();
                return Err(self.error("Unexpected block end", token));
            }
            items.push(self.parse_item()?);
            self.consume_newlines();
        }
        Ok(Module { items })
    }

    fn parse_item(&mut self) -> Result<Item, ParseError> {
        if self.matches(TokenKind::Use) {
            return Ok(Item::Use(self.parse_use_decl()?));
        }
        if self.matches(TokenKind::Pub) {
            if !self.check(TokenKind::Fn) {
                let token = self.current();
                return Err(self.error("Expected 'fn' after 'pub'", token));
            }
            self.advance();
            return Ok(Item::Fn(self.parse_fn_decl(true)?));
        }
        if self.matches(TokenKind::Fn) {
            return Ok(Item::Fn(self.parse_fn_decl(false)?));
        }
        if self.matches(TokenKind::Type) {
            return Ok(Item::Type(self.parse_type_decl()?));
        }
        if self.matches(TokenKind::Enum) {
            return Ok(Item::Enum(self.parse_enum_decl()?));
        }
        if self.matches(TokenKind::Impl) {
            return Ok(Item::Impl(self.parse_impl_decl()?));
        }
        if self.matches(TokenKind::Tool) {
            return Ok(Item::Tool(self.parse_tool_decl()?));
        }
        if self.matches(TokenKind::Policy) {
            return Ok(Item::Policy(self.parse_policy_decl()?));
        }
        if self.matches(TokenKind::Prompt) {
            return Ok(Item::Prompt(self.parse_prompt_decl()?));
        }
        if self.matches(TokenKind::Model) {
            return Ok(Item::Model(self.parse_model_decl()?));
        }
        if self.matches(TokenKind::Agent) {
            return Ok(Item::Agent(self.parse_agent_decl()?));
        }
        if self.matches(TokenKind::Async) {
            let token = self.previous();
            return Err(self.error("Async functions are not supported in v0.1", token));
        }
        Ok(Item::Stmt(self.parse_stmt()?))
    }

    fn parse_use_decl(&mut self) -> Result<UseDecl, ParseError> {
        let path = self.parse_path()?;
        let alias = if self.matches(TokenKind::As) {
            Some(self.expect_ident()?)
        } else {
            None
        };
        self.consume_stmt_end()?;
        Ok(UseDecl { path, alias })
    }

    fn parse_fn_decl(&mut self, is_pub: bool) -> Result<FnDecl, ParseError> {
        if self.matches(TokenKind::Async) {
            let token = self.previous();
            return Err(self.error("Async functions are not supported in v0.1", token));
        }
        let name = self.expect_ident()?;
        let params = self.parse_param_list()?;
        let return_type = if self.matches(TokenKind::Arrow) {
            Some(self.parse_type_ref()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        Ok(FnDecl {
            name,
            params,
            return_type,
            body,
            is_pub,
        })
    }

    fn parse_type_decl(&mut self) -> Result<TypeDecl, ParseError> {
        let name = self.expect_ident()?;
        let mut fields = Vec::new();
        self.expect(TokenKind::BlockStart)?;
        self.consume_newlines();
        while !self.check(TokenKind::BlockEnd) {
            let field_name = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            let field_type = self.parse_type_ref()?;
            self.consume_stmt_end()?;
            fields.push(TypeField {
                name: field_name,
                field_type,
            });
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(TypeDecl { name, fields })
    }

    fn parse_enum_decl(&mut self) -> Result<EnumDecl, ParseError> {
        let name = self.expect_ident()?;
        let mut variants = Vec::new();
        self.expect(TokenKind::BlockStart)?;
        self.consume_newlines();
        while !self.check(TokenKind::BlockEnd) {
            let variant = self.expect_ident()?;
            variants.push(variant);
            self.consume_stmt_end()?;
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(EnumDecl { name, variants })
    }

    fn parse_impl_decl(&mut self) -> Result<ImplDecl, ParseError> {
        let name = self.expect_ident()?;
        let mut methods = Vec::new();
        self.expect(TokenKind::BlockStart)?;
        self.consume_newlines();
        while !self.check(TokenKind::BlockEnd) {
            if self.matches(TokenKind::Fn) {
                methods.push(self.parse_fn_decl(false)?);
            } else {
                let token = self.current();
                return Err(self.error("Only 'fn' declarations are allowed in impl blocks", token));
            }
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(ImplDecl { name, methods })
    }

    fn parse_tool_decl(&mut self) -> Result<ToolDecl, ParseError> {
        let path = self.parse_path()?;
        let params = self.parse_param_list()?;
        let return_type = if self.matches(TokenKind::Arrow) {
            Some(self.parse_type_ref()?)
        } else {
            None
        };
        self.consume_stmt_end()?;
        Ok(ToolDecl {
            path,
            params,
            return_type,
        })
    }

    fn parse_policy_decl(&mut self) -> Result<PolicyDecl, ParseError> {
        let name = self.expect_ident()?;
        self.expect(TokenKind::BlockStart)?;
        let mut rules = Vec::new();
        self.consume_newlines();
        while !self.check(TokenKind::BlockEnd) {
            rules.push(self.parse_policy_rule()?);
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(PolicyDecl { name, rules })
    }

    fn parse_policy_rule(&mut self) -> Result<PolicyRule, ParseError> {
        let allow = if self.matches(TokenKind::Allow) {
            true
        } else if self.matches(TokenKind::Deny) {
            false
        } else {
            let token = self.current();
            return Err(self.error("Expected 'allow' or 'deny' in policy rule", token));
        };
        let capability = self.parse_capability_path()?;
        let mut filters = Vec::new();
        if self.matches(TokenKind::Comma) {
            // no-op to allow leading commas
        }
        while self.check_ident() {
            let name = self.expect_ident()?;
            self.expect(TokenKind::Equal)?;
            let value = if self.matches(TokenKind::LBracket) {
                let mut values = Vec::new();
                if !self.check(TokenKind::RBracket) {
                    values.push(self.parse_literal()?);
                    while self.matches(TokenKind::Comma) {
                        values.push(self.parse_literal()?);
                    }
                }
                self.expect(TokenKind::RBracket)?;
                LiteralOrList::List(values)
            } else {
                LiteralOrList::Literal(self.parse_literal()?)
            };
            filters.push(PolicyFilter { name, value });
            if !self.matches(TokenKind::Comma) {
                break;
            }
        }
        self.consume_stmt_end()?;
        Ok(PolicyRule {
            allow,
            capability,
            filters,
        })
    }

    fn parse_prompt_decl(&mut self) -> Result<PromptDecl, ParseError> {
        let name = self.expect_ident()?;
        self.expect(TokenKind::BlockStart)?;
        let mut input_fields = Vec::new();
        let mut output_type = None;
        let mut template = None;
        self.consume_newlines();
        while !self.check(TokenKind::BlockEnd) {
            if self.peek_ident_value("input") {
                self.advance();
                self.expect(TokenKind::BlockStart)?;
                self.consume_newlines();
                while !self.check(TokenKind::BlockEnd) {
                    let field_name = self.expect_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let field_type = self.parse_type_ref()?;
                    self.consume_stmt_end()?;
                    input_fields.push(TypeField {
                        name: field_name,
                        field_type,
                    });
                    self.consume_newlines();
                }
                self.expect(TokenKind::BlockEnd)?;
            } else if self.peek_ident_value("output") {
                self.advance();
                self.expect(TokenKind::BlockStart)?;
                self.consume_newlines();
                if !self.check(TokenKind::BlockEnd) {
                    output_type = Some(self.parse_type_ref()?);
                    self.consume_stmt_end()?;
                }
                self.consume_newlines();
                self.expect(TokenKind::BlockEnd)?;
            } else if self.peek_ident_value("template") {
                self.advance();
                self.expect(TokenKind::BlockStart)?;
                self.consume_newlines();
                if self.check_string() {
                    if let TokenKind::String(text) = self.advance().kind {
                        template = Some(text);
                    }
                    self.consume_stmt_end()?;
                }
                self.consume_newlines();
                self.expect(TokenKind::BlockEnd)?;
            } else {
                let token = self.current();
                return Err(self.error("Unexpected token in prompt block", token));
            }
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(PromptDecl {
            name,
            input_fields,
            output_type,
            template,
        })
    }

    fn parse_model_decl(&mut self) -> Result<ModelDecl, ParseError> {
        let name = self.expect_ident()?;
        self.expect(TokenKind::Assign)?;
        let expr = self.parse_expr()?;
        self.consume_stmt_end()?;
        Ok(ModelDecl { name, expr })
    }

    fn parse_agent_decl(&mut self) -> Result<AgentDecl, ParseError> {
        let name = self.expect_ident()?;
        self.expect(TokenKind::BlockStart)?;
        let mut items = Vec::new();
        self.consume_newlines();
        while !self.check(TokenKind::BlockEnd) {
            if self.matches(TokenKind::Policy) {
                let policy_name = self.expect_ident()?;
                self.consume_stmt_end()?;
                items.push(AgentItem::PolicyUse(policy_name));
            } else if self.matches(TokenKind::Memory) {
                items.push(AgentItem::Memory(self.parse_memory_decl()?));
            } else if self.matches(TokenKind::Fn) {
                items.push(AgentItem::Fn(self.parse_fn_decl(false)?));
            } else {
                items.push(AgentItem::Stmt(self.parse_stmt()?));
            }
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(AgentDecl { name, items })
    }

    fn parse_memory_decl(&mut self) -> Result<MemoryDecl, ParseError> {
        let name = self.expect_ident()?;
        if self.matches(TokenKind::LParen) {
            let path = if let TokenKind::String(text) = self.advance().kind {
                text
            } else {
                let token = self.previous();
                return Err(self.error("Expected string literal for memory path", token));
            };
            self.expect(TokenKind::RParen)?;
            self.consume_stmt_end()?;
            Ok(MemoryDecl::Path { name, path })
        } else if self.matches(TokenKind::Assign) {
            let expr = self.parse_expr()?;
            self.consume_stmt_end()?;
            Ok(MemoryDecl::Expr { name, expr })
        } else {
            let token = self.current();
            Err(self.error("Expected memory path or ':=' assignment", token))
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        if self.matches(TokenKind::Let) {
            return self.parse_let_stmt();
        }
        if self.matches(TokenKind::If) {
            return self.parse_if_stmt();
        }
        if self.matches(TokenKind::While) {
            return self.parse_while_stmt();
        }
        if self.matches(TokenKind::For) {
            return self.parse_for_stmt();
        }
        if self.matches(TokenKind::Match) {
            return self.parse_match_stmt();
        }
        if self.matches(TokenKind::Try) {
            return self.parse_try_stmt();
        }
        if self.matches(TokenKind::Return) {
            return self.parse_return_stmt();
        }
        if self.matches(TokenKind::Break) {
            self.consume_stmt_end()?;
            return Ok(Stmt::Break);
        }
        if self.matches(TokenKind::Continue) {
            self.consume_stmt_end()?;
            return Ok(Stmt::Continue);
        }
        self.parse_assign_or_expr_stmt()
    }

    fn parse_let_stmt(&mut self) -> Result<Stmt, ParseError> {
        let name = self.expect_ident()?;
        let type_ann = if self.matches(TokenKind::Colon) {
            Some(self.parse_type_ref()?)
        } else {
            None
        };
        if self.matches(TokenKind::Equal) {
            let token = self.previous();
            return Err(self.error("Use ':=' for bindings", token));
        }
        self.expect(TokenKind::Assign)?;
        let expr = self.parse_expr()?;
        self.consume_stmt_end()?;
        Ok(Stmt::Let {
            name,
            type_ann,
            expr,
        })
    }

    fn parse_if_stmt(&mut self) -> Result<Stmt, ParseError> {
        let cond = self.parse_expr()?;
        let then_block = self.parse_block()?;
        self.consume_newlines();
        let else_branch = if self.matches(TokenKind::Else) {
            if self.matches(TokenKind::If) {
                Some(ElseBranch::If(Box::new(self.parse_if_stmt()?)))
            } else {
                Some(ElseBranch::Block(self.parse_block()?))
            }
        } else {
            None
        };
        Ok(Stmt::If {
            cond,
            then_block,
            else_branch,
        })
    }

    fn parse_while_stmt(&mut self) -> Result<Stmt, ParseError> {
        let cond = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body })
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, ParseError> {
        let var = self.expect_ident()?;
        self.expect(TokenKind::In)?;
        let iter = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::For { var, iter, body })
    }

    fn parse_match_stmt(&mut self) -> Result<Stmt, ParseError> {
        let expr = self.parse_expr()?;
        let arms = self.parse_match_arms(true)?;
        Ok(Stmt::Match { expr, arms })
    }

    fn parse_try_stmt(&mut self) -> Result<Stmt, ParseError> {
        let body = self.parse_block()?;
        self.consume_newlines();
        self.expect(TokenKind::Catch)?;
        let catch_name = self.expect_ident()?;
        let catch_body = self.parse_block()?;
        Ok(Stmt::Try {
            body,
            catch_name,
            catch_body,
        })
    }

    fn parse_return_stmt(&mut self) -> Result<Stmt, ParseError> {
        if self.is_stmt_end() {
            self.consume_stmt_end()?;
            return Ok(Stmt::Return { expr: None });
        }
        let expr = self.parse_expr()?;
        self.consume_stmt_end()?;
        Ok(Stmt::Return { expr: Some(expr) })
    }

    fn parse_assign_or_expr_stmt(&mut self) -> Result<Stmt, ParseError> {
        if self.check_ident() {
            let mark = self.pos;
            if let Ok(lvalue) = self.parse_lvalue() {
                if self.matches(TokenKind::Assign) {
                    let expr = self.parse_expr()?;
                    self.consume_stmt_end()?;
                    return Ok(Stmt::Assign {
                        target: lvalue,
                        expr,
                    });
                }
            }
            self.pos = mark;
        }
        let expr = self.parse_expr()?;
        self.consume_stmt_end()?;
        Ok(Stmt::Expr(expr))
    }

    fn parse_lvalue(&mut self) -> Result<LValue, ParseError> {
        let base = self.expect_ident()?;
        let mut accesses = Vec::new();
        loop {
            if self.matches(TokenKind::Dot) {
                let name = self.expect_ident()?;
                accesses.push(LValueAccess::Field(name));
            } else if self.matches(TokenKind::LBracket) {
                let index = self.parse_expr()?;
                self.expect(TokenKind::RBracket)?;
                accesses.push(LValueAccess::Index(index));
            } else {
                break;
            }
        }
        Ok(LValue { base, accesses })
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        self.expect(TokenKind::BlockStart)?;
        let mut stmts = Vec::new();
        self.consume_newlines();
        while !self.check(TokenKind::BlockEnd) {
            stmts.push(self.parse_stmt()?);
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(Block { stmts })
    }

    fn parse_match_arms(&mut self, allow_block_body: bool) -> Result<Vec<MatchArm>, ParseError> {
        self.expect(TokenKind::BlockStart)?;
        let mut arms = Vec::new();
        self.consume_newlines();
        while !self.check(TokenKind::BlockEnd) {
            let pattern = self.parse_pattern()?;
            self.expect(TokenKind::FatArrow)?;
            let body = if self.check(TokenKind::BlockStart) {
                if !allow_block_body {
                    let token = self.current();
                    return Err(
                        self.error("Block arms are not allowed in match expressions", token)
                    );
                }
                ArmBody::Block(self.parse_block()?)
            } else {
                let expr = self.parse_expr()?;
                self.consume_stmt_end()?;
                ArmBody::Expr(expr)
            };
            arms.push(MatchArm { pattern, body });
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(arms)
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        if self.check_ident() {
            let name = self.expect_ident()?;
            if name == "_" {
                return Ok(Pattern::Wildcard);
            }
            return Ok(Pattern::Ident(name));
        }
        Ok(Pattern::Literal(self.parse_literal()?))
    }

    fn parse_literal(&mut self) -> Result<Literal, ParseError> {
        let token = self.advance();
        match token.kind {
            TokenKind::Int(value) => Ok(Literal::Int(value)),
            TokenKind::Float(value) => Ok(Literal::Float(value)),
            TokenKind::String(value) => Ok(Literal::String(value)),
            TokenKind::True => Ok(Literal::Bool(true)),
            TokenKind::False => Ok(Literal::Bool(false)),
            TokenKind::None => Ok(Literal::None),
            _ => Err(self.error("Expected literal", &token)),
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_and()?;
        while self.matches(TokenKind::Or) {
            let right = self.parse_and()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::Or,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_equality()?;
        while self.matches(TokenKind::And) {
            let right = self.parse_equality()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::And,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_comparison()?;
        loop {
            if self.matches(TokenKind::EqualEqual) {
                let right = self.parse_comparison()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Equal,
                    right: Box::new(right),
                };
            } else if self.matches(TokenKind::BangEqual) {
                let right = self.parse_comparison()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::NotEqual,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_additive()?;
        loop {
            let op = if self.matches(TokenKind::Less) {
                Some(BinaryOp::Less)
            } else if self.matches(TokenKind::LessEqual) {
                Some(BinaryOp::LessEqual)
            } else if self.matches(TokenKind::Greater) {
                Some(BinaryOp::Greater)
            } else if self.matches(TokenKind::GreaterEqual) {
                Some(BinaryOp::GreaterEqual)
            } else {
                None
            };
            if let Some(op) = op {
                let right = self.parse_additive()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_multiplicative()?;
        loop {
            if self.matches(TokenKind::Plus) {
                let right = self.parse_multiplicative()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Add,
                    right: Box::new(right),
                };
            } else if self.matches(TokenKind::Minus) {
                let right = self.parse_multiplicative()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Subtract,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_unary()?;
        loop {
            if self.matches(TokenKind::Star) {
                let right = self.parse_unary()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Multiply,
                    right: Box::new(right),
                };
            } else if self.matches(TokenKind::Slash) {
                let right = self.parse_unary()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Divide,
                    right: Box::new(right),
                };
            } else if self.matches(TokenKind::Percent) {
                let right = self.parse_unary()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Modulo,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if self.matches(TokenKind::Not) {
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(expr),
            });
        }
        if self.matches(TokenKind::Minus) {
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Negate,
                expr: Box::new(expr),
            });
        }
        if self.matches(TokenKind::Await) {
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Await,
                expr: Box::new(expr),
            });
        }
        if self.matches(TokenKind::Spawn) {
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Spawn,
                expr: Box::new(expr),
            });
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.matches(TokenKind::LParen) {
                let args = self.parse_arg_list()?;
                expr = Expr::Call {
                    callee: Box::new(expr),
                    args,
                };
            } else if self.matches(TokenKind::LBracket) {
                let index = self.parse_expr()?;
                self.expect(TokenKind::RBracket)?;
                expr = Expr::Index {
                    target: Box::new(expr),
                    index: Box::new(index),
                };
            } else if self.matches(TokenKind::Dot) {
                let name = self.expect_ident()?;
                expr = Expr::Field {
                    target: Box::new(expr),
                    name,
                };
            } else if self.matches(TokenKind::Question) {
                expr = Expr::Try(Box::new(expr));
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        if self.check(TokenKind::Equal) {
            let token = self.current();
            return Err(self.error(
                "Unexpected '='; use ':=' for assignment or 'name = value' for named args",
                token,
            ));
        }
        if self.matches(TokenKind::True) {
            return Ok(Expr::Literal(Literal::Bool(true)));
        }
        if self.matches(TokenKind::False) {
            return Ok(Expr::Literal(Literal::Bool(false)));
        }
        if self.matches(TokenKind::None) {
            return Ok(Expr::Literal(Literal::None));
        }
        if self.check_literal_start() {
            let literal = self.parse_literal()?;
            return Ok(Expr::Literal(literal));
        }
        if self.matches(TokenKind::Match) {
            let expr = self.parse_expr()?;
            let arms = self.parse_match_arms(false)?;
            return Ok(Expr::Match {
                expr: Box::new(expr),
                arms,
            });
        }
        if self.check_ident() {
            let name = self.expect_ident()?;
            return Ok(Expr::Ident(name));
        }
        if self.matches(TokenKind::LBracket) {
            let mut values = Vec::new();
            if !self.check(TokenKind::RBracket) {
                values.push(self.parse_expr()?);
                while self.matches(TokenKind::Comma) {
                    values.push(self.parse_expr()?);
                }
            }
            self.expect(TokenKind::RBracket)?;
            return Ok(Expr::List(values));
        }
        if self.check(TokenKind::LParen) {
            if self.looks_like_lambda()? {
                return self.parse_lambda();
            }
            self.expect(TokenKind::LParen)?;
            let expr = self.parse_expr()?;
            self.expect(TokenKind::RParen)?;
            return Ok(expr);
        }
        let token = self.current();
        Err(self.error("Expected expression", token))
    }

    fn parse_lambda(&mut self) -> Result<Expr, ParseError> {
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        if !self.check(TokenKind::RParen) {
            params.push(self.parse_param()?);
            while self.matches(TokenKind::Comma) {
                params.push(self.parse_param()?);
            }
        }
        self.expect(TokenKind::RParen)?;
        let return_type = if self.matches(TokenKind::Arrow) {
            Some(self.parse_type_ref()?)
        } else {
            None
        };
        self.expect(TokenKind::FatArrow)?;
        let body = self.parse_expr()?;
        Ok(Expr::Lambda {
            params,
            return_type,
            body: Box::new(body),
        })
    }

    fn parse_arg_list(&mut self) -> Result<Vec<Arg>, ParseError> {
        let mut args = Vec::new();
        self.consume_newlines();
        if self.check(TokenKind::RParen) {
            self.expect(TokenKind::RParen)?;
            return Ok(args);
        }
        loop {
            if self.check_ident() && self.peek_next_is(TokenKind::Equal) {
                let name = self.expect_ident()?;
                self.expect(TokenKind::Equal)?;
                let value = self.parse_expr()?;
                args.push(Arg::Named(name, value));
            } else {
                let value = self.parse_expr()?;
                args.push(Arg::Positional(value));
            }
            self.consume_newlines();
            if self.matches(TokenKind::Comma) {
                self.consume_newlines();
                continue;
            }
            self.expect(TokenKind::RParen)?;
            break;
        }
        Ok(args)
    }

    fn parse_param_list(&mut self) -> Result<Vec<Param>, ParseError> {
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        if !self.check(TokenKind::RParen) {
            params.push(self.parse_param()?);
            while self.matches(TokenKind::Comma) {
                params.push(self.parse_param()?);
            }
        }
        self.expect(TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let name = self.expect_ident()?;
        let type_ann = if self.matches(TokenKind::Colon) {
            Some(self.parse_type_ref()?)
        } else {
            None
        };
        let default = if self.matches(TokenKind::Equal) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(Param {
            name,
            type_ann,
            default,
        })
    }

    fn parse_type_ref(&mut self) -> Result<TypeRef, ParseError> {
        if self.matches(TokenKind::Fn) {
            self.expect(TokenKind::LParen)?;
            let mut params = Vec::new();
            if !self.check(TokenKind::RParen) {
                params.push(self.parse_type_ref()?);
                while self.matches(TokenKind::Comma) {
                    params.push(self.parse_type_ref()?);
                }
            }
            self.expect(TokenKind::RParen)?;
            self.expect(TokenKind::Arrow)?;
            let ret = self.parse_type_ref()?;
            return Ok(TypeRef::Function {
                params,
                ret: Box::new(ret),
            });
        }

        let mut path = Vec::new();
        path.push(self.expect_ident()?);
        while self.matches(TokenKind::Dot) {
            path.push(self.expect_ident()?);
        }

        let mut args = Vec::new();
        if self.matches(TokenKind::Less) {
            args.push(self.parse_type_ref()?);
            while self.matches(TokenKind::Comma) {
                args.push(self.parse_type_ref()?);
            }
            self.expect(TokenKind::Greater)?;
        }

        let optional = self.matches(TokenKind::Question);
        Ok(TypeRef::Named {
            path,
            args,
            optional,
        })
    }

    fn parse_path(&mut self) -> Result<Vec<String>, ParseError> {
        let mut path = Vec::new();
        path.push(self.expect_ident()?);
        while self.matches(TokenKind::Dot) {
            path.push(self.expect_ident()?);
        }
        Ok(path)
    }

    fn parse_capability_path(&mut self) -> Result<Vec<String>, ParseError> {
        let mut path = Vec::new();
        path.push(self.expect_capability_segment()?);
        while self.matches(TokenKind::Dot) {
            path.push(self.expect_capability_segment()?);
        }
        Ok(path)
    }

    fn expect_capability_segment(&mut self) -> Result<String, ParseError> {
        if self.check_ident() {
            return self.expect_ident();
        }
        if self.matches(TokenKind::Tool) {
            return Ok("tool".to_string());
        }
        if self.matches(TokenKind::Model) {
            return Ok("model".to_string());
        }
        if self.matches(TokenKind::Memory) {
            return Ok("memory".to_string());
        }
        let token = self.current();
        Err(self.error("Expected identifier", token))
    }

    fn consume_stmt_end(&mut self) -> Result<(), ParseError> {
        if self.matches(TokenKind::Semicolon) {
            self.consume_newlines();
            return Ok(());
        }
        if self.matches(TokenKind::Newline) {
            self.consume_newlines();
            return Ok(());
        }
        if self.check(TokenKind::Eof) || self.check(TokenKind::BlockEnd) {
            return Ok(());
        }
        let token = self.current();
        Err(self.error("Expected statement terminator", token))
    }

    fn consume_newlines(&mut self) {
        while self.matches(TokenKind::Newline) {}
    }

    fn is_stmt_end(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::Newline | TokenKind::Semicolon | TokenKind::Eof | TokenKind::BlockEnd
        )
    }

    fn looks_like_lambda(&self) -> Result<bool, ParseError> {
        if !self.check(TokenKind::LParen) {
            return Ok(false);
        }
        let mut depth = 0;
        let mut idx = self.pos;
        while idx < self.tokens.len() {
            let kind = &self.tokens[idx].kind;
            match kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        let next = self.tokens.get(idx + 1).map(|t| &t.kind);
                        let next_next = self.tokens.get(idx + 2).map(|t| &t.kind);
                        if matches!(next, Some(TokenKind::FatArrow)) {
                            return Ok(true);
                        }
                        if matches!(next, Some(TokenKind::Arrow))
                            && matches!(next_next, Some(TokenKind::FatArrow))
                        {
                            return Ok(true);
                        }
                        return Ok(false);
                    }
                }
                TokenKind::Eof => break,
                _ => {}
            }
            idx += 1;
        }
        Err(self.error("Unterminated parameter list", self.current()))
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    fn check_ident(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Ident(_))
    }

    fn check_string(&self) -> bool {
        matches!(self.peek().kind, TokenKind::String(_))
    }

    fn check_literal_start(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::Int(_) | TokenKind::Float(_) | TokenKind::String(_)
        )
    }

    fn peek_ident_value(&self, value: &str) -> bool {
        matches!(self.peek().kind, TokenKind::Ident(ref name) if name == value)
    }

    fn peek_next_is(&self, kind: TokenKind) -> bool {
        self.tokens
            .get(self.pos + 1)
            .map(|t| t.kind == kind)
            .unwrap_or(false)
    }

    fn matches(&mut self, kind: TokenKind) -> bool {
        if self.check(kind.clone()) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, ParseError> {
        if self.check(kind.clone()) {
            Ok(self.advance())
        } else {
            let token = self.current();
            Err(self.error(&format!("Expected {:?}", kind), token))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        let token = self.advance();
        if let TokenKind::Ident(name) = token.kind {
            Ok(name)
        } else {
            Err(self.error("Expected identifier", &token))
        }
    }

    fn current(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .unwrap_or_else(|| self.tokens.last().unwrap())
    }

    fn peek(&self) -> &Token {
        self.current()
    }

    fn previous(&self) -> &Token {
        if self.pos == 0 {
            self.current()
        } else {
            &self.tokens[self.pos - 1]
        }
    }

    fn advance(&mut self) -> Token {
        let token = self.current().clone();
        if !self.at_end() {
            self.pos += 1;
        }
        token
    }

    fn at_end(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    fn error(&self, message: &str, token: &Token) -> ParseError {
        ParseError {
            message: message.to_string(),
            line: token.line,
            col: token.col,
            source_name: self.source_name.clone(),
            snippet: format_snippet_from_lines(&self.source_lines, token.line, token.col),
        }
    }
}

fn format_snippet(source: &str, line: usize, col: usize) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    format_snippet_from_lines(
        &lines
            .iter()
            .map(|line| (*line).to_string())
            .collect::<Vec<_>>(),
        line,
        col,
    )
}

fn format_snippet_from_lines(lines: &[String], line: usize, col: usize) -> Option<String> {
    let line_text = lines.get(line.saturating_sub(1))?;
    let mut caret = String::new();
    let caret_pos = col.saturating_sub(1);
    for _ in 0..caret_pos {
        caret.push(' ');
    }
    caret.push('^');
    Some(format!("{}\n{}", line_text, caret))
}
