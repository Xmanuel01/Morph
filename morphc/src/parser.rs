use std::fmt;

use crate::ast::*;
use crate::diagnostic::{Diagnostic, Span};
use crate::lexer::{tokenize, LexerError, Token, TokenKind};

#[derive(Debug, Clone)]
pub struct ParseError {
    pub diagnostic: Diagnostic,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.diagnostic)
    }
}

impl std::error::Error for ParseError {}

impl From<LexerError> for ParseError {
    fn from(err: LexerError) -> Self {
        let diagnostic =
            Diagnostic::new(&err.message, None).with_span("here", Span::single(err.line, err.col));
        Self {
            diagnostic,
            line: err.line,
            col: err.col,
        }
    }
}

impl ParseError {
    fn from_lexer(err: LexerError, source: &str, source_name: Option<&str>) -> Self {
        let diagnostic = Diagnostic::new(&err.message, source_name)
            .with_span("here", Span::single(err.line, err.col))
            .with_source(source);
        Self {
            diagnostic,
            line: err.line,
            col: err.col,
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
    source: String,
    source_name: Option<String>,
}

impl Parser {
    fn new(tokens: Vec<Token>, source: &str, source_name: Option<&str>) -> Self {
        Self {
            tokens,
            pos: 0,
            source: source.to_string(),
            source_name: source_name.map(|name| name.to_string()),
        }
    }

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        let mut items = Vec::new();
        self.consume_newlines();
        // imports must appear at top of file
        while self.check(TokenKind::Import) || self.is_native_import_start() {
            if self.matches(TokenKind::Import) {
                items.push(Item::Import(self.parse_import_decl()?));
            } else if self.is_native_import_start() {
                items.push(Item::NativeImport(self.parse_native_import_decl()?));
            }
            self.consume_newlines();
        }
        while !self.at_end() {
            if self.check(TokenKind::Import) || self.is_native_import_start() {
                let token = self.current();
                return Err(self.error("Imports must appear before other items", token));
            }
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
        if self.matches(TokenKind::Pub) {
            return self.parse_pub_item();
        }
        if self.matches(TokenKind::Use) {
            return Ok(Item::Use(self.parse_use_decl(false)?));
        }
        if self.matches(TokenKind::Import) {
            return Ok(Item::Import(self.parse_import_decl()?));
        }
        if self.is_native_import_start() {
            return Ok(Item::NativeImport(self.parse_native_import_decl()?));
        }
        if self.matches(TokenKind::Fn) {
            return Ok(Item::Fn(self.parse_fn_decl(false)?));
        }
        if self.matches(TokenKind::Type) {
            return Ok(Item::Type(self.parse_type_decl(false)?));
        }
        if self.matches(TokenKind::Enum) {
            return Ok(Item::Enum(self.parse_enum_decl(false)?));
        }
        if self.matches(TokenKind::Impl) {
            return Ok(Item::Impl(self.parse_impl_decl()?));
        }
        if self.matches(TokenKind::Tool) {
            return Ok(Item::Tool(self.parse_tool_decl(false)?));
        }
        if self.matches(TokenKind::Policy) {
            return Ok(Item::Policy(self.parse_policy_decl(false)?));
        }
        if self.matches(TokenKind::Prompt) {
            return Ok(Item::Prompt(self.parse_prompt_decl(false)?));
        }
        if self.matches(TokenKind::Model) {
            return Ok(Item::Model(self.parse_model_decl(false)?));
        }
        if self.matches(TokenKind::Agent) {
            return Ok(Item::Agent(self.parse_agent_decl(false)?));
        }
        if self.matches(TokenKind::Async) {
            let token = self.previous();
            return Err(self.error("Async functions are not supported yet", token));
        }
        Ok(Item::Stmt(self.parse_stmt()?))
    }

    fn parse_pub_item(&mut self) -> Result<Item, ParseError> {
        if self.matches(TokenKind::Use) {
            return Ok(Item::Use(self.parse_use_decl(true)?));
        }
        if self.matches(TokenKind::Fn) {
            return Ok(Item::Fn(self.parse_fn_decl(true)?));
        }
        if self.matches(TokenKind::Type) {
            return Ok(Item::Type(self.parse_type_decl(true)?));
        }
        if self.matches(TokenKind::Enum) {
            return Ok(Item::Enum(self.parse_enum_decl(true)?));
        }
        if self.matches(TokenKind::Async) {
            let token = self.previous();
            return Err(self.error("Async functions are not supported yet", token));
        }
        let token = self.current();
        Err(self.error("Only fn, type, enum, and use can be public", token))
    }

    fn parse_use_decl(&mut self, is_pub: bool) -> Result<UseDecl, ParseError> {
        let (path, spans) = self.parse_path_with_spans()?;
        let mut symbols = Vec::new();
        if self.matches(TokenKind::ColonColon) {
            self.expect(TokenKind::LBrace)?;
            self.consume_newlines();
            if self.check(TokenKind::RBrace) {
                let token = self.current();
                return Err(self.error("Expected symbol in use list", token));
            }
            loop {
                let (name, span) = self.expect_ident_with_span()?;
                symbols.push(UseSymbol { name, span });
                self.consume_newlines();
                if self.matches(TokenKind::Comma) {
                    self.consume_newlines();
                    if self.check(TokenKind::RBrace) {
                        break;
                    }
                    continue;
                }
                break;
            }
            self.expect(TokenKind::RBrace)?;
        }
        let alias = if self.matches(TokenKind::As) {
            Some(self.expect_ident()?)
        } else {
            None
        };
        if !symbols.is_empty() && alias.is_some() {
            let token = self.previous();
            return Err(self.error("Alias is not supported for use lists", token));
        }
        self.consume_stmt_end()?;
        Ok(UseDecl {
            path,
            alias,
            is_pub,
            spans,
            symbols,
        })
    }

    fn parse_import_decl(&mut self) -> Result<ImportDecl, ParseError> {
        let (path, spans) = self.parse_import_path_with_spans()?;
        let alias = if self.matches(TokenKind::As) {
            Some(self.expect_ident()?)
        } else {
            None
        };
        self.consume_stmt_end()?;
        Ok(ImportDecl { path, alias, spans })
    }

    fn parse_native_import_decl(&mut self) -> Result<NativeImportDecl, ParseError> {
        let (name, _name_span) = self.expect_ident_with_span()?;
        if name != "native" {
            return Err(self.error("Expected native::import", self.previous()));
        }
        if !self.matches(TokenKind::ColonColon) {
            return Err(self.error("Expected native::import", self.current()));
        }
        if !self.matches(TokenKind::Import) {
            return Err(self.error("Expected native::import", self.current()));
        }
        let (library, library_span) = self.expect_string_with_span()?;
        self.expect(TokenKind::BlockStart)?;
        self.consume_newlines();
        let mut functions = Vec::new();
        while !self.check(TokenKind::BlockEnd) && !self.at_end() {
            if !self.matches(TokenKind::Fn) {
                let token = self.current();
                return Err(self.error("Expected fn in native import block", token));
            }
            functions.push(self.parse_native_fn_decl()?);
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(NativeImportDecl {
            library,
            library_span,
            functions,
        })
    }

    fn parse_native_fn_decl(&mut self) -> Result<NativeFnDecl, ParseError> {
        let (name, name_span) = self.expect_ident_with_span()?;
        let params = self.parse_native_param_list()?;
        self.expect(TokenKind::Arrow)?;
        let return_type = self.parse_type_ref()?;
        Ok(NativeFnDecl {
            name,
            name_span,
            params,
            return_type,
        })
    }

    fn parse_native_param_list(&mut self) -> Result<Vec<NativeParam>, ParseError> {
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        if !self.check(TokenKind::RParen) {
            params.push(self.parse_native_param()?);
            while self.matches(TokenKind::Comma) {
                params.push(self.parse_native_param()?);
            }
        }
        self.expect(TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_native_param(&mut self) -> Result<NativeParam, ParseError> {
        let (name, name_span) = self.expect_ident_with_span()?;
        if !self.matches(TokenKind::Colon) {
            return Err(self.error("Expected type annotation", self.current()));
        }
        let type_ann = self.parse_type_ref()?;
        if self.matches(TokenKind::Equal) {
            let token = self.previous();
            return Err(self.error("Default values are not allowed in native signatures", token));
        }
        Ok(NativeParam {
            name,
            name_span,
            type_ann,
        })
    }

    fn parse_fn_decl(&mut self, is_pub: bool) -> Result<FnDecl, ParseError> {
        if self.matches(TokenKind::Async) {
            let token = self.previous();
            return Err(self.error("Async functions are not supported yet", token));
        }
        let (name, name_span) = self.expect_ident_with_span()?;
        let params = self.parse_param_list()?;
        let return_type = if self.matches(TokenKind::Arrow) {
            Some(self.parse_type_ref()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        Ok(FnDecl {
            name,
            name_span,
            params,
            return_type,
            body,
            is_pub,
        })
    }

    fn parse_type_decl(&mut self, is_pub: bool) -> Result<TypeDecl, ParseError> {
        let (name, _name_span) = self.expect_ident_with_span()?;
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
        Ok(TypeDecl {
            name,
            fields,
            is_pub,
        })
    }

    fn parse_enum_decl(&mut self, is_pub: bool) -> Result<EnumDecl, ParseError> {
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
        Ok(EnumDecl {
            name,
            variants,
            is_pub,
        })
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

    fn parse_tool_decl(&mut self, is_pub: bool) -> Result<ToolDecl, ParseError> {
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
            is_pub,
        })
    }

    fn parse_policy_decl(&mut self, is_pub: bool) -> Result<PolicyDecl, ParseError> {
        let name = self.expect_ident()?;
        self.expect(TokenKind::BlockStart)?;
        let mut rules = Vec::new();
        self.consume_newlines();
        while !self.check(TokenKind::BlockEnd) {
            rules.push(self.parse_policy_rule()?);
            self.consume_newlines();
        }
        self.expect(TokenKind::BlockEnd)?;
        Ok(PolicyDecl {
            name,
            rules,
            is_pub,
        })
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
                    let (literal, _) = self.parse_literal()?;
                    values.push(literal);
                    while self.matches(TokenKind::Comma) {
                        let (literal, _) = self.parse_literal()?;
                        values.push(literal);
                    }
                }
                self.expect(TokenKind::RBracket)?;
                LiteralOrList::List(values)
            } else {
                LiteralOrList::Literal({
                    let (literal, _) = self.parse_literal()?;
                    literal
                })
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

    fn parse_prompt_decl(&mut self, is_pub: bool) -> Result<PromptDecl, ParseError> {
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
            is_pub,
        })
    }

    fn parse_model_decl(&mut self, is_pub: bool) -> Result<ModelDecl, ParseError> {
        let name = self.expect_ident()?;
        self.expect(TokenKind::Assign)?;
        let expr = self.parse_expr()?;
        self.consume_stmt_end()?;
        Ok(ModelDecl { name, expr, is_pub })
    }

    fn parse_agent_decl(&mut self, is_pub: bool) -> Result<AgentDecl, ParseError> {
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
        Ok(AgentDecl {
            name,
            items,
            is_pub,
        })
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
        let (name, name_span) = self.expect_ident_with_span()?;
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
            name_span,
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
        let (var, var_span) = self.expect_ident_with_span()?;
        self.expect(TokenKind::In)?;
        let iter = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::For {
            var,
            var_span,
            iter,
            body,
        })
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
        let (base, base_span) = self.expect_ident_with_span()?;
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
        Ok(LValue {
            base,
            base_span,
            accesses,
        })
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
        let (literal, _) = self.parse_literal()?;
        Ok(Pattern::Literal(literal))
    }

    fn parse_literal(&mut self) -> Result<(Literal, Span), ParseError> {
        let token = self.advance();
        let span = Span::single(token.line, token.col);
        match token.kind {
            TokenKind::Int(value) => Ok((Literal::Int(value), span)),
            TokenKind::Float(value) => Ok((Literal::Float(value), span)),
            TokenKind::String(value) => Ok((Literal::String(value), span)),
            TokenKind::True => Ok((Literal::Bool(true), span)),
            TokenKind::False => Ok((Literal::Bool(false), span)),
            TokenKind::None => Ok((Literal::None, span)),
            _ => Err(self.error("Expected literal", &token)),
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_and()?;
        while self.matches(TokenKind::Or) {
            let token = self.previous();
            let span = Span::single(token.line, token.col);
            let right = self.parse_and()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::Or,
                right: Box::new(right),
                span,
            };
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_equality()?;
        while self.matches(TokenKind::And) {
            let token = self.previous();
            let span = Span::single(token.line, token.col);
            let right = self.parse_equality()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::And,
                right: Box::new(right),
                span,
            };
        }
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_comparison()?;
        loop {
            if self.matches(TokenKind::EqualEqual) {
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let right = self.parse_comparison()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Equal,
                    right: Box::new(right),
                    span,
                };
            } else if self.matches(TokenKind::BangEqual) {
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let right = self.parse_comparison()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::NotEqual,
                    right: Box::new(right),
                    span,
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
                let token = self.previous();
                Some((BinaryOp::Less, Span::single(token.line, token.col)))
            } else if self.matches(TokenKind::LessEqual) {
                let token = self.previous();
                Some((BinaryOp::LessEqual, Span::single(token.line, token.col)))
            } else if self.matches(TokenKind::Greater) {
                let token = self.previous();
                Some((BinaryOp::Greater, Span::single(token.line, token.col)))
            } else if self.matches(TokenKind::GreaterEqual) {
                let token = self.previous();
                Some((BinaryOp::GreaterEqual, Span::single(token.line, token.col)))
            } else {
                None
            };
            if let Some((op, span)) = op {
                let right = self.parse_additive()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(right),
                    span,
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
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let right = self.parse_multiplicative()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Add,
                    right: Box::new(right),
                    span,
                };
            } else if self.matches(TokenKind::Minus) {
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let right = self.parse_multiplicative()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Subtract,
                    right: Box::new(right),
                    span,
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
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let right = self.parse_unary()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Multiply,
                    right: Box::new(right),
                    span,
                };
            } else if self.matches(TokenKind::Slash) {
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let right = self.parse_unary()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Divide,
                    right: Box::new(right),
                    span,
                };
            } else if self.matches(TokenKind::Percent) {
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let right = self.parse_unary()?;
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::Modulo,
                    right: Box::new(right),
                    span,
                };
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if self.matches(TokenKind::Not) {
            let token = self.previous();
            let span = Span::single(token.line, token.col);
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(expr),
                span,
            });
        }
        if self.matches(TokenKind::Minus) {
            let token = self.previous();
            let span = Span::single(token.line, token.col);
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Negate,
                expr: Box::new(expr),
                span,
            });
        }
        if self.matches(TokenKind::Await) {
            let token = self.previous();
            let span = Span::single(token.line, token.col);
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Await,
                expr: Box::new(expr),
                span,
            });
        }
        if self.matches(TokenKind::Spawn) {
            let token = self.previous();
            let span = Span::single(token.line, token.col);
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnaryOp::Spawn,
                expr: Box::new(expr),
                span,
            });
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.matches(TokenKind::LParen) {
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let args = self.parse_arg_list()?;
                expr = Expr::Call {
                    callee: Box::new(expr),
                    args,
                    span,
                };
            } else if self.matches(TokenKind::LBracket) {
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let index = self.parse_expr()?;
                self.expect(TokenKind::RBracket)?;
                expr = Expr::Index {
                    target: Box::new(expr),
                    index: Box::new(index),
                    span,
                };
            } else if self.matches(TokenKind::Dot) {
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                let name = self.expect_ident()?;
                expr = Expr::Field {
                    target: Box::new(expr),
                    name,
                    span,
                };
            } else if self.matches(TokenKind::Question) {
                let token = self.previous();
                let span = Span::single(token.line, token.col);
                expr = Expr::Try {
                    expr: Box::new(expr),
                    span,
                };
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
            let token = self.previous();
            return Ok(Expr::Literal {
                lit: Literal::Bool(true),
                span: Span::single(token.line, token.col),
            });
        }
        if self.matches(TokenKind::False) {
            let token = self.previous();
            return Ok(Expr::Literal {
                lit: Literal::Bool(false),
                span: Span::single(token.line, token.col),
            });
        }
        if self.matches(TokenKind::None) {
            let token = self.previous();
            return Ok(Expr::Literal {
                lit: Literal::None,
                span: Span::single(token.line, token.col),
            });
        }
        if self.check_literal_start() {
            let (literal, span) = self.parse_literal()?;
            return Ok(Expr::Literal { lit: literal, span });
        }
        if self.matches(TokenKind::Match) {
            let token = self.previous();
            let span = Span::single(token.line, token.col);
            let expr = self.parse_expr()?;
            let arms = self.parse_match_arms(false)?;
            return Ok(Expr::Match {
                expr: Box::new(expr),
                arms,
                span,
            });
        }
        if self.check_ident() {
            let (name, span) = self.expect_ident_with_span()?;
            return Ok(Expr::Ident { name, span });
        }
        if self.matches(TokenKind::LBracket) {
            let token = self.previous();
            let span = Span::single(token.line, token.col);
            let mut values = Vec::new();
            if !self.check(TokenKind::RBracket) {
                values.push(self.parse_expr()?);
                while self.matches(TokenKind::Comma) {
                    values.push(self.parse_expr()?);
                }
            }
            self.expect(TokenKind::RBracket)?;
            return Ok(Expr::List {
                items: values,
                span,
            });
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
        let token = self.expect(TokenKind::LParen)?;
        let span = Span::single(token.line, token.col);
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
            span,
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
        let (path, _) = self.parse_path_with_spans()?;
        Ok(path)
    }

    fn parse_path_with_spans(&mut self) -> Result<(Vec<String>, Vec<Span>), ParseError> {
        let mut path = Vec::new();
        let (name, span) = self.expect_ident_with_span()?;
        path.push(name);
        let mut spans = vec![span];
        while self.matches(TokenKind::Dot) {
            let (name, span) = self.expect_ident_with_span()?;
            path.push(name);
            spans.push(span);
        }
        Ok((path, spans))
    }

    fn parse_import_path_with_spans(&mut self) -> Result<(Vec<String>, Vec<Span>), ParseError> {
        let mut path = Vec::new();
        let (name, span) = self.expect_ident_with_span()?;
        path.push(name);
        let mut spans = vec![span];
        while self.matches(TokenKind::ColonColon) {
            let (name, span) = self.expect_ident_with_span()?;
            path.push(name);
            spans.push(span);
        }
        Ok((path, spans))
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

    fn is_native_import_start(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Ident(ref name) if name == "native")
            && matches!(
                self.tokens.get(self.pos + 1).map(|t| &t.kind),
                Some(TokenKind::ColonColon)
            )
            && matches!(
                self.tokens.get(self.pos + 2).map(|t| &t.kind),
                Some(TokenKind::Import)
            )
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

    fn expect_ident_with_span(&mut self) -> Result<(String, Span), ParseError> {
        let token = self.advance();
        if let TokenKind::Ident(name) = token.kind {
            let span = Span::with_length(token.line, token.col, name.len());
            Ok((name, span))
        } else {
            Err(self.error("Expected identifier", &token))
        }
    }

    fn expect_string_with_span(&mut self) -> Result<(String, Span), ParseError> {
        let token = self.advance();
        if let TokenKind::String(value) = token.kind {
            let span = Span::with_length(token.line, token.col, value.len());
            Ok((value, span))
        } else {
            Err(self.error("Expected string literal", &token))
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
        let diagnostic = Diagnostic::new(message, self.source_name.as_deref())
            .with_span("here", Span::single(token.line, token.col))
            .with_source(&self.source);
        ParseError {
            diagnostic,
            line: token.line,
            col: token.col,
        }
    }
}
