use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub col: usize,
}

impl Token {
    fn new(kind: TokenKind, line: usize, col: usize) -> Self {
        Self { kind, line, col }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    BlockStart,
    BlockEnd,
    Newline,
    Eof,

    Ident(String),
    Int(i64),
    Float(f64),
    String(String),

    Let,
    Fn,
    If,
    Else,
    While,
    For,
    In,
    Match,
    Return,
    Break,
    Continue,
    Try,
    Catch,
    Type,
    Enum,
    Impl,
    Import,
    Use,
    As,
    Pub,
    Async,
    Tool,
    Agent,
    Policy,
    Memory,
    Prompt,
    Model,
    Allow,
    Deny,
    True,
    False,
    None,
    And,
    Or,
    Not,
    Await,
    Spawn,

    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semicolon,
    Dot,

    Assign,   // :=
    Equal,    // =
    Arrow,    // ->
    FatArrow, // =>

    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    EqualEqual,
    BangEqual,

    Question,
    ColonColon,
}

#[derive(Debug, Clone)]
pub struct LexerError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}:{}", self.message, self.line, self.col)
    }
}

impl std::error::Error for LexerError {}

pub fn tokenize(source: &str) -> Result<Vec<Token>, LexerError> {
    let lexer = Lexer::new(source);
    lexer.tokenize()
}

struct Lexer {
    chars: Vec<char>,
    index: usize,
    line: usize,
    col: usize,
    in_block_comment: bool,
    paren_depth: i32,
    bracket_depth: i32,
    block_depth: i32,
    line_tokens: Vec<Token>,
    output: Vec<Token>,
}

impl Lexer {
    fn new(source: &str) -> Self {
        Self {
            chars: source.chars().collect(),
            index: 0,
            line: 1,
            col: 1,
            in_block_comment: false,
            paren_depth: 0,
            bracket_depth: 0,
            block_depth: 0,
            line_tokens: Vec::new(),
            output: Vec::new(),
        }
    }

    fn tokenize(mut self) -> Result<Vec<Token>, LexerError> {
        while self.index < self.chars.len() {
            let c = self.chars[self.index];
            if c == '\n' {
                self.finish_line()?;
                if self.paren_depth == 0 && self.bracket_depth == 0 {
                    self.output
                        .push(Token::new(TokenKind::Newline, self.line, self.col));
                }
                self.index += 1;
                self.line += 1;
                self.col = 1;
                continue;
            }

            if self.in_block_comment {
                if c == '*' && self.peek_char(1) == Some('/') {
                    self.in_block_comment = false;
                    self.advance(2);
                    continue;
                }
                self.advance(1);
                continue;
            }

            if c.is_whitespace() {
                self.advance(1);
                continue;
            }

            if c == '/' && self.peek_char(1) == Some('/') {
                self.skip_line_comment();
                continue;
            }

            if c == '/' && self.peek_char(1) == Some('*') {
                self.in_block_comment = true;
                self.advance(2);
                continue;
            }

            if c == '"' {
                let token = self.lex_string()?;
                self.line_tokens.push(token);
                continue;
            }

            if c.is_ascii_digit() {
                let token = self.lex_number()?;
                self.line_tokens.push(token);
                continue;
            }

            if is_ident_start(c) {
                let token = self.lex_ident_or_keyword();
                self.line_tokens.push(token);
                continue;
            }

            let token = self.lex_punct()?;
            self.track_nesting(&token.kind)?;
            self.line_tokens.push(token);
        }

        self.finish_line()?;

        if self.block_depth != 0 {
            return Err(self.error("Unclosed block", self.line, self.col));
        }

        self.output
            .push(Token::new(TokenKind::Eof, self.line, self.col));
        Ok(self.output)
    }

    fn finish_line(&mut self) -> Result<(), LexerError> {
        if self.line_tokens.is_empty() {
            return Ok(());
        }

        let len = self.line_tokens.len();
        if len > 0 && self.line_tokens[len - 1].kind == TokenKind::ColonColon {
            if len == 1 {
                let line = self.line_tokens[len - 1].line;
                let col = self.line_tokens[len - 1].col;
                self.line_tokens[len - 1].kind = TokenKind::BlockEnd;
                if self.block_depth == 0 {
                    return Err(self.error("Block end without matching start", line, col));
                }
                self.block_depth -= 1;
            } else {
                self.line_tokens[len - 1].kind = TokenKind::BlockStart;
                self.block_depth += 1;
            }
        }

        self.output.append(&mut self.line_tokens);
        Ok(())
    }

    fn lex_string(&mut self) -> Result<Token, LexerError> {
        let start_line = self.line;
        let start_col = self.col;
        self.advance(1);
        let mut value = String::new();
        while let Some(c) = self.peek_char(0) {
            if c == '"' {
                self.advance(1);
                return Ok(Token::new(TokenKind::String(value), start_line, start_col));
            }
            if c == '\n' {
                return Err(self.error("Unterminated string literal", start_line, start_col));
            }
            if c == '\\' {
                let escaped = match self.peek_char(1) {
                    Some('n') => '\n',
                    Some('t') => '\t',
                    Some('"') => '"',
                    Some('\\') => '\\',
                    Some(other) => other,
                    None => {
                        return Err(self.error(
                            "Unterminated string escape",
                            start_line,
                            start_col,
                        ));
                    }
                };
                value.push(escaped);
                self.advance(2);
                continue;
            }
            value.push(c);
            self.advance(1);
        }
        Err(self.error("Unterminated string literal", start_line, start_col))
    }

    fn lex_number(&mut self) -> Result<Token, LexerError> {
        let start_line = self.line;
        let start_col = self.col;
        let mut raw = String::new();
        let mut has_dot = false;
        let mut has_exp = false;

        while let Some(c) = self.peek_char(0) {
            if c.is_ascii_digit() || c == '_' {
                raw.push(c);
                self.advance(1);
                continue;
            }
            if c == '.'
                && !has_dot
                && !has_exp
                && self
                    .peek_char(1)
                    .map(|d| d.is_ascii_digit())
                    .unwrap_or(false)
            {
                has_dot = true;
                raw.push(c);
                self.advance(1);
                continue;
            }
            if (c == 'e' || c == 'E') && !has_exp {
                has_exp = true;
                raw.push(c);
                self.advance(1);
                if let Some(sign) = self.peek_char(0) {
                    if sign == '+' || sign == '-' {
                        raw.push(sign);
                        self.advance(1);
                    }
                }
                continue;
            }
            break;
        }

        let clean = raw.replace('_', "");
        if has_dot || has_exp {
            let value = clean
                .parse::<f64>()
                .map_err(|_| self.error("Invalid float literal", start_line, start_col))?;
            Ok(Token::new(TokenKind::Float(value), start_line, start_col))
        } else {
            let value = clean
                .parse::<i64>()
                .map_err(|_| self.error("Invalid integer literal", start_line, start_col))?;
            Ok(Token::new(TokenKind::Int(value), start_line, start_col))
        }
    }

    fn lex_ident_or_keyword(&mut self) -> Token {
        let start_line = self.line;
        let start_col = self.col;
        let mut ident = String::new();
        while let Some(c) = self.peek_char(0) {
            if is_ident_continue(c) {
                ident.push(c);
                self.advance(1);
            } else {
                break;
            }
        }

        let kind = match ident.as_str() {
            "let" => TokenKind::Let,
            "fn" => TokenKind::Fn,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "match" => TokenKind::Match,
            "return" => TokenKind::Return,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "try" => TokenKind::Try,
            "catch" => TokenKind::Catch,
            "type" => TokenKind::Type,
            "enum" => TokenKind::Enum,
            "impl" => TokenKind::Impl,
            "import" => TokenKind::Import,
            "use" => TokenKind::Use,
            "as" => TokenKind::As,
            "pub" => TokenKind::Pub,
            "async" => TokenKind::Async,
            "tool" => TokenKind::Tool,
            "agent" => TokenKind::Agent,
            "policy" => TokenKind::Policy,
            "memory" => TokenKind::Memory,
            "prompt" => TokenKind::Prompt,
            "model" => TokenKind::Model,
            "allow" => TokenKind::Allow,
            "deny" => TokenKind::Deny,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "none" => TokenKind::None,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            "await" => TokenKind::Await,
            "spawn" => TokenKind::Spawn,
            _ => TokenKind::Ident(ident),
        };

        Token::new(kind, start_line, start_col)
    }

    fn lex_punct(&mut self) -> Result<Token, LexerError> {
        let start_line = self.line;
        let start_col = self.col;
        let c = self.peek_char(0).unwrap();
        let next = self.peek_char(1);
        let kind = match (c, next) {
            (':', Some(':')) => {
                self.advance(2);
                TokenKind::ColonColon
            }
            (':', Some('=')) => {
                self.advance(2);
                TokenKind::Assign
            }
            ('-', Some('>')) => {
                self.advance(2);
                TokenKind::Arrow
            }
            ('=', Some('>')) => {
                self.advance(2);
                TokenKind::FatArrow
            }
            ('=', Some('=')) => {
                self.advance(2);
                TokenKind::EqualEqual
            }
            ('!', Some('=')) => {
                self.advance(2);
                TokenKind::BangEqual
            }
            ('<', Some('=')) => {
                self.advance(2);
                TokenKind::LessEqual
            }
            ('>', Some('=')) => {
                self.advance(2);
                TokenKind::GreaterEqual
            }
            _ => {
                self.advance(1);
                match c {
                    '(' => TokenKind::LParen,
                    ')' => TokenKind::RParen,
                    '[' => TokenKind::LBracket,
                    ']' => TokenKind::RBracket,
                    '{' => TokenKind::LBrace,
                    '}' => TokenKind::RBrace,
                    ',' => TokenKind::Comma,
                    ':' => TokenKind::Colon,
                    ';' => TokenKind::Semicolon,
                    '.' => TokenKind::Dot,
                    '=' => TokenKind::Equal,
                    '+' => TokenKind::Plus,
                    '-' => TokenKind::Minus,
                    '*' => TokenKind::Star,
                    '/' => TokenKind::Slash,
                    '%' => TokenKind::Percent,
                    '<' => TokenKind::Less,
                    '>' => TokenKind::Greater,
                    '?' => TokenKind::Question,
                    _ => {
                        return Err(self.error(
                            &format!("Unexpected character '{}'", c),
                            start_line,
                            start_col,
                        ));
                    }
                }
            }
        };
        Ok(Token::new(kind, start_line, start_col))
    }

    fn track_nesting(&mut self, kind: &TokenKind) -> Result<(), LexerError> {
        match kind {
            TokenKind::LParen => self.paren_depth += 1,
            TokenKind::RParen => {
                self.paren_depth -= 1;
                if self.paren_depth < 0 {
                    return Err(self.error("Unmatched ')'", self.line, self.col));
                }
            }
            TokenKind::LBracket => self.bracket_depth += 1,
            TokenKind::RBracket => {
                self.bracket_depth -= 1;
                if self.bracket_depth < 0 {
                    return Err(self.error("Unmatched ']'", self.line, self.col));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn skip_line_comment(&mut self) {
        while let Some(c) = self.peek_char(0) {
            if c == '\n' {
                break;
            }
            self.advance(1);
        }
    }

    fn peek_char(&self, offset: usize) -> Option<char> {
        self.chars.get(self.index + offset).copied()
    }

    fn advance(&mut self, count: usize) {
        for _ in 0..count {
            if let Some(c) = self.peek_char(0) {
                self.index += 1;
                if c == '\n' {
                    self.line += 1;
                    self.col = 1;
                } else {
                    self.col += 1;
                }
            }
        }
    }

    fn error(&self, message: &str, line: usize, col: usize) -> LexerError {
        LexerError {
            message: message.to_string(),
            line,
            col,
        }
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}
