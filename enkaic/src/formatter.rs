use std::collections::HashSet;
use std::fmt;

use crate::lexer::{tokenize, LexerError, TokenKind};

#[derive(Debug, Clone)]
pub struct FormatError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl FormatError {
    fn new(message: &str, line: usize, col: usize) -> Self {
        Self {
            message: message.to_string(),
            line,
            col,
        }
    }

    fn from_lexer(err: LexerError) -> Self {
        Self::new(&err.message, err.line, err.col)
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}:{}", self.message, self.line, self.col)
    }
}

impl std::error::Error for FormatError {}

pub fn format_source(source: &str) -> Result<String, FormatError> {
    let tokens = tokenize(source).map_err(FormatError::from_lexer)?;
    let mut block_start_lines = HashSet::new();
    let mut block_end_lines = HashSet::new();
    for token in tokens {
        match token.kind {
            TokenKind::BlockStart => {
                block_start_lines.insert(token.line);
            }
            TokenKind::BlockEnd => {
                block_end_lines.insert(token.line);
            }
            _ => {}
        }
    }

    let normalized = normalize_line_endings(source);
    let mut depth = 0usize;
    let mut out = String::new();
    for (idx, raw_line) in normalized.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw_line.trim_start();
        if trimmed.is_empty() {
            out.push('\n');
            continue;
        }
        if block_end_lines.contains(&line_no) {
            if depth == 0 {
                return Err(FormatError::new(
                    "Block end without matching start",
                    line_no,
                    1,
                ));
            }
            depth -= 1;
        }
        let indent = " ".repeat(depth * 4);
        out.push_str(&indent);
        out.push_str(trimmed);
        out.push('\n');
        if block_start_lines.contains(&line_no) {
            depth += 1;
        }
    }
    Ok(out)
}

pub fn check_format(source: &str) -> Result<(), FormatError> {
    let formatted = format_source(source)?;
    let normalized = normalize_line_endings(source);
    if normalized != formatted {
        let line = first_diff_line(&normalized, &formatted);
        return Err(FormatError::new("File is not formatted", line, 1));
    }
    Ok(())
}

fn normalize_line_endings(input: &str) -> String {
    input.replace("\r\n", "\n")
}

fn first_diff_line(left: &str, right: &str) -> usize {
    let mut left_iter = left.lines();
    let mut right_iter = right.lines();
    let mut line = 1;
    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(a), Some(b)) => {
                if a != b {
                    return line;
                }
            }
            (None, None) => return 1,
            _ => return line,
        }
        line += 1;
    }
}
