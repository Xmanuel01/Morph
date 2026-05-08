use std::collections::{HashMap, HashSet};
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
    let normalized = normalize_line_endings(source);
    let lines = normalized.lines().collect::<Vec<_>>();
    let mut block_start_lines = HashSet::new();
    let mut block_end_lines = HashSet::new();
    let mut tagged_end_lines = HashMap::new();
    let mut stack: Vec<BlockFrame> = Vec::new();
    for token in &tokens {
        match &token.kind {
            TokenKind::BlockStart => {
                block_start_lines.insert(token.line);
                if let Some(parent) = stack.last_mut() {
                    parent.child_blocks += 1;
                }
                stack.push(BlockFrame {
                    tag: infer_block_tag(lines.get(token.line.saturating_sub(1)).copied()),
                    start_line: token.line,
                    child_blocks: 0,
                });
            }
            TokenKind::BlockEnd => {
                block_end_lines.insert(token.line);
                let frame = stack.pop().ok_or_else(|| {
                    FormatError::new("Block end without matching start", token.line, token.col)
                })?;
                if let Some(tag) = preferred_tag_for_frame(&frame, token.line) {
                    tagged_end_lines.insert(token.line, format!("::{}", tag));
                }
            }
            TokenKind::BlockEndTagged(found) => {
                block_end_lines.insert(token.line);
                let frame = stack.pop().ok_or_else(|| {
                    FormatError::new("Block end without matching start", token.line, token.col)
                })?;
                if let Some(expected) = frame.tag {
                    if !formatter_block_tag_matches(expected, found) {
                        return Err(FormatError::new(
                            &format!(
                                "SyntaxError: expected ::{} to close {} block, found ::{}",
                                expected, expected, found
                            ),
                            token.line,
                            token.col,
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(frame) = stack.pop() {
        return Err(FormatError::new(
            "Block start without matching end",
            frame.start_line,
            1,
        ));
    }

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
        if let Some(tagged) = tagged_end_lines.get(&line_no) {
            if trimmed == "::" {
                out.push_str(tagged);
            } else {
                out.push_str(trimmed);
            }
        } else {
            out.push_str(trimmed);
        }
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

#[derive(Debug, Clone)]
struct BlockFrame {
    tag: Option<&'static str>,
    start_line: usize,
    child_blocks: usize,
}

fn preferred_tag_for_frame(frame: &BlockFrame, end_line: usize) -> Option<&'static str> {
    let tag = frame.tag?;
    let length = end_line.saturating_sub(frame.start_line);
    let should_tag = matches!(
        tag,
        "fn" | "policy" | "while" | "for" | "match" | "struct" | "enum" | "impl"
            | "module" | "class" | "trait" | "try" | "catch" | "agent" | "prompt" | "model"
            | "native" | "input" | "output" | "template"
    ) || matches!(tag, "if" | "else")
        || length > 5
        || frame.child_blocks > 0;
    should_tag.then_some(tag)
}

fn infer_block_tag(line: Option<&str>) -> Option<&'static str> {
    let mut trimmed = line?.trim_start();
    if trimmed.starts_with("pub ") {
        trimmed = trimmed.trim_start_matches("pub ").trim_start();
    }
    if trimmed.starts_with("async ") {
        trimmed = trimmed.trim_start_matches("async ").trim_start();
    }
    if trimmed.starts_with("else if ") {
        return Some("if");
    }
    if trimmed.starts_with("native::import") {
        return Some("native");
    }
    let first = trimmed
        .split(|ch: char| ch.is_whitespace() || ch == '(' || ch == ':')
        .next()
        .unwrap_or_default();
    match first {
        "fn" => Some("fn"),
        "if" => Some("if"),
        "else" => Some("else"),
        "while" => Some("while"),
        "for" => Some("for"),
        "policy" => Some("policy"),
        "match" => Some("match"),
        "type" => Some("struct"),
        "enum" => Some("enum"),
        "impl" => Some("impl"),
        "module" => Some("module"),
        "class" => Some("class"),
        "trait" => Some("trait"),
        "try" => Some("try"),
        "catch" => Some("catch"),
        "agent" => Some("agent"),
        "prompt" => Some("prompt"),
        "model" => Some("model"),
        "input" => Some("input"),
        "output" => Some("output"),
        "template" => Some("template"),
        _ => None,
    }
}

fn formatter_block_tag_matches(expected: &str, found: &str) -> bool {
    expected == found || (expected == "struct" && found == "type")
}
