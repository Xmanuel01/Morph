use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
    pub end_col: usize,
}

impl Span {
    pub fn single(line: usize, col: usize) -> Self {
        Self {
            line,
            col,
            end_col: col,
        }
    }

    pub fn with_length(line: usize, col: usize, len: usize) -> Self {
        let end_col = if len == 0 { col } else { col + len - 1 };
        Self { line, col, end_col }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LabeledSpan {
    pub span: Span,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub message: String,
    pub source_name: Option<String>,
    pub spans: Vec<LabeledSpan>,
    pub snippet: Option<String>,
}

impl Diagnostic {
    pub fn new(message: &str, source_name: Option<&str>) -> Self {
        Self {
            message: message.to_string(),
            source_name: source_name.map(|name| name.to_string()),
            spans: Vec::new(),
            snippet: None,
        }
    }

    pub fn with_span(mut self, label: &str, span: Span) -> Self {
        self.spans.push(LabeledSpan {
            span,
            label: label.to_string(),
        });
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        let lines: Vec<String> = source.lines().map(|line| line.to_string()).collect();
        self.snippet = format_snippet_from_lines(&lines, &self.spans);
        self
    }

    pub fn with_source_lines(mut self, lines: &[String]) -> Self {
        self.snippet = format_snippet_from_lines(lines, &self.spans);
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error: {}", self.message)?;
        if let Some(primary) = self.spans.first() {
            if let Some(name) = &self.source_name {
                write!(
                    f,
                    "\n--> {}:{}:{}",
                    name, primary.span.line, primary.span.col
                )?;
            } else {
                write!(f, "\n--> {}:{}", primary.span.line, primary.span.col)?;
            }
        } else if let Some(name) = &self.source_name {
            write!(f, "\n--> {}", name)?;
        }
        if let Some(snippet) = &self.snippet {
            write!(f, "\n{}", snippet)?;
        }
        Ok(())
    }
}

fn format_snippet_from_lines(lines: &[String], spans: &[LabeledSpan]) -> Option<String> {
    if spans.is_empty() {
        return None;
    }
    let mut output = String::new();
    let mut spans_by_line: Vec<(usize, Vec<&LabeledSpan>)> = Vec::new();
    for span in spans {
        let line = span.span.line;
        if let Some((_, existing)) = spans_by_line.iter_mut().find(|(l, _)| *l == line) {
            existing.push(span);
        } else {
            spans_by_line.push((line, vec![span]));
        }
    }
    spans_by_line.sort_by_key(|(line, _)| *line);
    let width = spans_by_line
        .last()
        .map(|(line, _)| line.to_string().len())
        .unwrap_or(1);
    for (idx, (line_no, line_spans)) in spans_by_line.iter().enumerate() {
        let line_text = lines.get(line_no.saturating_sub(1))?;
        if idx > 0 {
            output.push('\n');
        }
        output.push_str(&format!(
            "{:>width$} | {}",
            line_no,
            line_text,
            width = width
        ));
        for span in line_spans {
            output.push('\n');
            let mut caret_line = String::new();
            let caret_pos = span.span.col.saturating_sub(1);
            for _ in 0..caret_pos {
                caret_line.push(' ');
            }
            let len = span.span.end_col.saturating_sub(span.span.col) + 1;
            for _ in 0..len {
                caret_line.push('^');
            }
            if !span.label.is_empty() {
                caret_line.push(' ');
                caret_line.push_str(&span.label);
            }
            output.push_str(&format!("{:>width$} | {}", "", caret_line, width = width));
        }
    }
    Some(output)
}
