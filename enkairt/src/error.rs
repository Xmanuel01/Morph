use std::fmt;

#[derive(Debug, Clone)]
pub struct RuntimeFrame {
    pub function: Option<String>,
    pub source: Option<String>,
    pub line: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub message: String,
    pub frames: Vec<RuntimeFrame>,
}

impl RuntimeError {
    pub fn new(msg: &str) -> Self {
        Self {
            message: msg.to_string(),
            frames: Vec::new(),
        }
    }

    pub fn with_frames(mut self, frames: Vec<RuntimeFrame>) -> Self {
        if self.frames.is_empty() {
            self.frames = frames;
        } else {
            self.frames.extend(frames);
        }
        self
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.message)?;
        for frame in &self.frames {
            let func = frame.function.as_deref().unwrap_or("<anonymous>");
            match (&frame.source, frame.line) {
                (Some(source), Some(line)) => {
                    writeln!(f, "  at {} ({}:{})", func, source, line)?;
                }
                (Some(source), None) => {
                    writeln!(f, "  at {} ({})", func, source)?;
                }
                (None, Some(line)) => {
                    writeln!(f, "  at {} (line {})", func, line)?;
                }
                (None, None) => {
                    writeln!(f, "  at {}", func)?;
                }
            }
        }
        Ok(())
    }
}

impl std::error::Error for RuntimeError {}
