ENKAI PROGRAMMING LANGUAGE
[![CI](https://github.com/Xmanuel01/Enkai/actions/workflows/ci.yml/badge.svg)](https://github.com/Xmanuel01/Enkai/actions/workflows/ci.yml)

Overview
Enkai is a programming language with block structure defined by :: tokens, a clean
assignment operator (:=), and an AI-native roadmap (tools, agents, memory, policy).
This repository contains the v0.8 implementation in Rust.

Status (v0.8)
- Bytecode VM + globals + type-checking
- Module system with public/private exports
- CLI: run/check/fmt/test/train/eval
- FFI runtime + native std modules (fsx/zstd/hash)
- Tokenizer + dataset streaming + checkpoints

Workspace structure
- enkaic: compiler front-end (lexer/parser/AST/type-check stubs)
- enkairt: runtime/interpreter
- enkai: CLI wrapper

Spec
See docs/Enkai.spec for the grammar, keywords, and :: block rules.

Quick example
fn greet(name: String) -> String ::
    return "Hello " + name
::

let msg := greet("Enkai")
print(msg)

Install (users, no Rust required)
- Option A: one-line installer (recommended)
  - Windows (PowerShell):
    - `iwr -useb https://raw.githubusercontent.com/Xmanuel01/Enkai/main/install/install.ps1 | iex`
  - Linux/macOS:
    - `curl -fsSL https://raw.githubusercontent.com/Xmanuel01/Enkai/main/install/install.sh | sh`
- Option B: manual download
  - Download the correct archive from GitHub Releases:
    - `enkai-<version>-windows-x86_64.zip`
    - `enkai-<version>-linux-x86_64.tar.gz`
    - `enkai-<version>-macos-x86_64.tar.gz`
    - `enkai-<version>-macos-aarch64.tar.gz`
  - Unzip and run:
    - `enkai --version`
    - `enkai run examples/hello/main.enk`

Developer install (requires Rust)
- `cargo build -p enkai --release`
- `cargo run -p enkai -- run path\\to\\file.enk`
- `cargo run -p enkai -- run examples\\project_v02`
- `cargo run -p enkai -- fmt --check examples\\project_v02\\src\\main.enk`
- `cargo test`

License
Apache 2.0

Created by
Emmanuel Odhiambo Onyango


