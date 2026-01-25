MORPH PROGRAMMING LANGUAGE
[![CI](https://github.com/Xmanuel01/Morph/actions/workflows/ci.yml/badge.svg)](https://github.com/Xmanuel01/Morph/actions/workflows/ci.yml)

Overview
Morph is a programming language with block structure defined by :: tokens, a clean
assignment operator (:=), and an AI-native roadmap (tools, agents, memory, policy).
This repository contains the v0.8 implementation in Rust.

Status (v0.8)
- Bytecode VM + globals + type-checking
- Module system with public/private exports
- CLI: run/check/fmt/test/train/eval
- FFI runtime + native std modules (fsx/zstd/hash)
- Tokenizer + dataset streaming + checkpoints

Workspace structure
- morphc: compiler front-end (lexer/parser/AST/type-check stubs)
- morphrt: runtime/interpreter
- morph: CLI wrapper

Spec
See docs/Morph.spec for the grammar, keywords, and :: block rules.

Quick example
fn greet(name: String) -> String ::
    return "Hello " + name
::

let msg := greet("Morph")
print(msg)

Install (users, no Rust required)
- Option A: one-line installer (recommended)
  - Windows (PowerShell):
    - `iwr -useb https://raw.githubusercontent.com/Xmanuel01/Morph/main/install/install.ps1 | iex`
  - Linux/macOS:
    - `curl -fsSL https://raw.githubusercontent.com/Xmanuel01/Morph/main/install/install.sh | sh`
- Option B: manual download
  - Download the correct archive from GitHub Releases:
    - `morph-<version>-windows-x86_64.zip`
    - `morph-<version>-linux-x86_64.tar.gz`
    - `morph-<version>-macos-x86_64.tar.gz`
    - `morph-<version>-macos-aarch64.tar.gz`
  - Unzip and run:
    - `morph --version`
    - `morph run examples/hello/main.morph`

Developer install (requires Rust)
- `cargo build -p morph --release`
- `cargo run -p morph -- run path\\to\\file.morph`
- `cargo run -p morph -- run examples\\project_v02`
- `cargo run -p morph -- fmt --check examples\\project_v02\\src\\main.morph`
- `cargo test`

License
Apache 2.0

Created by
Emmanuel Odhiambo Onyango
