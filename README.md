MORPH PROGRAMMING LANGUAGE
[![CI](https://github.com/Xmanuel01/MorphLan/actions/workflows/ci.yml/badge.svg)](https://github.com/Xmanuel01/MorphLan/actions/workflows/ci.yml)

Overview
Morph is a programming language with block structure defined by :: tokens, a clean
assignment operator (:=), and an AI-native roadmap (tools, agents, memory, policy).
This repository contains the v0.3 implementation in Rust.

Status (v0.3)
- Lexer + parser + AST (per docs/Morph.spec)
- Tree-walk interpreter
- Module loading + use resolution + pub exports/re-exports
- Diagnostics with line/col + labeled spans
- Runtime stack traces
- Minimal formatter (morph fmt --check)
- Policy enforcement (default deny + allow rules + filters)
- Local path dependencies in morph.toml
- std.string + std.fs (policy-gated)

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

Running
- Install (recommended):
  - Windows (PowerShell):
    - `iwr -useb https://.../install.ps1 | iex`
  - Linux/macOS:
    - `curl -fsSL https://.../install.sh | sh`
  - Release assets are expected to follow:
    - `morph-<version>-windows-x86_64.zip`
    - `morph-<version>-linux-x86_64.tar.gz`
    - `morph-<version>-linux-aarch64.tar.gz`
    - `morph-<version>-macos-x86_64.tar.gz`
    - `morph-<version>-macos-aarch64.tar.gz`
    - `morph-setup-<version>.exe`
- cargo run -p morph -- run path\\to\\file.morph
- cargo run -p morph -- run examples\\project_v02
- cargo run -p morph -- fmt --check examples\\project_v02\\src\\main.morph
- cargo test

License
Apache 2.0

Created by
Emmanuel Odhiambo Onyango
