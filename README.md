MORPH PROGRAMMING LANGUAGE
[![CI](https://github.com/Xmanuel01/MorphLan/actions/workflows/ci.yml/badge.svg)](https://github.com/Xmanuel01/MorphLan/actions/workflows/ci.yml)

Overview
Morph is a programming language with block structure defined by :: tokens, a clean
assignment operator (:=), and an AI-native roadmap (tools, agents, memory, policy).
This repository contains the v0.2 implementation in Rust.

Status (v0.2)
- Lexer + parser + AST (per docs/Morph.spec)
- Tree-walk interpreter
- Module loading + use resolution
- Diagnostics with line/col + snippet
- Runtime stack traces
- Minimal formatter (morph fmt --check)
- Policy enforcement MVP (default deny + allow rules)

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
- cargo run -p morph -- run path\\to\\file.morph
- cargo run -p morph -- run examples\\project_v02
- cargo run -p morph -- fmt --check examples\\project_v02\\src\\main.morph
- cargo test

License
Apache 2.0

Created by
Emmanuel Odhiambo Onyango
