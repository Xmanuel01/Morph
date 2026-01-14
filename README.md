MORPH PROGRAMMING LANGUAGE
[![CI](https://github.com/Xmanuel01/MorphLan/actions/workflows/ci.yml/badge.svg)](https://github.com/Xmanuel01/MorphLan/actions/workflows/ci.yml)

Overview
Morph is a programming language with block structure defined by :: tokens, a clean
assignment operator (:=), and an AI-native roadmap (tools, agents, memory, policy).
This repository contains the v0.1 implementation in Rust.

Status (v0.1)
- Lexer + parser + AST (per docs/Morph.spec)
- Tree-walk interpreter
- Minimal stdlib stubs (print, readline, len)
- CLI wrapper (morph run, morph fmt/test stubs)

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
- cargo test

License
Apache 2.0

Created by
Emmanuel Odhiambo Onyango
