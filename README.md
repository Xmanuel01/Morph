ENKAI PROGRAMMING LANGUAGE
[![CI](https://github.com/Xmanuel01/Enkai/actions/workflows/ci.yml/badge.svg)](https://github.com/Xmanuel01/Enkai/actions/workflows/ci.yml)

Overview
Enkai is a programming language with block structure defined by :: tokens, a clean
assignment operator (:=), and an AI-native roadmap (tools, agents, memory, policy).
This repository contains the v1.9.6 implementation in Rust.

Status (v1.9.6)
- Bytecode VM + globals + type-checking
- Module system with public/private exports
- CLI: run/serve/new/sdk/check/fmt/fmt-lite/lint-lite/tokenizer-lite/dataset-lite/litec/build/test/train/eval/migrate/doctor
- FFI runtime + native std modules (fsx/zstd/hash/db/tls)
- Tokenizer + dataset streaming + checkpoints
- Backend serving stack: routing, middleware/auth/rate-limit, SSE/WebSocket streaming, TLS/SQLite/Postgres helpers
- Frontend stack: React/TypeScript scaffolds + typed SDK generation
- Serve/frontend contract snapshots and compatibility freeze gates for generated backend + SDK
- Schema-versioned conversation persistence (`schema_version: 1`) with startup migration hook for legacy scaffold state
- Bootstrap-lite/core toolchain path with `litec` stage0/stage1 bytecode equivalence checks, phase staging (`litec stage`), and self-host CI corpus validation (`litec selfhost-ci`)
- Self-host replacement-readiness gate with Stage1/Stage2 fixed-point checks (`litec replace-check`)
- Compatibility/deprecation governance and self-host fallback workflow docs for v1.9.6 release readiness
- Master release pipeline and GPU evidence verification scripts for v1.9.6 operational sign-off

Workspace structure
- enkaic: compiler front-end (lexer/parser/AST/type-check stubs)
- enkairt: bytecode VM runtime (production) + legacy tree-walk interpreter (non-production/reference)
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

Container image
- Built on CI for pushes/tags: `ghcr.io/Xmanuel01/Enkai:latest` (or `:<tag>`).
- Run a program from your host:
  - `docker run --rm -e ENKAI_STD_PATH=/opt/enkai/std -v $PWD:/work -w /work ghcr.io/Xmanuel01/Enkai:latest enkai run examples/hello/main.enk`
- The image ships the compiled `enkai` binary and bundled stdlib at `/opt/enkai/std`.

Developer install (requires Rust)
- `cargo build -p enkai --release`
- `cargo run -p enkai -- run path\\to\\file.enk`
- `cargo run -p enkai -- run examples\\project_v02`
- `cargo run -p enkai -- fmt --check examples\\project_v02\\src\\main.enk`
- `cargo test`

Tensor/FFI features (current)
- Backend selection with CPU fallback: `enkai_backend_set("torch"|"cpu")`, guards on all extern ops.
- Ref-counted handles with stale-handle detection and live leak counters for tensors/devices/optimizers/scalers.
- Panic guard on every extern "C" entry; errors reported via thread-local `enkai_tensor_last_error`.
- AMP support: GradScaler handles, autocast enter/exit, and `enkai_amp_step` convenience.
- Ranked checkpoint save/load with SHA-256 integrity files (per-rank shards).
- Distributed runtime is environment-gated for multi-rank mode (`ENKAI_ENABLE_DIST=1`).
- Multi-rank init/allreduce paths are implemented for `enkai_tensor` builds with `torch,dist`.
- Operator-run GPU soak evidence is still required for final hardware sign-off.
- Multi-rank distributed mode is explicitly gated: set `ENKAI_ENABLE_DIST=1`.
- See `docs/tensor_api.md` for the full surface and safety contracts.

License
Apache 2.0

Created by
Emmanuel Odhiambo Onyango

