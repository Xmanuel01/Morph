# Enkai Language Specification (v0.1 -> v0.9.3)

Status: stable.
This document is the normative language and runtime surface for Enkai v0.9.3,
including compatibility constraints carried from v0.1 onward.

-------------------------------------------------------------------------------
1. Scope
-------------------------------------------------------------------------------

This specification covers:
- Core syntax and block rules.
- Module/import semantics.
- Type and expression forms supported by parser, checker, compiler, and VM.
- Built-in runtime modules shipped in v0.9.2.
- CLI entrypoints used in production.

This specification does not claim features that are still stubbed or not yet implemented.
Those are listed in Section 12.

-------------------------------------------------------------------------------
1.1 Version Coverage and Compatibility
-------------------------------------------------------------------------------

Compatibility baseline:
- v0.1-v0.3: core syntax, parser/VM foundations.
- v0.4: bytecode + VM execution and strict checker integration.
- v0.5: multi-file modules/imports, visibility rules, formatter/test command.
- v0.6: FFI surface and native std bindings.
- v0.7: concurrency/network/http/json runtime modules.
- v0.8: tokenizer/dataset/checkpoint/train-eval pipeline.
- v0.9-v0.9.2: tensor backend integration, optimizer/autograd helpers,
  checkpoint C ABI hooks, and distributed hooks with known limits.
- v0.9.3: TinyLM transformer forward + cross-entropy training path, single-device
  soak/integrity harnesses, and gated multi-GPU harness scripts.

Compatibility policy:
- `.enk` and `.en` are primary source extensions.
- Legacy compatibility paths may exist in runtime/FFI loaders, but are not the
  primary contract unless listed explicitly.

-------------------------------------------------------------------------------
1.2 Validation Gate Status (v0.9.3)
-------------------------------------------------------------------------------

Current verification status:
- Training path: TinyLM forward + cross-entropy is active in runtime.
- Single-device soak gate: CPU-only validation pass is verified.
- Single-GPU CUDA soak gate: pending operator run on CUDA-capable host.
- 2-GPU and 4-GPU distributed gates: harnesses are implemented and gated, pending
  operator validation on CUDA/NCCL-capable host.

Gate policy:
- `ENKAI_SINGLE_GPU_GREEN=1` is a release gate input for multi-GPU harness
  execution.
- Multi-GPU harness scripts require explicit opt-in via
  `ENKAI_RUN_MULTI_GPU_TESTS=1`.

-------------------------------------------------------------------------------
2. Lexical Structure
-------------------------------------------------------------------------------

Source and identifiers:
- Source files are UTF-8 text.
- Identifiers are ASCII and match `[A-Za-z_][A-Za-z0-9_]*`.
- Keywords are case-sensitive.

Comments:
- Line comments: `// ...`
- Block comments: `/* ... */` (non-nesting)

Literals:
- Int, Float, String, Bool (`true`/`false`), `none`.

Operators and punctuation:
- Block delimiters: `::`
- Binding/assignment: `:=`
- Named argument/default/filter value: `=`
- Arrows: `->`, `=>`
- Arithmetic: `+ - * / %`
- Comparison: `< <= > >= == !=`
- Logical: `and or not`
- Postfix: `.`, `()`, `[]`, `?`

Keywords:
`agent allow and as async await break catch continue deny else enum false fn for if impl import in let match memory model none not or policy prompt pub return spawn tool true try type use while`

-------------------------------------------------------------------------------
3. Block Model (`::`)
-------------------------------------------------------------------------------

Block start:
- A block starts when a header line ends with `::`.

Block end:
- A line containing only `::` ends the current block.

Validation:
- Extra tokens on a block-end line are invalid.
- `::` close without an open block is invalid.
- EOF with unclosed blocks is invalid.

-------------------------------------------------------------------------------
4. Compilation Unit and Items
-------------------------------------------------------------------------------

A module is a sequence of top-level items.

Top-level items:
- `import`
- `native::import`
- `use` / `pub use`
- `fn` / `pub fn`
- `type` / `pub type`
- `enum` / `pub enum`
- `impl`
- `tool`
- `policy`
- `prompt`
- `model`
- `agent`
- top-level statements

Ordering rule:
- `import` and `native::import` must appear before any other top-level item.

-------------------------------------------------------------------------------
5. Modules and Imports
-------------------------------------------------------------------------------

Two import forms are supported:

1) `import` (module import, `::` path)
- Syntax: `import app::utils` or `import app::utils as utils`
- Path separator: `::`

2) `use` (symbol/module use, `.` path)
- Syntax:
  - `use app.utils`
  - `use app.utils as u`
  - `use app.utils::{add, sub}`
  - `pub use app.utils`
  - `pub use app.utils::{add, sub}`
- Path separator: `.`
- `use` list aliases are not supported.

Visibility:
- Only `fn`, `type`, `enum`, and `use` declarations may be `pub`.
- Symbols are private by default.

Project/module layout:
- Entry supports file or project root (`enkai.toml` + `src/main.enk`).
- Module resolution supports file-based modules and local path dependencies.

-------------------------------------------------------------------------------
6. Declarations
-------------------------------------------------------------------------------

Functions:
- `fn name(params) -> Type :: ... ::`
- Parameters may include type annotations and default values.

Types:
- `type Name :: field: Type ::`
- `enum Name :: VariantA VariantB ::`
- `impl Name :: fn method(...) :: ... :: ::`

Native FFI import:
- Top-level only.
- Form:
  - `native::import "library" ::`
  - `    fn symbol(a: Int, b: String) -> Int`
  - `::`
- Native signatures do not allow default parameter values.

Policy/tool/AI declarations:
- `tool path.name(params) -> Type`
- `policy Name :: allow cap.rule ... ::`
- `prompt`, `model`, `agent`, `memory` parse and are part of the language surface.

-------------------------------------------------------------------------------
7. Statements and Expressions
-------------------------------------------------------------------------------

Statements:
- `let` bindings: `let x := expr` or `let x: Type := expr`
- Assignment statement: `target := expr`
- `if` / `else`
- `while`
- `for x in expr`
- `match` statement with `=>` arms
- `try :: ... :: catch err :: ... ::`
- `return`, `break`, `continue`
- expression statement

Expressions:
- Literals, identifiers
- Unary: `not`, unary `-`, `await`, `spawn`
- Binary: arithmetic, comparison, equality, logical `and`/`or`
- Call: `f(...)`, named arguments supported
- Indexing: `a[i]`
- Field: `obj.field`
- Lists: `[a, b, c]`
- Lambda: `(a: Int, b: Int) -> Int => a + b`
- Match expression
- Try postfix: `expr?`

Assignment form:
- `:=` is statement-level assignment/binding.
- `=` is not valid for binding/assignment.

-------------------------------------------------------------------------------
8. Types
-------------------------------------------------------------------------------

Core types used in v0.9.2:
- `Int`, `Float`, `Bool`, `String`, `Void`
- Optional: `T?`
- Function: `fn(T1, T2) -> R`
- Named types with optional generic arguments in syntax

Typechecking behavior (production):
- Function arity/type checks for declared function signatures.
- Operator compatibility checks for arithmetic/logical/comparison usage.
- Return type checks against function signatures.
- Visibility checks for private symbol access across modules.
- Native import signature validation for supported FFI types.

-------------------------------------------------------------------------------
9. Runtime and Execution
-------------------------------------------------------------------------------

Execution pipeline:
- Source -> parser/AST -> typecheck -> bytecode -> VM.

Diagnostics:
- Parse/type/compile diagnostics include line/column and snippets.
- Runtime errors include a lightweight stack trace with source locations.

Formatting and tests:
- Deterministic formatter (`enkai fmt`, `enkai fmt --check`).
- Project test runner (`enkai test`) compiles and executes test files.

-------------------------------------------------------------------------------
10. Built-in Runtime Modules (v0.9.3)
-------------------------------------------------------------------------------

Concurrency:
- `task.spawn(fn0)`
- `task.join(handle)`
- `task.sleep(ms)`

Channels:
- `chan.make()`
- `chan.send(channel, value)`
- `chan.recv(channel)`

TCP networking:
- `net.bind(host, port)`
- listener methods: `accept()`, `port()`, `close()`
- connection methods: `read(n)`, `read_all()`, `write(buf)`, `close()`

HTTP server/client:
- `http.serve(host, port, handler)`
- `http.get(url)`
- `http.post(url, body)`
- response builders: `http.response(status, body)`, `http.ok(body)`, `http.bad_request(body)`, `http.not_found(body)`

JSON:
- `json.parse(text)`
- `json.stringify(value)`

Tokenizer:
- `tokenizer.train(config)`
- `tokenizer.load(path)`
- tokenizer methods: `encode(text)`, `decode(tokens)`, `save(path)`

Dataset streaming:
- `dataset.open(path, tokenizer, config)`
- stream method: `next_batch()`

Checkpointing:
- `checkpoint.save(dir, state)`
- `checkpoint.load(path)`
- `checkpoint.latest(dir)`
- `checkpoint.rotate(dir, keep_last)`

Native-backed std modules:
- `std::fsx`
- `std::zstd`
- `std::hash`

Tensor backend (`std::tensor`, v0.9.2 surface):
- device/tensor creation, math ops, shape/dtype/device transforms
- autograd and optimizer helper APIs
- AMP scaler/autocast APIs
- ranked checkpoint save/load APIs
- backend selection (`torch`/`cpu`) with guarded extern calls
- native training loss entrypoint currently integrated as TinyLM transformer forward + CE loss

Tensor C ABI checkpoint/distributed hooks:
- checkpoint hooks are present (`enkai_checkpoint_save`, `enkai_checkpoint_load`, ranked variants)
- distributed hooks are present (`enkai_dist_*`) and backend-loadable
- actual behavior remains environment-gated by CUDA/NCCL/runtime support

For full tensor C ABI contracts and safety preconditions, see `docs/tensor_api.md` and `docs/gpu_backend.md`.

-------------------------------------------------------------------------------
11. CLI Contract (v0.9.3)
-------------------------------------------------------------------------------

Commands:
- `enkai run <file|dir> [--trace-vm] [--disasm] [--trace-task] [--trace-net]`
- `enkai check <file|dir>`
- `enkai fmt [--check] <file|dir>`
- `enkai test [project_root]`
- `enkai train <config.enk>`
- `enkai eval <config.enk>`

Project entry resolution:
- Running with a directory resolves project root and `src/main.enk`.

-------------------------------------------------------------------------------
12. Known Limits in v0.9.3
-------------------------------------------------------------------------------

The following are intentionally not fully implemented yet:
- `async fn` declarations are rejected by parser.
- `await`/`spawn` unary operators parse, but language-level async/await semantics are not fully implemented.
- AI-native declarations (`tool`/`agent`/`prompt`/`model`/`memory`) exist in syntax, but selected runtime behavior remains stubbed depending on feature path.
- Distributed tensor operations are partial:
  - hook symbols exist and are backend-loadable,
  - single-process/single-rank path is the fully supported baseline,
  - CUDA/NCCL multi-rank behavior is environment-gated and not guaranteed on all targets.
- Current training-forward integration in runtime uses a TinyLM transformer forward/loss path and is not yet a full-scale Transformer stack.
- Engine-level checkpoint helpers exist, but full train-loop orchestration and multi-rank resume policy are constrained to currently integrated paths.
- v0.9.3 validation note:
  - CPU-mode single-device soak has passing evidence.
  - CUDA single-GPU long-soak and distributed (2-GPU/4-GPU) reliability remain
    operator-run requirements and are not auto-proven by repository state alone.

These limits are part of the current stable contract and should be treated as production constraints.

-------------------------------------------------------------------------------
13. Change Control
-------------------------------------------------------------------------------

For any language/runtime surface change after v0.9.2:
1) Update this specification first.
2) Add/adjust compiler/runtime tests.
3) Update changelog and targeted docs (`docs/xx_*.md`, `docs/tensor_api.md`, etc.).
