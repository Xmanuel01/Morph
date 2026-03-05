# Enkai Language Specification (v0.1 -> v1.9.4)

Status: stable.
Grammar and CLI contracts are frozen at the v0.9.3 baseline for the v1.x line.
This document is the normative language and runtime surface for Enkai v1.9.4,
including compatibility constraints carried from v0.1 onward.

-------------------------------------------------------------------------------
1. Scope
-------------------------------------------------------------------------------

This specification covers:
- Core syntax and block rules.
- Module/import semantics.
- Type and expression forms supported by parser, checker, compiler, and VM.
- Built-in runtime modules shipped in v1.9.4.
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
- v1.0: train/eval config schema v1 and checkpoint format v1.
- v1.1: runtime semantics for type/enum/impl and AI declarations; std::nn/loss/optim.
- v1.2: multi-GPU runtime wiring, AMP, grad accumulation/clipping, ranked checkpoints,
  dataset prefetch/metrics, build cache + lockfile, and expanded stdlib.
- v1.3: serving/backend platform primitives (`enkai serve`, routed HTTP + streaming,
  auth/rate-limit middleware, JSONL observability), HTTP client config APIs, TLS
  inspection helper, SQLite/Postgres connectors, and model-registry version pinning hooks.
- v1.4: frontend developer stack (`enkai new` scaffolds, typed SDK generation via
  `enkai sdk`, React/TypeScript reference app contract, and SDK/backend endpoint
  contract tests).
- v1.5: bootstrap-lite toolchain commands (`fmt-lite`, `lint-lite`,
  `tokenizer-lite`, `dataset-lite`), bootstrap runtime module, and deterministic
  parity tests between Rust and Enkai tool paths.
- v1.6: bootstrap-core `litec` command surface, runtime `compiler` module for
  subset parse/check/emit, and stage0/stage1 bytecode equivalence verification.
- v1.7: self-host beta flow via `enkai litec selfhost`, staged bootstrap frontend
  flow via `enkai litec stage`, expanded bootstrap-core subset
  (`use`/`type`/`enum`/`impl` + non-capturing lambda), and self-host CI lane coverage.
- v1.8: compatibility/deprecation policy, legacy config/checkpoint compatibility
  gates, and documented self-host daily workflow + fallback process.
- v1.9-v1.9.4: stage1 execution command (`enkai litec run`), unified master pipeline
  smoke test, and GPU evidence verification scripts for operator-run soak gates.

Compatibility policy:
- `.enk` and `.en` are primary source extensions.
- Legacy compatibility paths may exist in runtime/FFI loaders, but are not the
  primary contract unless listed explicitly.

-------------------------------------------------------------------------------
1.2 Validation Gate Status (v1.9.4)
-------------------------------------------------------------------------------

Current verification status:
- Training path: TinyLM forward + cross-entropy is active in runtime.
- Non-GPU CI/test suite: passing in repository state.
- Single-device soak gate: pending operator run on production hardware.
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
- Dependency resolution prefers `enkai.lock` when present for deterministic builds.

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

Runtime semantics (v1.4):
- `type` emits a constructor function that returns a record with `__type = "Name"` and
  the declared fields.
- `enum` emits a record of variant records. Each variant record includes
  `__type = "Name"` and `__variant = "Variant"`.
- `impl` registers methods on a per-type method table. `obj.method` on a record with
  a matching `__type` returns a bound function with `self` inserted as the first
  argument.

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

Runtime semantics (v1.1):
- `tool` declarations compile to host-invoked functions via `tool.invoke(name, args)`.
  Execution requires explicit host configuration:
  - `ENKAI_TOOL_<PATH>` set to a command JSON array
  - or `ENKAI_TOOL_RUNNER` set to a command JSON array (tool path appended as the final arg).
- Legacy whitespace-split command specs remain available only with
  `ENKAI_TOOL_ALLOW_LEGACY_SPLIT=1`.
- Tool calls are policy-checked through `tool.invoke` capability mapping.
- Tool/policy runtime failures are machine-parseable and prefixed with stable codes:
  - `E_POLICY_DENIED`, `E_POLICY_UNKNOWN`
  - `E_TOOL_CONFIG`, `E_TOOL_SPAWN`, `E_TOOL_TIMEOUT`, `E_TOOL_WAIT`
  - `E_TOOL_IO`, `E_TOOL_PAYLOAD`, `E_TOOL_EXIT`, `E_TOOL_OUTPUT_FORMAT`
- `prompt` compiles to a record:
  `{ __kind: "prompt", name, template, inputs }`.
- `model` compiles to a record:
  `{ __kind: "model", value }`.
- `agent` compiles to a record:
  `{ __kind: "agent", ... }` plus optional `policy_name`, memory entries, and
  any `agent`-scoped functions as fields. Agent body statements are not executed
  by the VM runtime.
- `memory` entries inside agents compile to records:
  `{ __kind: "memory", path | expr }`.
- `policy` declarations compile into `policy.register(name, rules, is_default)` calls and
  are enforced by the VM runtime for native capability checks. Calls to native IO/FS/env/
  process/network/HTTP/checkpoint/dataset/tokenizer APIs require an active policy; otherwise
  they raise a runtime error. `agent` method calls apply the agent's `policy_name` when bound.

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

`await expr` is lowered to `task.join(expr)`. `spawn expr` is lowered to
`task.spawn(expr)`.

Assignment form:
- `:=` is statement-level assignment/binding.
- `=` is not valid for binding/assignment.

-------------------------------------------------------------------------------
8. Types
-------------------------------------------------------------------------------

Core types used in v1.9.4:
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
10. Built-in Runtime Modules (v1.9.4)
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
- `http.serve_with(host, port, routes, config)`
- `http.route(method, path, handler)` (`:param` segments supported)
- `http.middleware(name, config)` with kinds:
  - `auth` (token auth + tenant mapping)
  - `rate_limit` (token-bucket; key by IP or token)
  - `jsonl_log` (structured request logs)
  - `default` (fallback route handler)
- `http.get(url)`
- `http.post(url, body)`
- `http.request(config)` (`method`, `url`, `headers`, `timeout_ms`, `retries`, `retry_backoff_ms`)
- request helpers: `http.header(req, name)`, `http.query(req, name)`
- streaming helpers: `http.stream_open(status, headers)`, `http.stream_send(stream, chunk)`, `http.stream_close(stream)`
- response builders: `http.response(status, body)`, `http.ok(body)`, `http.bad_request(body)`, `http.not_found(body)`
- default error responses are JSON with `error.code` and `error.message`; runtime headers include
  request-id metadata for observability.

JSON:
- `json.parse(text)`
- `json.stringify(value)`

Bootstrap:
- `bootstrap.format(source)` -> canonical formatted source text
- `bootstrap.check(source)` -> `Bool`
- `bootstrap.lint(source)` -> issue list records (`line`, `code`, `message`)
- `bootstrap.lint_count(source)` -> `Int`
- `bootstrap.lint_json(file, source)` -> JSON string report

Compiler (bootstrap-core subset):
- `compiler.parse_subset(source)` -> summary record (`items`, `functions`)
- `compiler.check_subset(source)` -> `Bool`
- `compiler.emit_subset(source, output_path)` -> `Bool` and writes bytecode program

Tokenizer:
- `tokenizer.train(config)`
- `tokenizer.load(path)`
- `tokenizer.save(tokenizer, path)`
- `tokenizer.encode(tokenizer, text)`
- `tokenizer.decode(tokenizer, tokens)`
- tokenizer methods: `encode(text)`, `decode(tokens)`, `save(path)`
  - `tokenizer.train` config supports optional `seed` for deterministic tie-breaks.

Dataset streaming:
- `dataset.open(path, tokenizer, config)`
- `dataset.next_batch(stream)`
- stream method: `next_batch()`
  - dataset config supports optional `seed`, `shuffle`, and `prefetch_batches` for deterministic
    file order and background prefetch.
  - batch records include `token_count` and `packing_efficiency` for throughput metrics.

Checkpointing:
- `checkpoint.save(dir, state)`
- `checkpoint.load(path)`
- `checkpoint.latest(dir)`
- `checkpoint.rotate(dir, keep_last)`

Native-backed std modules:
- `std::fsx`
- `std::zstd`
- `std::hash`
- `std::http`
- `std::nn` (core ML layers)
- `std::loss` (loss functions)
- `std::optim` (optimizer helpers)
- `std::env`
- `std::path`
- `std::time`
- `std::log`
- `std::io`
- `std::process`
- `std::db` (SQLite + Postgres connectors)
- `std::tls` (TLS peer certificate fingerprint helper)
- `std::model_registry` (serve-time env contract helpers)

Tensor backend (`std::tensor`, v1.9.4 surface):
- device/tensor creation, math ops, shape/dtype/device transforms
- autograd and optimizer helper APIs
- AMP scaler/autocast APIs
- ranked checkpoint save/load APIs
- backend selection (`torch`/`cpu`) with guarded extern calls
- native training loss entrypoint currently integrated as TinyLM transformer forward + CE loss

Tensor C ABI checkpoint/distributed hooks:
- checkpoint hooks are present (`enkai_checkpoint_save`, `enkai_checkpoint_load`, ranked variants)
- distributed hooks (`enkai_dist_init`, `enkai_dist_allreduce_sum_multi`) are wired and invoked
  when `world_size > 1`; behavior remains environment-gated by CUDA/NCCL/runtime support
  and explicit opt-in via `ENKAI_ENABLE_DIST=1`

For full tensor C ABI contracts and safety preconditions, see `docs/tensor_api.md` and `docs/gpu_backend.md`.

-------------------------------------------------------------------------------
11. CLI Contract (v1.9.4)
-------------------------------------------------------------------------------

Commands:
- `enkai run <file|dir> [--trace-vm] [--disasm] [--trace-task] [--trace-net]`
- `enkai serve [--host <host>] [--port <port>] [--registry <dir> --model <name> [--model-version <v>|--latest] | --checkpoint <path>] [--trace-vm] [--disasm] [--trace-task] [--trace-net] [file|dir]`
- `enkai new <backend|frontend-chat|fullstack-chat> <target_dir> [--api-version <v>] [--backend-url <url>] [--force]`
- `enkai sdk generate <output_file> [--api-version <v>]`
- `enkai check <file|dir>`
- `enkai fmt [--check] <file|dir>`
- `enkai fmt-lite [--check] <file|dir>`
- `enkai lint-lite [--deny-warn] <file|dir>`
- `enkai tokenizer-lite train <dataset_path> <tokenizer_path> [--vocab-size <n>] [--min-freq <n>] [--seed <n>] [--lowercase]`
- `enkai dataset-lite inspect <dataset_path> <tokenizer_path> --seq-len <n> --batch-size <n> [--seed <n>] [--shuffle] [--drop-remainder|--keep-remainder] [--no-add-eos] [--prefetch-batches <n>] [--output <path>]`
- `enkai litec check <input.enk>`
- `enkai litec compile <input.enk> --out <program.bin>`
- `enkai litec verify <input.enk>`
- `enkai litec run <input.enk>`
- `enkai litec stage <parse|check|codegen> <input.enk> [--out <program.bin>]`
- `enkai litec selfhost <corpus_dir>`
- `enkai litec selfhost-ci <corpus_dir> [--no-compare-stage0]`
- `enkai litec replace-check <corpus_dir> [--no-compare-stage0]`
- `enkai build [dir]`
- `enkai test [project_root]`
- `enkai train <config.enk>`
- `enkai eval <config.enk>`
- `enkai migrate config-v1 <in_config.enk> <out_config.enk|out.json>`
- `enkai migrate checkpoint-meta-v1 <checkpoint_dir> [--dry-run] [--verify]`
- `enkai doctor [path]`

Serve model-selection contract:
- `enkai serve` resolves model paths from either:
  - explicit `--checkpoint <path>`, or
  - registry tuple `--registry <dir> --model <name>` with `--model-version <v>` or `--latest`.
- resolved values are exported for program/runtime consumption:
  - `ENKAI_SERVE_MODEL_PATH`
  - `ENKAI_SERVE_MODEL_NAME`
  - `ENKAI_SERVE_MODEL_VERSION`
  - `ENKAI_SERVE_MODEL_REGISTRY`

Frontend scaffolding + SDK contract:
- `enkai new backend` creates an Enkai HTTP backend scaffold with versioned routes:
  - `GET /api/<version>/health`
  - `POST /api/<version>/chat`
  - `GET /api/<version>/chat/stream`
- `enkai new frontend-chat` creates React/TypeScript UI scaffolding with:
  - typed SDK (`src/sdk/enkaiClient.ts`)
  - env contract (`VITE_ENKAI_API_BASE_URL`, `VITE_ENKAI_API_VERSION`, `VITE_ENKAI_API_TOKEN`)
  - streaming chat UI and error handling conventions.
- `enkai new fullstack-chat` emits both backend and frontend scaffolds with aligned API version defaults.
- Generated SDK pins `x-enkai-api-version` and path prefix `/api/<version>`.

Project entry resolution:
- Running with a directory resolves project root and `src/main.enk`.

Build caching and lockfile:
- `enkai build` resolves dependencies and writes `enkai.lock`.
- Build cache lives under `target/enkai/` and is reused by `enkai run` when valid.

Train/Eval config schema:
- v1 config requires `config_version: 1` and the mandatory fields listed in
  `docs/25_train_eval_cli.md`.
- Optional v1.2+ fields include `world_size`, `rank`, `grad_accum_steps`, `grad_clip_norm`,
  `amp { enabled, dtype, init_scale, growth_factor, backoff_factor, growth_interval }`,
  `shuffle`, and `prefetch_batches`.
- Legacy configs without `config_version` are accepted with a runtime warning; planned
  removal target is v2.0.

Checkpoint format:
- v1 checkpoints include `format_version: 1` in `meta.json`.
- Ranked checkpoints write `rank{n}/` subdirectories and a `manifest.json` with `world_size`.
- Migration tooling:
  - `enkai migrate config-v1` emits canonical v1 config with `config_version: 1`.
  - `enkai migrate checkpoint-meta-v1` upgrades/validates checkpoint `meta.json` files and
    supports `--dry-run` and `--verify`.
  - `enkai doctor` scans configs/checkpoints for v2.0 strict-contract blockers.

-------------------------------------------------------------------------------
12. Known Limits in v1.9.4
-------------------------------------------------------------------------------

The following are intentionally not fully implemented yet:
- The bytecode VM is the normative production runtime. The legacy tree-walk
  interpreter module is retained for compatibility/debugging only and is not
  kept at full feature parity with VM semantics.
- `await`/`spawn` compile to task operations but do not provide a structured async runtime beyond the existing task model.
- `async fn` declarations are accepted and compile as regular functions with task-based async primitives.
- AI-native tool declarations compile to `tool.invoke` host calls. Execution requires explicit
  host configuration (`ENKAI_TOOL_<PATH>` or `ENKAI_TOOL_RUNNER`) and policy allow rules.
- Distributed tensor operations are environment-gated:
  - multi-rank init/allreduce paths are implemented for `enkai_tensor` builds with `torch,dist`,
  - rank-device mapping and explicit opt-in (`ENKAI_ENABLE_DIST=1`) are enforced,
  - operator-run GPU soak evidence remains required for production sign-off on specific hardware.
- HTTP serving supports routed handlers, chunked streaming, and server-side WebSocket
  upgrade + send/recv/close APIs.
- `std::db` ships SQLite in-tree and includes Postgres connector functions (`pg_open/pg_exec/pg_query/pg_close`)
  through `enkai_native` using direct `NoTls` connections.
- Model registry support is filesystem-based (`--registry` directory scanning). Remote registries and artifact pull/auth flows are out of scope in v1.x.
- Frontend scaffolds target React + TypeScript web projects; non-web/mobile generators are not part of the current v1.x scope.
- Current training-forward integration in runtime uses a TinyLM transformer forward/loss path and is not yet a full-scale Transformer stack.
- Engine-level checkpoint helpers exist, but full train-loop orchestration and multi-rank resume policy are constrained to currently integrated paths.
- Training metrics include best-effort GPU memory/utilization sampling via `nvidia-smi` for CUDA devices.
  On hosts without `nvidia-smi` or compatible drivers these fields remain `null`.
- `enkai litec` is intentionally subset-scoped; unsupported subset constructs
  (for/match/try/break/continue and `match` expressions) are rejected by
  bootstrap-core validation.
- Lambda expressions compile as non-capturing function values; references to
  outer locals from lambda bodies are not supported in the current bytecode model.
- `enkai litec selfhost` verifies bytecode equivalence for the expanded subset, but
  does not yet replace the Rust Stage0 compiler in release builds.
- `enkai litec selfhost-ci` runs subset corpus compile/execute parity and optional
  stage0 result comparison; it does not yet build full production binaries from
  an Enkai-compiled compiler artifact.
- `enkai litec replace-check` validates stage0/stage1/stage2 corpus compilation/runtime
  equivalence and reports compiler fixed-point status for the bootstrap subset; it is not
  yet a full replacement for Rust Stage0 compiler releases.
- v1.9.4 validation note:
  - CPU-mode single-device soak requires operator-run evidence on production hardware.
  - CUDA single-GPU long-soak and distributed (2-GPU/4-GPU) reliability remain
    operator-run requirements and are not auto-proven by repository state alone.
  - Operator logs can be verified with `scripts/verify_gpu_gates.ps1` or
    `scripts/verify_gpu_gates.sh`.

These limits are part of the current stable contract and should be treated as production constraints.

-------------------------------------------------------------------------------
13. Change Control
-------------------------------------------------------------------------------

For any language/runtime surface change after v1.9.4:
1) Implement the change and add/adjust compiler/runtime tests.
2) Update this specification to match the shipped behavior.
3) Update changelog and targeted docs (`docs/xx_*.md`, `docs/tensor_api.md`, etc.).
4) If compatibility/deprecation behavior changes, update `docs/29_compatibility_policy.md`.




