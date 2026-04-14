# Enkai Language Specification (v0.1 -> v3.1.2)

Status: stable.
Grammar and CLI contracts are frozen at the v0.9.3 baseline for the v1.x/v2.x line.
This document is the normative language and runtime surface for Enkai v3.0.0,
including compatibility constraints carried from v0.1 onward.

-------------------------------------------------------------------------------
1. Scope
-------------------------------------------------------------------------------

This specification covers:
- Core syntax and block rules.
- Module/import semantics.
- Type and expression forms supported by parser, checker, compiler, and VM.
- Built-in runtime modules shipped in v3.0.0.
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
- v1.9-v3.0.0: stage1 execution command (`enkai litec run`), unified master pipeline smoke test, GPU evidence verification scripts for operator-run soak gates, frontend/serve contract snapshot freeze with persisted conversation schema v1, deterministic packaging/checksum/SBOM release gates, RC evidence-archive tooling, capability-complete release reporting from archived evidence manifests, additive multi-node orchestration controls (`dist.topology`, `dist.rendezvous`, `dist.retry_budget`, `dist.device_map`) plus `enkai cluster validate|plan|run`, additive readiness filtering via `--skip-check <id>`, archived simulation std/runtime plus native FFI smoke evidence for the `enkai sim` command line, coroutine-facing `std::sim` APIs, Adam-0 smoke/profile evidence for the 100-agent deterministic baseline, the bounded Adam-0 reference suite with archived 100 / 1000 / 10000 agent evidence, and signed registry convergence across checkpoint, simulation, environment, and native-extension artifacts.

Compatibility policy:
- `.enk` and `.en` are primary source extensions.
- Legacy compatibility paths may exist in runtime/FFI loaders, but are not the
  primary contract unless listed explicitly.

-------------------------------------------------------------------------------
1.2 Validation Gate Status (v3.0.0)
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
- Full-platform non-hardware readiness additionally requires simulation CLI smoke
  evidence through:
  - `artifacts/readiness/sim_smoke.json`
  - `artifacts/readiness/sim_evidence_verify.json`
  - `artifacts/readiness/sim_native_smoke.json`
  - `artifacts/readiness/sim_native_evidence_verify.json`
  - `artifacts/readiness/sim_stdlib_smoke.json`
  - `artifacts/readiness/sim_stdlib_evidence_verify.json`
  - `artifacts/sim/smoke_run.json`
  - `artifacts/sim/smoke_profile.json`
  - `artifacts/sim/smoke_replay.json`
  - `artifacts/sim/native_smoke_run.json`
  - `artifacts/sim/native_smoke_profile.json`
  - `artifacts/sim/stdlib_smoke_run.json`
  - `artifacts/sim/stdlib_smoke_profile.json`
- Full-platform non-hardware readiness also requires proof-grade CPU validation
  evidence through:
  - `artifacts/validation/ffi_correctness.json`
  - `artifacts/validation/determinism_event_queue.json`
  - `artifacts/validation/determinism_sim_replay.json`
  - `artifacts/validation/pool_safety.json`
  - `artifacts/validation/ffi_safety.json`
  - `artifacts/validation/adam0_fake10.json`
  - `artifacts/validation/adam0_ref100.json`
  - `artifacts/validation/perf_ffi_noop.json`
  - `artifacts/validation/perf_sparse_dot.json`
  - `artifacts/validation/perf_adam0_reference_100.json`
  - `artifacts/validation/adam0_stress1000.json`
  - `artifacts/validation/adam0_target10000.json`
  - `artifacts/validation/perf_adam0_reference_1000.json`
  - `artifacts/validation/perf_adam0_reference_10000.json`

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
- Supported FFI scalar/slice types:
  - `Int`, `Float`, `Bool`, `String`, `Buffer`, `Handle`, `Void`
  - optional FFI types are limited to `String?`, `Buffer?`, and `Handle?`
- `Handle` is an opaque native pointer value owned by the VM/runtime.
- If a library exports either `enkai_abi_version` or `enkai_symbol_table`, it must export both.
- Official native library ABI policy:
  - `enkai_abi_version() -> Int`
  - `enkai_symbol_table() -> String`
  - `enkai_symbol_table` returns JSON with `abi_version` and `exports`
  - `["*"]` is accepted, but exact exported symbol lists are preferred for new native modules
- Libraries returning `String` or `Buffer` must export `enkai_free(ptr, len)`.
- Libraries returning `Handle` must export `enkai_handle_free(ptr)`.

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

Core types used in v3.0.0:
- `Int`, `Float`, `Bool`, `String`, `Buffer`, `Handle`, `SimCoroutine`, `Void`
- Optional: `T?`
- Function: `fn(T1, T2) -> R`
- Named types with optional generic arguments in syntax

Typechecking behavior (production):
- Function arity/type checks for declared function signatures.
- Operator compatibility checks for arithmetic/logical/comparison usage.
- Return type checks against function signatures.
- Visibility checks for private symbol access across modules.
- Native import signature validation for supported FFI types.
- FFI diagnostics use stable runtime codes for:
  - library load failures
  - symbol resolution failures
  - ABI/version or symbol-table failures
  - invalid argument/return type usage
  - missing `enkai_free` / `enkai_handle_free`

-------------------------------------------------------------------------------
9. Runtime and Execution
-------------------------------------------------------------------------------

Execution pipeline:
- Source -> parser/AST -> typecheck -> bytecode -> VM.

Diagnostics:
- Parse/type/compile diagnostics include line/column and snippets.
- Runtime errors include a lightweight stack trace with source locations.
- VM benchmark profile artifacts include FFI-native diagnostics:
  - native call count
  - marshaled bytes in/out
  - marshal/copy operation count
  - native handle object count
  - native-call time vs VM execution time

Formatting and tests:
- Deterministic formatter (`enkai fmt`, `enkai fmt --check`).
- Project test runner (`enkai test`) compiles and executes test files.

-------------------------------------------------------------------------------
10. Built-in Runtime Modules (v3.0.0)
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
  - `rate_limit` (token-bucket; key by `ip`, `token`, `tenant`, `model`, or `tenant_model`)
  - `backpressure` (`max_inflight` cap; deterministic `503 backpressure_overloaded` when exceeded)
  - `jsonl_log` (structured request logs)
  - `default` (fallback route handler)
- `http.get(url)`
- `http.post(url, body)`
- `http.request(config)` (`method`, `url`, `headers`, `timeout_ms`, `retries`, `retry_backoff_ms`)
- request helpers: `http.header(req, name)`, `http.query(req, name)`
- streaming helpers: `http.stream_open(status, headers)`, `http.stream_send(stream, chunk)`, `http.stream_close(stream)`
- response builders: `http.response(status, body)`, `http.ok(body)`, `http.bad_request(body)`, `http.not_found(body)`
- default error responses are JSON with `error.code` and `error.message`; runtime headers include:
  - `x-enkai-request-id`
  - `x-enkai-correlation-id`
  - `x-enkai-queue-ms`
  - `x-enkai-latency-ms`
  - `x-enkai-inflight`
  - `x-enkai-tenant` (when auth maps tenant)
  - `x-enkai-model-name`, `x-enkai-model-version`, `x-enkai-model-registry` (when configured)
  - `x-enkai-error-code` (machine-parseable deterministic error code)

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
- `compiler.describe_subset(source)` -> structural subset record used by the self-host bootstrap validator
- `compiler.check_subset(source)` -> `Bool`
- `compiler.emit_subset(source, output_path)` -> `Bool` and writes bytecode program
- `compiler.parse_subset_file(path)` -> summary record (`items`, `functions`) with package-aware import resolution
- `compiler.describe_subset_file(path)` -> structural subset record with package-aware import resolution
- `compiler.describe_subset_package_file(path)` -> package/module structural record with package-aware import resolution
- `compiler.describe_program_file(path)` -> structural summary of an emitted bytecode program
- `compiler.check_subset_file(path)` -> `Bool` using the source file's package context
- `compiler.check_subset_raw(source)` -> `Bool` using the compiler/typechecker path without Rust-owned subset validation
- `compiler.check_subset_raw_file(path)` -> `Bool` using package context without Rust-owned subset validation
- `compiler.emit_subset_file(path, output_path)` -> `Bool` and writes bytecode program while preserving package-root imports
- `compiler.emit_subset_raw(source, output_path)` -> `Bool` and writes bytecode program without Rust-owned subset validation
- `compiler.emit_subset_raw_file(path, output_path)` -> `Bool` and writes bytecode program while preserving package-root imports and relying on self-host subset validation

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
- `std::sparse`
- `std::event`
- `std::pool`
- `std::sim`
- `std::spatial`
- `std::snn`
- `std::agent`
- `std::env`
- `std::path`
- `std::time`
- `std::log`
- `std::io`
- `std::process`
- `std::db` (SQLite + Postgres connectors)
- `std::tls` (TLS peer certificate fingerprint helper)
- `std::model_registry` (serve-time env contract helpers)
- `std::analysis` (CSV/JSONL ingest, typed schema inference/validation, filter/project/join/group aggregates, describe/histogram/quantiles/rolling stats, deterministic pipeline manifests)
- `std::algo` (sorting/searching/path, priority/merge/stream helpers, ML metric/eval/scheduler helpers)

`std::sparse`, `std::event`, and `std::pool` keep stable Enkai-level APIs while using
native-backed acceleration through `enkai_native` when available. When native acceleration
is unavailable or `ENKAI_SIM_ACCEL=0`, they fall back to deterministic VM/runtime behavior.

`std::sparse` semantics:
- negative row/column/vector indices are rejected
- setting a sparse entry to `0.0` removes it
- `get(...)` / `get_vector(...)` return `none` for missing entries
- `nonzero(...)` and `nonzero_vector(...)` are deterministic and sorted by key order
- `dot(...)` / `matvec(...)` treat missing dense entries as zero rather than reading out of bounds

`std::event` semantics:
- ordering is strictly `(time ASC, insertion_seq ASC)`
- `peek(...)` is non-destructive
- `len(...)` / `is_empty(...)` reflect the same queue state in native-backed and VM-fallback modes

`std::pool` semantics:
- fixed pools drop releases when full and increment `dropped_on_full`
- growable pools expand capacity deterministically
- `reset(...)` clears available items but preserves accumulated counters for auditability
- `stats(...)` exposes `available`, `capacity`, `growable`, `acquire_hits`, `acquire_misses`,
  `releases`, `dropped_on_full`, and `high_watermark`

`std::sim` additive APIs (v3.0.0):
- world/scheduler helpers:
  - `make()`
  - `make_seeded(seed)`
  - `time(world)`
  - `seed(world)`
  - `pending(world)`
  - `schedule(world, time, event)`
  - `step(world)`
  - `run(world, max_steps)`
  - `snapshot(world)`
  - `restore(snapshot)`
  - `replay(snapshot, steps)`
  - `log(world)`
  - `entity_set(world, id, value)`
  - `entity_get(world, id)`
  - `entity_remove(world, id)`
  - `entity_ids(world)`
- coroutine-facing helpers:
  - `coroutine(world, fn)`
  - `coroutine_with(world, fn, state)`
  - `coroutine_args(world, fn, args)`
  - `world(coroutine)`
  - `state(coroutine)`
  - `emit(coroutine, value)`
  - `next(coroutine)`
  - `join(coroutine)`
  - `done(coroutine)`

`std::spatial` additive APIs (v3.0.0):
- `make()`
- `upsert(index, id, x, y)`
- `remove(index, id)`
- `radius(index, x, y, radius)`
- `nearest(index, x, y)`
- `occupancy(index, min_x, min_y, max_x, max_y)`

`std::snn` additive APIs (v3.0.0):
- `make(neuron_count)`
- `connect(network, from, to, weight)`
- `set_potential(network, neuron, value)`
- `get_potential(network, neuron)`
- `set_threshold(network, neuron, value)`
- `get_threshold(network, neuron)`
- `set_decay(network, value)`
- `get_decay(network)`
- `step(network, inputs)`
- `spikes(network)`
- `potentials(network)`
- `synapses(network)`

`std::agent` additive APIs (v3.0.0):
- `make(world, spatial_index)`
- `register(env, agent_id, body, memory, x, y)`
- `state(env, agent_id)`
- `body(env, agent_id)`
- `memory(env, agent_id)`
- `set_body(env, agent_id, value)`
- `set_memory(env, agent_id, value)`
- `position(env, agent_id)`
- `set_position(env, agent_id, x, y)`
- `neighbors(env, agent_id, radius)`
- `reward_add(env, agent_id, delta)`
- `reward_get(env, agent_id)`
- `reward_take(env, agent_id)`
- `sense_push(env, agent_id, value)`
- `sense_take(env, agent_id)`
- `action_push(env, agent_id, value)`
- `action_take(env, agent_id)`
- `stream(env, agent_id, domain)`
- `next_float(stream)`
- `next_int(stream, upper)`

`std::analysis` additive APIs (v3.0.0):
- `read_csv(path, delimiter, has_header)`
- `read_jsonl(path)`
- `infer_schema(rows)`
- `infer_schema_typed(rows)`
- `validate_schema(rows, schema)`
- `filter_eq(rows, field, value)`
- `project(rows, columns)`
- `group_sum(rows, key, field)`
- `group_agg(rows, key, field, agg)`
- `join(left_rows, right_rows, left_key, right_key, how)` where `how` is `inner|left|right|outer`
- `describe(values)`
- `histogram(values, bins)`
- `quantiles(values, quantiles)`
- `rolling_mean(values, window)`
- `run_pipeline(rows, pipeline)` -> `{ rows, manifest }` with deterministic stage stats + hashes

`std::algo` additive APIs (v3.0.0):
- software/algorithm primitives:
  - `sort_ints(values)`
  - `binary_search_ints(values, target)`
  - `top_k_ints(values, k)`
  - `merge_sorted_ints(left, right)`
  - `shortest_path(edges, start, goal)`
  - `count_frequencies(values)`
  - `merge_count_maps(left, right)`
  - `window_sum(values, window)`
  - `cumulative_sum(values)`
  - `window_mean(values, window)`
- ML utility primitives:
  - `accuracy(pred, target)`
  - `mse(pred, target)`
  - `mae(pred, target)`
  - `rmse(pred, target)`
  - `precision_recall_f1(pred, target, positive_label)`
  - `split_indices(total, test_ratio, seed, shuffle)`
  - `scheduler_linear_warmup(step, total_steps, warmup_steps, base_lr, min_lr)`

Tensor backend (`std::tensor`, v3.0.0 surface):
- device/tensor creation, math ops, shape/dtype/device transforms
- autograd and optimizer helper APIs
- AMP scaler/autocast APIs
- ranked checkpoint save/load APIs
- backend selection (`torch`/`cpu`) with guarded extern calls
- native training loss entrypoint currently integrated as TinyLM transformer forward + CE loss

Tensor C ABI checkpoint/distributed hooks:
- checkpoint hooks are present (`enkai_checkpoint_save`, `enkai_checkpoint_load`, ranked variants)
- distributed hooks (`enkai_dist_config`, `enkai_dist_init`, `enkai_dist_allreduce_sum_multi`) are wired and invoked
  when `world_size > 1`; behavior remains environment-gated by CUDA/NCCL/runtime support
  and explicit opt-in via `ENKAI_ENABLE_DIST=1`.
- runtime distributed error contracts are machine-parseable (`E_DIST_*`) for init/env/symbol/
  mapping failures and retry exhaustion paths.

For full tensor C ABI contracts and safety preconditions, see `docs/tensor_api.md` and `docs/gpu_backend.md`.

-------------------------------------------------------------------------------
11. CLI Contract (v3.0.0)
-------------------------------------------------------------------------------

Commands:
- `enkai run <file|dir> [--trace-vm] [--disasm] [--trace-task] [--trace-net]`
- `enkai sim run [--trace-vm] [--disasm] [--trace-task] [--trace-net] [--json] [--output <file>] [--lineage-output <file>] [--snapshot-manifest-output <file>] <file|dir>`
- `enkai sim profile [--trace-vm] [--disasm] [--trace-task] [--trace-net] [--case <id>] --output <file> [--lineage-output <file>] [--snapshot-manifest-output <file>] <file|dir>`
- `enkai sim replay --snapshot <file> --steps <n> [--json] [--output <file>] [--lineage-output <file>] [--snapshot-manifest-output <file>]`
- `enkai validate ffi-correctness [--json] [--output <file>]`
- `enkai validate ffi-safety [--json] [--output <file>]`
- `enkai validate determinism --suite <event_queue|sim_replay> [--runs <n>] [--json] [--output <file>]`
- `enkai validate perf-baseline --suite <ffi_noop|sparse_dot|adam0_reference_100|adam0_reference_1000|adam0_reference_10000> [--json] [--output <file>]`
- `enkai validate pool-safety [--json] [--output <file>]`
- `enkai validate adam0-cpu --scenario <fake10|ref100|stress1000|target10000> [--json] [--output <file>]`
- `enkai bench run [--suite <name>] [--baseline <python|none>] [--output <file>] [--machine-profile <file>] [--iterations <n>] [--warmup <n>] [--target-speedup <pct>] [--target-memory <pct>] [--enforce-target] [--enforce-all-cases] [--python <command>] [--enkai-bin <path>]`
- `enkai readiness check [--profile production|full_platform] [--json] [--output <file>] [--skip-check <id>]`
- `enkai readiness verify-blockers --profile full_platform --report <file> [--json] [--output <file>] [--require-gpu-evidence] [--skip-release-evidence] [--version <x.y.z>]`
- `enkai cluster validate <config.enk> [--json]`
- `enkai cluster plan <config.enk> [--json]`
- `enkai cluster run <config.enk> [--dry-run] [--json]`
- `enkai serve [--host <host>] [--port <port>] [--registry <dir> --model <name> [--model-version <v>|--latest] [--require-loaded] | --multi-model --registry <dir> | --checkpoint <path>] [--trace-vm] [--disasm] [--trace-task] [--trace-net] [file|dir]`
- `enkai model register <registry_dir> <name> <version> <artifact_path> [--activate] [--artifact-kind <checkpoint|simulation|environment|native-extension>] [--artifact-manifest <file>] [--lineage-manifest <file>]`
- `enkai model list <registry_dir> [name] [--json]`
- `enkai model load <registry_dir> <name> <version>`
- `enkai model unload <registry_dir> <name> <version>`
- `enkai model loaded <registry_dir> [name] [--json]`
- `enkai model push <registry_dir> <name> <version> --registry <remote_registry_dir> [--sign]`
- `enkai model pull <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
- `enkai model verify-signature <registry_dir> <name> <version> --registry <remote_registry_dir>`
- `enkai model promote-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
- `enkai model retire-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
- `enkai model rollback-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
- `enkai model promote <registry_dir> <name> <version>`
- `enkai model retire <registry_dir> <name> <version>`
- `enkai model rollback <registry_dir> <name> <version>`
- `enkai new <backend|service|llm-backend|frontend-chat|fullstack-chat|llm-fullstack> <target_dir> [--api-version <v>] [--backend-url <url>] [--force]`
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
- `enkai litec selfhost-ci <corpus_dir> [--no-compare-stage0] [--triage-dir <dir>]`
- `enkai litec replace-check <corpus_dir> [--no-compare-stage0] [--triage-dir <dir>]`
- `enkai litec mainline-ci <corpus_dir> [--triage-dir <dir>]`
- `enkai litec release-ci <corpus_dir> [--triage-dir <dir>]`
- `enkai deploy validate <project_dir> --profile <backend|fullstack> --strict [--json] [--output <file>]`
- `enkai build [dir]`
- `enkai test [project_root]`
- `enkai train <config.enk> [--strict-contracts|--lenient-contracts]`
- `enkai pretrain <config.enk> [--strict-contracts|--lenient-contracts]`
- `enkai eval <config.enk> [--strict-contracts|--lenient-contracts]`
- `enkai migrate config-v1 <in_config.enk> <out_config.enk|out.json>`
- `enkai migrate checkpoint-meta-v1 <checkpoint_dir> [--dry-run] [--verify] [--strict-contracts]`
- `enkai doctor [path] [--json] [--strict-contracts|--lenient]`

Contract enforcement note:
- In v3.0.0, train/eval run strict contracts by default.
- `--lenient-contracts` requires `ENKAI_ALLOW_LEGACY_CONTRACTS=1`.

Serve model-selection contract:
- `enkai serve` resolves model paths from either:
  - explicit `--checkpoint <path>`, or
  - registry tuple `--registry <dir> --model <name>` with `--model-version <v>` or `--latest`,
  - multi-model registry mode `--multi-model --registry <dir>` with per-request model selector headers.
- resolved values are exported for program/runtime consumption:
  - `ENKAI_SERVE_MODEL_PATH`
  - `ENKAI_SERVE_MODEL_NAME`
  - `ENKAI_SERVE_MODEL_VERSION`
  - `ENKAI_SERVE_MODEL_REGISTRY`
  - `ENKAI_SERVE_MULTI_MODEL` (`1|true|yes|on` enables per-request model selector mode)
  - `ENKAI_REQUIRE_MODEL_VERSION_HEADER` (`1|true|yes|on` to require `x-enkai-model-version`)
  - `ENKAI_HTTP_MAX_INFLIGHT` (`Int >= 0`; `0` disables cap)
- deterministic model pin checks in HTTP runtime:
  - missing version header -> `400 missing_model_version`
  - version mismatch -> `409 model_version_mismatch`
  - model-name mismatch (when supplied) -> `409 model_name_mismatch`
  - multi-model selector missing -> `400 missing_model_selector`
  - multi-model requested version not loaded -> `409 model_not_loaded`

Frontend scaffolding + SDK contract:
- `enkai new backend` creates an Enkai HTTP backend scaffold with versioned routes:
  - `GET /api/<version>/health`
  - `GET /api/<version>/ready`
  - `POST /api/<version>/chat`
  - `GET /api/<version>/chat/stream`
  - `GET /api/<version>/chat/ws`
  - backend contract snapshots:
    - `contracts/backend_api.snapshot.json`
    - `contracts/conversation_state.schema.json`
    - `contracts/deploy_env.snapshot.json`
  - backend env/migration assets:
    - `.env.example`
    - `scripts/validate_env_contract.py`
    - `migrations/001_conversation_state.sql`
    - `migrations/002_conversation_state_index.sql`
- `enkai new service` and `enkai new llm-backend` emit backend-profile variants:
  - `ENKAI_APP_PROFILE` pin in `.env.example`
  - `llm-backend` additionally requires `ENKAI_MODEL_NAME` and `ENKAI_MODEL_VERSION`
- `enkai new frontend-chat` creates React/TypeScript UI scaffolding with:
  - typed SDK (`src/sdk/enkaiClient.ts`)
  - SDK contract snapshot (`contracts/sdk_api.snapshot.json`)
  - env contract (`VITE_ENKAI_API_BASE_URL`, `VITE_ENKAI_API_VERSION`, `VITE_ENKAI_API_TOKEN`)
  - streaming chat UI and error handling conventions (SSE + WebSocket SDK support).
- `enkai new fullstack-chat` emits both backend and frontend scaffolds with aligned API version defaults.
- `enkai new llm-fullstack` emits `llm-backend` + frontend scaffolds with aligned API version defaults.
- Generated SDK pins `x-enkai-api-version` and path prefix `/api/<version>`.
- `enkai deploy validate` supports machine-readable validation output via `--json` and `--output <file>`.
- Deploy validation enforces additive contract checks for:
  - migration sequence/content under `migrations/`,
  - Docker/systemd deploy assets against required env keys,
  - frontend scaffold package/SDK fragments for fullstack projects.
- Backend scaffold persistence contract is schema-versioned (`schema_version: 1`) and includes
  startup migration of legacy `conversation_state.json` data without `schema_version`.
- Backend scaffold durability contract writes both:
  - `conversation_state.json`
  - `conversation_state.backup.json`

Project entry resolution:
- Running with a directory resolves project root and `src/main.enk`.

Build caching and lockfile:
- `enkai build` resolves dependencies and writes `enkai.lock`.
- Build cache lives under `target/enkai/` and is reused by `enkai run` when valid.

Benchmarking:
- `enkai bench run` executes deterministic suites from `bench/suites/*.json`.
- Official v3.0.0 bounded claim suite: `official_v2_3_0_matrix`.
- `--enforce-target` validates suite-level median targets.
- `--enforce-all-cases` additionally requires every individual case target to pass.
- Baseline comparisons are bounded to pinned suite/machine profiles and emit structured JSON reports.
- Official performance claim thresholds are represented by `--target-speedup` and `--target-memory`,
  with optional hard gating via `--enforce-target`.

Readiness filtering:
- `--skip-check <id>` omits named manifest checks from a readiness run.
- Unknown check ids are rejected deterministically before gate execution begins.
- Release pipelines use `--skip-check` only to avoid duplicate execution of stronger gates
  that are run separately in the same pipeline.
- The full-platform readiness profile includes simulation smoke execution via
  `scripts/readiness_sim_smoke.py`, which archives:
  - `artifacts/readiness/sim_smoke.json`
  - `artifacts/readiness/sim_evidence_verify.json`
  - `artifacts/sim/smoke_run.json`
  - `artifacts/sim/smoke_profile.json`
  - `artifacts/sim/smoke_replay.json`
- The full-platform readiness profile also includes simulation native FFI smoke execution via
  `scripts/readiness_sim_native_smoke.py`, which archives:
  - `artifacts/readiness/sim_native_smoke.json`
  - `artifacts/readiness/sim_native_evidence_verify.json`
  - `artifacts/sim/native_smoke_run.json`
  - `artifacts/sim/native_smoke_profile.json`
- The full-platform readiness profile also includes simulation stdlib primitive smoke execution via
  `scripts/readiness_sim_stdlib_smoke.py`, which archives:
  - `artifacts/readiness/sim_stdlib_smoke.json`
  - `artifacts/readiness/sim_stdlib_evidence_verify.json`
  - `artifacts/sim/stdlib_smoke_run.json`
  - `artifacts/sim/stdlib_smoke_profile.json`

Train/Eval config schema:
- v1 config requires `config_version: 1` and the mandatory fields listed in
  `docs/25_train_eval_cli.md`.
- Optional v1.2+ fields include `world_size`, `rank`, `grad_accum_steps`, `grad_clip_norm`,
  `amp { enabled, dtype, init_scale, growth_factor, backoff_factor, growth_interval }`,
  `shuffle`, and `prefetch_batches`.
- Optional v3.0.0 additive distributed orchestration fields include:
  - `dist.topology` (`standalone|single-node|multi-node`)
  - `dist.rendezvous` (non-empty string; required tcp endpoint for `multi-node`)
  - `dist.retry_budget` (`Int >= 0`)
  - `dist.device_map` (CSV string or list of Ints; one-to-one rank->device mapping)
- Optional v2.1.x additive fields include:
  - `model.preset` (`tinylm|gpt2-small|gpt2-medium|llama-7b`)
  - `model.ff_mult`, `model.activation` (`gelu|relu|silu`), `model.norm` (`layernorm|rmsnorm`)
  - `model.tie_embeddings`, `model.dropout`
  - divergence guard controls:
    `ema_decay`, `divergence_factor`, `divergence_patience`, `divergence_warmup_steps`
  - pretraining run metadata:
    `run_id`, `parent_run_id`, `run_name`
  - checkpoint lifecycle policy:
    `checkpoint_policy.validate_on_save`, `checkpoint_policy.validate_on_resume`,
    `checkpoint_policy.retention_recent`, `checkpoint_policy.retention_milestone_every`,
    `checkpoint_policy.retention_milestone_keep`
- Train/pretrain write additive operational metadata under `checkpoint_dir`:
  - `run_state.json` (resumable run identity + status),
  - `runs/index.jsonl` (append-only run events),
  - `checkpoint_lifecycle.json` (integrity digests + tiered retention metadata).
- Simulation runs can write additive operational metadata through the CLI:
  - `--lineage-output <file>` emits `simulation_lineage_v1`
  - `--snapshot-manifest-output <file>` emits `world_snapshot_v1`
  - manifests carry run identity, source hash, environment hash, config hash, and snapshot hash.
- v3.0.0 strict behavior rejects configs missing `config_version`.
- Optional legacy recovery mode:
  - `--lenient-contracts` is accepted only when `ENKAI_ALLOW_LEGACY_CONTRACTS=1`.

Checkpoint format:
- v1 checkpoints include `format_version: 1` in `meta.json`.
- Ranked checkpoints write `rank{n}/` subdirectories and a `manifest.json` with `world_size`.
- Migration tooling:
  - `enkai migrate config-v1` emits canonical v1 config with `config_version: 1`.
  - `enkai migrate checkpoint-meta-v1` upgrades/validates checkpoint `meta.json` files and
    supports `--dry-run`, `--verify`, and strict verification via `--strict-contracts`.
  - `enkai doctor` scans configs/checkpoints for strict-contract blockers (strict by default),
    with machine-readable output via `--json`.

-------------------------------------------------------------------------------
12. Known Limits in v3.1.2
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
- Model registry support is filesystem-based (`--registry` directory scanning) and includes additive
  remote sync commands (`push|pull|promote-remote|retire-remote|rollback-remote|verify-signature`) with immutable
  artifact manifests and optional signature verification.
- The registry lifecycle is shared across artifact kinds:
  - `checkpoint`
  - `simulation`
  - `environment`
  - `native-extension`
  with additive metadata for artifact and lineage manifests.
- Remote registry auth/replication over external services is not part of the in-repo v2.5.x implementation.
- `std::analysis` provides deterministic ingest + aggregate primitives, but does not yet provide
  a full dataframe/columnar query planner.
- `std::algo` provides foundational software/ML utility routines, but not exhaustive domain
  libraries for every specialized algorithm family.
- Frontend scaffolds target React + TypeScript web projects; non-web/mobile generators are not part of the current v2.x scope.
- Runtime training-forward integration supports configurable decoder-only transformer blocks
  via model-spec metadata with TinyLM-compatible fallback behavior for older tensor builds.
  Full-scale pretraining/serving envelopes still depend on operator hardware validation gates.
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
- `enkai litec selfhost-ci` runs subset corpus compile/execute parity with optional
  Stage0 result comparison and can emit deterministic triage reports via `--triage-dir`.
- `enkai litec replace-check` validates stage0/stage1/stage2 corpus compilation/runtime
  equivalence, reports compiler fixed-point status for the bootstrap subset, and can emit
  deterministic triage reports via `--triage-dir`; it is not
  yet a full replacement for Rust Stage0 compiler releases.
- `enkai litec mainline-ci` composes `selfhost-ci --no-compare-stage0` and
  `replace-check --no-compare-stage0` to make the Enkai-built compiler path the
  default CI self-host lane while preserving a separate mandatory Stage0 fallback lane.
- `enkai cluster run` supervises bounded simulation workloads in v3.0.0, including
  windowed `sim run`/`sim replay` execution with snapshot output and bounded retry/recovery.
  Multi-node train execution remains operator-managed.
- v3.0.0 validation note:
  - CPU-mode single-device soak requires operator-run evidence on production hardware.
  - CUDA single-GPU long-soak and distributed (2-GPU/4-GPU) reliability remain
    operator-run requirements and are not auto-proven by repository state alone.
  - Full-platform non-hardware gate bundle for the v2.5+ program is executed with:
    `enkai readiness check --profile full_platform --json --output artifacts/readiness/full_platform.json`
    and governed by:
    `enkai/contracts/readiness_full_platform_v2_5_0.json`,
    `enkai/contracts/full_platform_release_blockers_v2_5_0.json`.
- v3.0.0 native-backed simulation note:
  - stdlib simulation primitive verification now requires profile evidence that the native
    acceleration path was used for `std::sparse`/`std::event`/`std::pool` hot operations.
- v3.0.0 coroutine and Adam-0 note:
  - release sign-off now requires archived evidence that task-backed `std::sim` coroutines
    executed successfully, that coroutine audits recorded seed/config/log/snapshot/replay hashes
    plus coroutine counters, that the in-tree Adam-0 100-agent deterministic baseline
    completed with verified run/profile artifacts, and that the bounded Adam-0 reference
    suite (`examples/adam0_reference.enk`) completed with verified 100 / 1000 / 10000
    agent run/profile/snapshot/replay artifacts.
- v3.0.0 SNN/agent kernel note:
  - release sign-off now also requires archived evidence that the in-tree
    `examples/snn_agent_kernel.enk` workload completed with verified run/profile artifacts
    and exercised native-backed spatial/SNN acceleration paths.
- v3.0.0 scale/reliability note:
  - release sign-off now also requires archived evidence that `enkai cluster run`
    completed a bounded multi-node simulation workload with deterministic host/partition
    routing, persisted per-window snapshots, and at least one snapshot-based retry recovery.
  - degraded registry verification also requires a signed pull path that succeeds with
    `--fallback-local` when the remote registry becomes unavailable.
- Machine-readable blocker verification is executed with:
  `enkai readiness verify-blockers --profile full_platform --report artifacts/readiness/full_platform.json --json --output artifacts/readiness/full_platform_blockers.json`
  and validates required readiness checks plus required evidence artifacts for the current version line.
- Strict release evidence archives must include both:
  `artifacts/readiness/full_platform.json` and
  `artifacts/readiness/full_platform_blockers.json`.
- Strict release evidence archives must also include:
  `artifacts/readiness/sim_smoke.json`,
  `artifacts/readiness/sim_evidence_verify.json`,
  `artifacts/readiness/sim_native_smoke.json`,
  `artifacts/readiness/sim_native_evidence_verify.json`,
  `artifacts/readiness/sim_stdlib_smoke.json`,
  `artifacts/readiness/sim_stdlib_evidence_verify.json`,
  `artifacts/readiness/adam0_reference_suite.json`,
  `artifacts/readiness/adam0_reference_suite_verify.json`,
`artifacts/readiness/runtime_safety.json`,
`artifacts/readiness/runtime_safety_verify.json`,
  `artifacts/readiness/cluster_scale_smoke.json`,
  `artifacts/readiness/cluster_scale_evidence_verify.json`,
  `artifacts/readiness/registry_degraded_smoke.json`,
  `artifacts/readiness/registry_degraded_evidence_verify.json`,
  `artifacts/sim/smoke_run.json`,
  `artifacts/sim/smoke_profile.json`, and
  `artifacts/sim/smoke_replay.json`,
  `artifacts/sim/native_smoke_run.json`, and
  `artifacts/sim/native_smoke_profile.json`,
  `artifacts/sim/stdlib_smoke_run.json`, and
  `artifacts/sim/stdlib_smoke_profile.json`,
  `artifacts/sim/adam0_baseline_100_run.json`,
  `artifacts/sim/adam0_baseline_100_profile.json`,
  `artifacts/sim/adam0_baseline_100_snapshot.json`,
  `artifacts/sim/adam0_baseline_100_replay.json`,
  `artifacts/sim/adam0_stress_1000_run.json`,
  `artifacts/sim/adam0_stress_1000_profile.json`,
  `artifacts/sim/adam0_stress_1000_snapshot.json`,
  `artifacts/sim/adam0_stress_1000_replay.json`,
  `artifacts/sim/adam0_target_10000_run.json`,
  `artifacts/sim/adam0_target_10000_profile.json`,
  `artifacts/sim/adam0_target_10000_snapshot.json`,
  `artifacts/sim/adam0_target_10000_replay.json`,
  `artifacts/cluster_scale/run.json`,
  `artifacts/registry_degraded/cache/audit.log.jsonl`,
`artifacts/registry/remote/adam0-sim/v3.0.0/remote.manifest.json`,
`artifacts/registry/remote/adam0-sim/v3.0.0/remote.manifest.sig`,
`artifacts/registry_degraded/remote_offline/adam0-degraded/v3.0.0/remote.manifest.json`, and
`artifacts/registry_degraded/remote_offline/adam0-degraded/v3.0.0/remote.manifest.sig`.
- Strict capability reporting requires the archived blocker report to be present and to record a passing final verification (`all_passed: true`, `skip_release_evidence: false`).
- Release pipelines may additionally pass:
  `--allow-skipped-required-check selfhost-mainline --allow-skipped-required-check selfhost-stage0-fallback`
  because those checks are intentionally superseded by the stronger `enkai litec release-ci` gate.
- Add `--require-gpu-evidence` when validating final operator evidence bundles.
- Add `--skip-release-evidence` only for constrained-host reduced runs where packaging artifacts are intentionally skipped.
  - Operator logs can be verified with `scripts/verify_gpu_gates.ps1` or
    `scripts/verify_gpu_gates.sh`.

These limits are part of the current stable contract and should be treated as production constraints.

-------------------------------------------------------------------------------
13. Change Control
-------------------------------------------------------------------------------

For any language/runtime surface change after v3.0.0:
1) Implement the change and add/adjust compiler/runtime tests.
2) Update this specification to match the shipped behavior.
3) Update changelog and targeted docs (`docs/xx_*.md`, `docs/tensor_api.md`, etc.).
4) If compatibility/deprecation behavior changes, update `docs/29_compatibility_policy.md`.

















