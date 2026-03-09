# Changelog

## Unreleased

### Highlights
- None yet.

### Fixes
- None yet.

### Breaking changes
- None.

## v2.1.9

### Highlights
- Completed v2.1.x stability-cut evidence hardening:
  - release pipelines now run benchmark target gates and emit bounded-claim artifacts:
    - `dist/benchmark_official_v2_1_9_linux.json`
    - `dist/benchmark_official_v2_1_9_windows.json`
  - release evidence archival now captures expanded categories:
    - `dist` (archive/checksum/SBOM/benchmark)
    - `selfhost` triage artifacts
    - `contracts` snapshots
    - `gpu` operator evidence (when required)
- Added capability-complete report generation:
  - `scripts/generate_capability_report.py`
  - outputs:
    - `artifacts/release/v<version>/capability_complete.json`
    - `artifacts/release/v<version>/capability_complete.md`
- Added current-line RC wrappers:
  - `scripts/v2_1_9_rc_pipeline.ps1`
  - `scripts/v2_1_9_rc_pipeline.sh`

### Fixes
- `scripts/collect_release_evidence.py` now supports strict evidence validation and self-host/contract category capture.
- RC pipelines now generate capability reports in both dry-run and full modes; full mode enforces strict evidence checks when package gates run.
- Updated benchmark/CI/docs contracts to pin `official_v2_1_9`.

### Breaking changes
- None.

## v2.1.8

### Highlights
- Completed v2.1.x performance hardening pass:
  - VM arithmetic hot-path optimization for `Int` operations (`+`, `-`, `*`, `/`, `%`)
  - direct integer comparison paths for `<`, `>`, `<=`, `>=`
- Added official bounded benchmark suite for v2.1.8:
  - `bench/suites/official_v2_1_8.json`
  - focused on production-representative tokenizer/dataset/serving workloads
- Added benchmark target enforcement controls:
  - `--enforce-target` now validates suite-level median targets
  - `--enforce-all-cases` adds strict per-case target enforcement

### Fixes
- Added deterministic runtime errors for divide-by-zero and modulo-by-zero in VM numeric ops.
- Updated CI benchmark lanes:
  - smoke lane now uses `official_v2_1_8`
  - new `benchmark-target-gate` runs on Linux + Windows with release binaries and target enforcement.
- Updated machine profiles to pin `official_v2_1_8` for bounded claim evidence.

### Breaking changes
- None.

## v2.1.7

### Highlights
- Completed bootstrap mainline integration hardening:
  - added `enkai litec mainline-ci <corpus_dir> [--triage-dir <dir>]`
  - mainline lane composes:
    - `enkai litec selfhost-ci ... --no-compare-stage0`
    - `enkai litec replace-check ... --no-compare-stage0`
- Added deterministic self-host triage report outputs:
  - `litec_selfhost_ci_report.json`
  - `litec_replace_check_report.json`
  - `litec_mainline_ci_report.json`
- CI self-hosting now runs two explicit lanes:
  - `selfhost-mainline` (Enkai-built compiler default path)
  - `selfhost-stage0-fallback` (mandatory Stage0 safety path)

### Fixes
- `selfhost-ci` and `replace-check` source-file traversal is now sorted for deterministic report ordering.
- Release pipelines (`scripts/release_pipeline.sh`, `scripts/release_pipeline.ps1`) now execute the mainline lane plus Stage0 fallback lane.
- Added bootstrap regression tests for triage-report emission and `mainline-ci` flow.

### Breaking changes
- None.

## v2.1.6

### Highlights
- Completed fullstack platform contract freeze hardening:
  - expanded scaffolds:
    - `enkai new service`
    - `enkai new llm-backend`
    - `enkai new llm-fullstack`
  - backend scaffolds now include deployment env + migration assets:
    - `.env.example`
    - `contracts/deploy_env.snapshot.json`
    - `scripts/validate_env_contract.py`
    - `migrations/001_conversation_state.sql`
    - `migrations/002_conversation_state_index.sql`
- Hardened generated backend persistence behavior:
  - dual-write durability (`conversation_state.json` + `conversation_state.backup.json`)
  - startup fallback migration from backup when primary state is missing
- Added contract snapshots for deployment env profiles:
  - `enkai/contracts/deploy_env_backend_v1.snapshot.json`
  - `enkai/contracts/deploy_env_llm_v1.snapshot.json`

### Fixes
- Re-enabled and expanded `frontend::tests` on Windows in `enkai/src/frontend.rs` (previously skipped by module-level platform guard).
- Added generated-fullstack upgrade regression coverage:
  - force-rescaffold version upgrade now checked for backend/frontend snapshot alignment.
- Added scaffold regression coverage for service/LLM profile outputs.

### Breaking changes
- None.

## v2.1.5

### Highlights
- Completed additive `std::algo` expansion for v2.1.5:
  - software primitives: `top_k_ints`, `merge_sorted_ints`, `merge_count_maps`
  - streaming transforms: `cumulative_sum`, `window_mean` (in addition to existing `window_sum`)
  - ML utility metrics/eval: `mae`, `rmse`, `precision_recall_f1`
  - deterministic split helper: `split_indices(total, test_ratio, seed, shuffle)`
  - scheduler utility: `scheduler_linear_warmup(...)`
- Added algorithm complexity/perf baseline suite:
  - `bench/suites/algorithm_kernels.json`
  - paired Enkai/Python kernels under `bench/enkai/` and `bench/python/`

### Fixes
- Added deterministic ordering and merge behavior for hash-map aggregation utilities.
- Added native and runtime integration coverage for new algorithm + ML helper APIs.
- Added golden-corpus regression assertions for algorithm outputs and deterministic split behavior.

### Breaking changes
- None.

## v2.1.3

### Highlights
- Hardened HTTP serving/runtime observability contract:
  - deterministic response headers: `x-enkai-correlation-id`, `x-enkai-queue-ms`,
    `x-enkai-latency-ms`, `x-enkai-inflight`, model tags, and `x-enkai-error-code`
  - structured JSONL logging now includes correlation, queue/inflight, and model metadata
- Added serving safety controls:
  - backpressure middleware (`http.middleware("backpressure", ...)`) with deterministic
    `503 backpressure_overloaded`
  - model-version enforcement (`missing_model_version`, `model_version_mismatch`,
    `model_name_mismatch`) for pinned serving flows
- Expanded rate-limit keying options:
  - `tenant`, `model`, and `tenant_model` in addition to existing `ip`/`token`

### Fixes
- Standardized malformed-request and runtime-error responses to deterministic JSON error payloads.
- Added regression coverage for:
  - model header enforcement
  - backpressure overload behavior
  - correlation header roundtrip
  - structured internal error responses
  - observability header emission
  - tenant/model-scoped rate-limit isolation
- Updated release/docs/spec references from `v2.1.2` to `v2.1.3`.

### Breaking changes
- None.

## v2.1.2

### Highlights
- Added additive pretraining command:
  - `enkai pretrain <config.enk> [--strict-contracts|--lenient-contracts]`
- Added run identity + lineage metadata support in train/pretrain configs:
  - `run_id`, `parent_run_id`, `run_name`
- Added run lifecycle artifacts under `checkpoint_dir`:
  - `run_state.json`
  - `runs/index.jsonl`
  - `checkpoint_lifecycle.json`
- Added checkpoint lifecycle policy controls:
  - `checkpoint_policy.validate_on_save`
  - `checkpoint_policy.validate_on_resume`
  - `checkpoint_policy.retention_recent`
  - `checkpoint_policy.retention_milestone_every`
  - `checkpoint_policy.retention_milestone_keep`

### Fixes
- Added checkpoint integrity validation on save/resume against lifecycle metadata.
- Added retention pruning logic that preserves milestone checkpoints by policy.
- Added regression coverage for pretraining metadata output and lifecycle integrity failures.

### Breaking changes
- None.

## v2.1.1

### Highlights
- Added additive model-architecture training config surface:
  - `model.preset`, `model.ff_mult`, `model.activation`, `model.norm`,
    `model.tie_embeddings`, `model.dropout`
- Added native runtime/tensor configurable LM path:
  - new tensor FFI symbols `enkai_tensor_lm_init` and `enkai_tensor_forward_lm`
  - compatibility fallback to TinyLM symbols when newer LM symbols are unavailable
- Added deterministic divergence guard controls for long-running training:
  - `ema_decay`, `divergence_factor`, `divergence_patience`, `divergence_warmup_steps`

### Fixes
- Added parser and regression tests for new architecture fields and validation failures.
- Hardened model-signature checkpoint matching with legacy compatibility fallback.
- Updated training/tensor docs for additive architecture + safety controls.

### Breaking changes
- None.

## v2.1.0

### Highlights
- Added benchmark foundation and CLI surface:
  - `enkai bench run --suite <name> --baseline <python|none> --output <file>`
  - deterministic benchmark harness + suites under `bench/`
  - official bounded claim suite: `official_v2_1_0`
  - pinned Linux/Windows reference machine manifests in `bench/machines/`
- Added model lifecycle CLI:
  - `enkai model register|list|promote|retire|rollback`
  - active-version pointer + checkpoint pointer serving resolution support in `enkai serve`
- Added additive std modules:
  - `std::analysis` (CSV/JSONL ingest + schema/filter/project/group/describe/histogram)
  - `std::algo` (sorting/search/path + ML metric helpers)
- Added benchmark CI lane (`benchmark-smoke`) with artifact upload.

### Fixes
- Fixed `scripts/v1_9_release_pipeline.ps1` wrapper argument forwarding for switch parameters.
- Added regression coverage for new `std::analysis` and `std::algo` module behavior in runtime tests.
- Updated docs/spec/roadmap to include v2.1.0 benchmark and model-lifecycle contracts.

### Breaking changes
- None.

## v2.0.0

### Highlights
- Strict contracts are now enforced by default on train/eval runtime acceptance paths.
- Added explicit legacy-recovery gate for operators:
  - `--lenient-contracts` requires `ENKAI_ALLOW_LEGACY_CONTRACTS=1`.
- Added release-line RC wrappers:
  - `scripts/v2_0_0_rc_pipeline.ps1`
  - `scripts/v2_0_0_rc_pipeline.sh`
- Version line advanced to `2.0.0` across workspace crates.

### Fixes
- Hardened strict-checkpoint enforcement and migration/doctor parity behavior.
- Added regression tests for strict defaults and lenient-gate behavior.
- Updated spec/docs/release metadata for the v2.0.0 stability cut.

### Breaking changes
- `enkai train` / `enkai eval` now reject configs without `config_version` by default.
- Checkpoint loads now reject missing required v1 metadata by default.

## v1.9.9

### Highlights
- Added strict-contract preflight execution modes:
  - `enkai train <config> [--strict-contracts|--lenient-contracts]`
  - `enkai eval <config> [--strict-contracts|--lenient-contracts]`
  - env default: `ENKAI_STRICT_CONTRACTS=1`
- Added strict checkpoint-meta verification mode:
  - `enkai migrate checkpoint-meta-v1 <checkpoint_dir> --verify --strict-contracts`
- Hardened doctor workflow:
  - strict-by-default scan for v2.0 blockers
  - machine-readable output via `enkai doctor --json`
  - transition mode via `enkai doctor --lenient`
- Added release-line RC wrappers:
  - `scripts/v1_9_9_rc_pipeline.ps1`
  - `scripts/v1_9_9_rc_pipeline.sh`

### Fixes
- Added regression tests for strict contract behavior in train/eval and migration checks.
- Aligned docs/spec/readme/release metadata to `v1.9.9`.

### Breaking changes
- None.

## v1.9.8

### Highlights
- Added RC freeze pipeline and wrappers:
  - `scripts/rc_pipeline.ps1`
  - `scripts/rc_pipeline.sh`
  - `scripts/v1_9_8_rc_pipeline.ps1`
  - `scripts/v1_9_8_rc_pipeline.sh`
- Added release evidence archive tool:
  - `scripts/collect_release_evidence.py`
  - writes `artifacts/release/v<version>/manifest.json`
- Published `v2.0.0` RC policy and migration docs:
  - `docs/31_v2_rc_notes.md`
  - `docs/32_v2_migration_guide.md`

### Fixes
- Extended docs consistency checks to enforce RC docs and checklist references for:
  - RC pipeline scripts
  - release evidence archive tooling
  - v2 migration command references
- Updated CI with RC dry-run lane and release metadata/docs alignment for `v1.9.8`.

### Breaking changes
- None.

## v1.9.7

### Highlights
- Added deterministic packaging + verification tooling:
  - `scripts/package_release.py`
  - `scripts/verify_release_artifact.py`
- Added version-neutral release pipelines:
  - `scripts/release_pipeline.ps1`
  - `scripts/release_pipeline.sh`
- Added provenance/security tooling:
  - `scripts/license_audit.py`
  - `scripts/generate_sbom.py`
- Updated CI package checks to run on Linux + Windows with deterministic archive checks, checksum verification, smoke execution, and SBOM artifact output.

### Fixes
- Hardened PowerShell pipeline gate handling so native-command failures propagate deterministically.
- Stabilized HTTP stream test behavior under connection-reset races (`enkairt/tests/http.rs`).
- Updated release docs/checklists/spec references for `v1.9.7` and version-neutral release scripts.

### Breaking changes
- None.

## v1.9.6

### Highlights
- Frozen serve/frontend compatibility contract with explicit snapshot artifacts:
  - `backend/contracts/backend_api.snapshot.json`
  - `backend/contracts/conversation_state.schema.json`
  - `frontend/contracts/sdk_api.snapshot.json`
- Added CI/release gate for contract snapshots:
  - `frontend::tests::contract_snapshots_match_reference_files`
- Expanded generated backend contract with WebSocket route:
  - `GET /api/<version>/chat/ws`
- Expanded generated SDK contract with `streamChatWs(...)` helper and improved structured error-detail parsing.

### Fixes
- Hardened scaffold persistence contract:
  - `conversation_state.json` now writes `schema_version: 1` and structured `messages` payload.
  - Legacy scaffold state files (without `schema_version`) are migrated at backend startup.
- Added fullstack contract assertions for scaffolded snapshot files and persistence schema upgrade behavior.

### Breaking changes
- None.

## v1.9.5

### Highlights
- Added v1.9.5 distributed reliability tooling:
  - first-party multi-GPU launcher flow via `scripts/gpu_harness.py` used by both PowerShell and shell harness scripts
  - structured GPU evidence outputs in `artifacts/gpu`:
    - `single_gpu_evidence.json`
    - `multi_gpu_evidence.json`
    - `soak_4gpu_evidence.json`
- Added deterministic parity report generation for 2-GPU harness runs (loss parity + grad parity checks).

### Fixes
- Hardened distributed runtime failures with machine-parseable error codes (`E_DIST_*`) and clearer remediation guidance for missing `torch,dist` builds.
- Updated GPU evidence verifier scripts to support both legacy log format and structured JSON evidence format.
- Updated GPU harness scripts to first-party wrappers on Windows/Linux without requiring ad-hoc launcher composition for the 2-GPU path.

### Breaking changes
- None.

## v1.9.4

### Highlights
- Added migration + readiness CLI tooling:
  - `enkai migrate config-v1 <in> <out>`
  - `enkai migrate checkpoint-meta-v1 <checkpoint_dir> [--dry-run] [--verify]`
  - `enkai doctor [path]`
- Added checkpoint metadata verifier with required-key checks and cross-tree consistency checks for:
  - `config_hash`
  - `model_sig`
  - `dtype`
  - `device`
- Added canonical config-v1 emission path for migration output.

### Fixes
- Added fixture-backed migration/doctor tests and checkpoint metadata migration tests.
- Aligned docs/spec/readme references and command lists for `v1.9.4`.

### Breaking changes
- None.

## v1.9.3

### Highlights
- Hardened runtime/tool failure contracts with stable machine-parseable error codes:
  - `E_POLICY_DENIED`, `E_POLICY_UNKNOWN`
  - `E_TOOL_CONFIG`, `E_TOOL_SPAWN`, `E_TOOL_TIMEOUT`, `E_TOOL_WAIT`
  - `E_TOOL_IO`, `E_TOOL_PAYLOAD`, `E_TOOL_EXIT`, `E_TOOL_OUTPUT_FORMAT`
- Added explicit runtime contract documentation that bytecode VM behavior is production-normative and the legacy tree-walk interpreter is compatibility/reference only.
- Expanded policy/tool safety test coverage:
  - tool spawn/timeout/policy-denial regression tests
  - policy denial tests for `std::process`, `std::db`, and HTTP without policy capability.

### Fixes
- Ensured tool payload/output conversion failures are surfaced with deterministic coded errors.
- Added stable error-code assertions in AI declaration and policy integration tests.
- Aligned docs and release references to `v1.9.3`.

### Breaking changes
- None.

## v1.9.2

### Highlights
- Added version single-sourcing for language version reporting via build-time env wiring.
- Fixed CI package-check workflow flow control and added workflow linting.
- Added docs contract consistency gate scripts:
  - `scripts/check_docs_consistency.py`
  - `scripts/check_docs_consistency.ps1`
- Aligned top-level docs/spec/readme references for v1.9 release-line consistency.

### Fixes
- Removed stale “stub/placeholder” runtime wording from production paths where behavior is implemented.
- Added CI/docs checks to prevent release/version contract drift.

### Breaking changes
- None.

## v1.9.1

### Highlights
- Added bootstrap replacement-readiness command:
  - `enkai litec replace-check <corpus_dir> [--no-compare-stage0]`
- Added Stage1->Stage2 replacement-readiness checks and Stage2->Stage3 fixed-point status reporting for `enkai_lite` compiler artifacts.
- Added explicit distributed runtime opt-in gate:
  - `ENKAI_ENABLE_DIST=1` required for multi-rank distributed mode.
- Hardened distributed tensor runtime paths:
  - explicit multi-rank init/allreduce failure modes in `enkai_tensor`,
  - rank-device validation and reconfiguration guards,
  - fallback distributed symbols with clear build-feature errors when `torch,dist` is missing.
- Added Linux 2-GPU and 4-GPU harness parity scripts:
  - `scripts/multi_gpu_harness.sh`
  - `scripts/soak_4gpu.sh`
- Added server-side WebSocket runtime APIs for HTTP handlers:
  - `http.ws_open(req)`
  - `http.ws_send(ws, message)`
  - `http.ws_recv(ws, timeout_ms)`
  - `http.ws_close(ws)`
- Added host tool invocation runtime for `tool` declarations:
  - compiled tool declarations now route through `tool.invoke(name, args)`
  - host binding via `ENKAI_TOOL_<PATH>` or `ENKAI_TOOL_RUNNER`
- Enabled `async fn` parsing in modules and impl blocks (compiled with existing task-based async primitives).
- Added Postgres connector APIs in `std::db` backed by `enkai_native`:
  - `pg_open`, `pg_exec`, `pg_query`, `pg_close`
- Added best-effort GPU profiling fields in engine metrics for CUDA device configs:
  - `gpu_mem_mb`, `gpu_util` sampled via `nvidia-smi` when available.
- Expanded v1.8 release pipeline scripts to include replacement fixed-point gates and optional GPU evidence verification.

### Fixes
- Added parser-level distributed config guard in training config handling to prevent accidental multi-rank runs without explicit opt-in.
- Updated CUDA rank-device test to set distributed opt-in env gate.
- Added bootstrap test coverage for replacement fixed-point command.
- Added distributed guard tests in `enkai_tensor/tests/dist_guards.rs`.
- Added HTTP integration test coverage for WebSocket upgrade + text frame delivery.
- Added HTTP integration test coverage for inbound WebSocket recv + echo flow.
- Added parser coverage for `async fn` in module/public/impl contexts.
- Added AI declaration coverage for configured host tool invocation.
- Added `std::db` regression test coverage for Postgres open failure semantics (`none` on unreachable DSN).

### Breaking changes
- None yet.

## v1.9.0

### Highlights
- Added stage1 execution command:
  - `enkai litec run <input.enk>`
- Added master pipeline smoke test:
  - `master_pipeline_cpu_smoke` (train/eval + frontend scaffold + self-host checks)
- Added GPU evidence verification scripts:
  - `scripts/verify_gpu_gates.ps1`
  - `scripts/verify_gpu_gates.sh`
- Added consolidated v1.9 release pipeline scripts:
  - `scripts/v1_9_release_pipeline.ps1`
  - `scripts/v1_9_release_pipeline.sh`

### Fixes
- Added self-host regression test coverage for `litec run`.
- Extended self-host corpus fixture to include method dispatch usage in stage1 flow.
- Updated CI to run v1.9 release pipeline lane.

### Breaking changes
- None.

## v1.8.0

### Highlights
- Added compatibility + deprecation governance docs:
  - `docs/29_compatibility_policy.md`
  - `docs/28_selfhost_workflow.md`
- Added consolidated v1.8 release pipeline scripts:
  - `scripts/v1_8_release_pipeline.ps1`
  - `scripts/v1_8_release_pipeline.sh`
- Added explicit compatibility gates for legacy training artifacts:
  - legacy train config without `config_version`
  - legacy checkpoints missing `format_version`

### Fixes
- Added train/eval warning path for legacy configs to guide migration to `config_version: 1`.
- Added self-host CI test coverage for the `--no-compare-stage0` fallback mode.
- Added repository self-host corpus fixtures for stable day-to-day bootstrap checks.

### Breaking changes
- None.

## v1.7.0

### Highlights
- Added bootstrap self-host beta command:
  - `enkai litec selfhost <corpus_dir>`
- Added staged bootstrap frontend command:
  - `enkai litec stage <parse|check|codegen> <input.enk> [--out <program.bin>]`
- Added self-host CI command:
  - `enkai litec selfhost-ci <corpus_dir> [--no-compare-stage0]`
- Expanded bootstrap-core subset validation for Stage1 corpus support:
  - allows `use`, `type`, `enum`, `impl` declarations
  - allows non-capturing lambda expressions
- Refactored `enkai_lite.enk` into explicit parse/check/codegen phase functions, so stage orchestration for selected compiler components is executed in Enkai.
- Added shared Stage0/Stage1 bytecode equivalence helper used by `litec verify`, `litec selfhost`, and `litec selfhost-ci`.

### Fixes
- Added self-host beta tests covering expanded subset acceptance and corpus verification behavior.
- Added CI `selfhost-beta` lane to run bootstrap self-host regression tests and execute `litec selfhost-ci` against repository corpus.

### Breaking changes
- None.

## v1.6.0

### Highlights
- Added bootstrap-core CLI commands:
  - `enkai litec check <input.enk>`
  - `enkai litec compile <input.enk> --out <program.bin>`
  - `enkai litec verify <input.enk>`
- Added Enkai-scripted bootstrap-core driver (`enkai/tools/bootstrap/enkai_lite.enk`) for Stage1 orchestration.
- Added runtime/compiler `compiler` module:
  - `parse_subset`
  - `check_subset`
  - `emit_subset`
- Added subset validation in runtime for bootstrap-core compilation flow.
- Added bootstrap-core docs: `docs/bootstrap_core.md`.

### Fixes
- Added stage0/stage1 bytecode equivalence verification path (`enkai litec verify`) and tests.
- Hardened policy checks for compiler emission output paths.
- Standardized CI bootstrap parity lane by building `enkai_native` before parity tests.

### Breaking changes
- None.

## v1.5.0

### Highlights
- Added bootstrap-lite CLI commands:
  - `enkai fmt-lite [--check] <file|dir>`
  - `enkai lint-lite [--deny-warn] <file|dir>`
  - `enkai tokenizer-lite train ...`
  - `enkai dataset-lite inspect ...`
- Added Enkai-scripted bootstrap tooling implementations embedded into the CLI and executed through the VM.
- Added runtime/compiler support for bootstrap tooling primitives:
  - new `bootstrap` runtime module (`format`, `check`, `lint`, `lint_count`, `lint_json`)
  - compiler + checker built-in registration for `bootstrap`
  - expanded tokenizer/dataset callable surface for script-based utilities.
- Added bootstrap subset specification: `docs/bootstrap_subset.md`.

### Fixes
- Added deterministic parity tests comparing Enkai bootstrap-lite execution paths with Rust baselines for formatter, linter, tokenizer, and dataset inspection flows.
- Added CI parity lane for bootstrap-lite (`bootstrap::tests`).
- Improved tokenizer capability handling for config-record input in policy context extraction.

### Breaking changes
- None.

## v1.4.0

### Highlights
- Added frontend stack scaffolding in CLI:
  - `enkai new backend`
  - `enkai new frontend-chat`
  - `enkai new fullstack-chat`
- Added typed SDK generator command:
  - `enkai sdk generate <output_file> [--api-version <v>]`
- Shipped React/TypeScript reference chat UI scaffold with streaming response handling, auth token input, and error UX conventions.
- Standardized frontend/backend version pinning contract:
  - route prefix `/api/<version>`
  - request header `x-enkai-api-version`.

### Fixes
- Added contract tests to validate scaffolded backend route layout and generated SDK endpoint/header behavior.
- Added end-to-end fullstack contract test coverage for generated backend/frontend scaffolds, including streaming event parsing and version mismatch behavior.
- Added persisted conversation flow in backend scaffold with conversation ID continuity across stream/chat APIs and persisted state snapshots.
- Hardened CLI argument validation for new SDK/scaffold commands with deterministic error messages.

### Breaking changes
- None.

## v1.3.0

### Highlights
- Added serving-oriented runtime APIs: `http.serve_with`, `http.route`, `http.middleware`, `http.request`, request helpers, and streaming helpers.
- Added HTTP middleware surface for auth, rate limiting, JSONL logging, and default route handling.
- Added structured HTTP error payloads and runtime response metadata headers (`x-enkai-request-id`, `x-enkai-latency-ms`, tenant/error-code where applicable).
- Added CLI `enkai serve` model-selection flow with registry/version pinning or direct checkpoint selection.
- Added filesystem-based model registry helpers via `std::model_registry`.
- Added `std::db` (SQLite) and `std::tls` helper modules backed by `enkai_native`.

### Fixes
- Normalized rate-limit IP keys to avoid per-connection bypass when client source ports change.
- Stabilized HTTP integration tests by serializing test server lifecycles to prevent port-binding races.

### Breaking changes
- None.

## v1.2.0

### Highlights
- Added multi-GPU runtime wiring (world size/rank config), grad accumulation, grad clipping, and AMP config support.
- Implemented ranked checkpoints with manifest metadata for distributed runs.
- Added dataset prefetch and packing efficiency metrics.
- Added build cache + deterministic `enkai.lock` resolver via `enkai build`.
- Expanded stdlib with `std::env`, `std::path`, `std::time`, `std::log`, `std::io`, `std::process`.
- Hardened policy/capability checks for new IO and networking surfaces.

### Fixes
- Improved checkpoint metadata compatibility in ranked loads.
- Deterministic dataset sharding across ranks.

### Breaking changes
- None.

## v1.1.0

### Highlights
- Implemented runtime semantics for previously parsed declarations:
  - `type`, `enum`, `impl` method dispatch
  - AI-native declarations: `tool`, `agent`, `prompt`, `model`, `memory`
- Added first-class ML stdlib modules:
  - `std::nn`
  - `std::loss`
  - `std::optim`
- Added deterministic seed controls in train/tokenizer/dataset flows.
- Extended checker/runtime surface with `Tensor`, `Device`, `DType`, and `Shape` contracts used by std ML modules.

### Fixes
- Added regression coverage for method dispatch, task `spawn`/`await` semantics, tokenizer determinism, and dataset determinism.
- Added `examples/nn_sanity.enk` for end-to-end `std::nn`-based training sanity checks.

### Breaking changes
- None.

## v1.0.0

### Highlights
- Froze v1 grammar/CLI compatibility baseline to the v0.9.3 language contract.
- Introduced train/eval config schema v1 (`config_version: 1`) with explicit validation of backend/device/dtype contracts.
- Froze checkpoint metadata schema v1 (`format_version: 1`) and preserved backward compatibility for legacy checkpoints.
- Locked training runtime to TinyLM CE forward path (`forward_tinylm`) and removed accidental training-path usage of `forward_l2`.
- Added formal release + validation governance:
  - `VALIDATION.md`
  - `docs/RELEASE_CHECKLIST.md`

### Fixes
- Added metadata compatibility tests for checkpoint load/save and legacy fallback behavior.
- Added release gate checks to prevent config/checkpoint schema drift.

### Breaking changes
- None.

## v0.9.3

### Highlights
- Synced release notes with the `docs/Enkai.spec` v0.1 -> v0.9.3 compatibility framing.
- Clarified v0.9.3 runtime status for tensor training hooks:
  - native loss path currently integrated as `forward_tinylm`,
  - checkpoint C ABI hooks are present and backend-loadable,
  - distributed hooks are present but environment-gated by CUDA/NCCL/runtime setup.
- Aligned release/version references to the current production line (`v0.9.3`) in top-level docs.
- Added `enkai --version` output with language version.

### Fixes
- Reduced spec/changelog drift by aligning guaranteed behavior vs known limits.

### Breaking changes
- None.

## v0.9.2

### Highlights
- Aligned release/version references to the current production line (`v0.9.2`) in top-level docs.
- Reframed `docs/Enkai.spec` as a compatibility baseline (`v0.1 -> v0.9.2`) to reduce spec/version drift.
- Added `docs/README.md` as a docs index with versioning policy and recommended reading order.
- Added roadmap guidance that historical milestones are context, while `docs/Enkai.spec` is the current language source of truth.

### Fixes
- Clarified legacy milestone labels in `docs/Enkai.spec` (features introduced in earlier versions) as baseline behavior in `v0.9.2`.

## v0.7.1

### Fixes
- Bumped workspace crate versions to 0.7.1 for release builds.

## v0.7.0

### Highlights
- Added cooperative tasks, channels, and a TCP networking layer for concurrent workloads.
- Shipped HTTP server/client helpers and JSON parse/stringify (record/list mappings).
- Added install scripts plus automated release packaging (zip/tar.gz + checksums).
- Added Windows installer builds (Inno Setup + WiX) with signing/notarization hooks.
- Expanded docs and learn lessons for concurrency, networking, HTTP, and JSON.

### Fixes
- Stabilized scheduler + IO blocking behavior with VM tests for tasks, channels, net, HTTP, and JSON.

## v0.5.0

### Highlights
- Added multi-file module imports with caching and circular import detection.
- Enforced public/private visibility across modules with compile + typecheck validation.
- Introduced structured diagnostics (source snippets) and runtime stack traces.
- Shipped deterministic `enkai fmt` and a CLI `enkai test` runner.
- Added v0.5 docs and learn lessons for modules, visibility, formatting, and testing.

### Fixes
- Added regression coverage for formatter/test CLI paths and private access span tracking.

## v0.4.0

### Highlights
- Introduced the bytecode + VM execution pipeline with chunk constants, stack safety, globals, and CLI flags (`--trace-vm`/`--disasm`).
- Added deep scope-aware symbol allocation, short-circuit logic ops, and Optional[T] coverage in the new type checker.
- Published learn/docs axes (learn lessons + docs/dev_debugging) and stabilized the CLI to always run `enkai check` before execution.

### Fixes
- Stabilized parser/compiler/VM tests, cleaned warnings, and ensured `cargo fmt`, `cargo clippy -D warnings`, and `cargo test` pass in the v0.4 path.

## v0.3.1

### Fixes
- Correct loader diagnostics for private symbol imports and reduce `use` resolution duplication.
- Use-list edge case diagnostics and expanded parser coverage for visibility rules.
- Policy filter matching now uses path components normalization and list-valued tests.
- Added tests for string native error paths and dependency manifest variants.

## v0.3.0

### Features
- Public exports + re-exports via `pub` and `pub use`.
- Labeled-span diagnostics for parse/loader errors.
- Local path dependencies in `enkai.toml`.
- Policy filters for `domain` and `path_prefix`.
- `std.string` helpers and `std.fs` read/write (policy-gated).

### Breaking changes
- Importing private symbols now errors; only `pub` exports are importable.

## v0.2.0

### Features
- Module loader + use resolution with file-based modules.
- `enkai run .` for project roots using `enkai.toml` and `src/main.enk`.
- Parser diagnostics include line/col + snippet.
- Runtime stack traces for errors.
- Minimal formatter with `enkai fmt` and `enkai fmt --check`.
- Policy enforcement MVP with default deny and allow/deny rules.

### Release Notes v0.2.0
- Smoke test:
  - `cargo run -p enkai -- run examples/project_v02`
  - `cargo run -p enkai -- fmt --check examples/project_v02/src/main.enk`

### Breaking changes
- IO and tool calls are now policy-gated; code that called `print` or tools without a policy will fail until a policy allows those capabilities.



