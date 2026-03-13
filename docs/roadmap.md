Enkai Roadmap

Note:
- Historical milestones below capture the path that led to current releases.
- Current production release line is v2.5.1.
- v2.5.x remains additive/integration work (no contract-breaking removals).
- Use `docs/Enkai.spec` as the source of truth for current language behavior.

v2.5.1 (done)
- LLM runtime reliability hardening for single-node train/eval lifecycle:
  - strict resume-time run-state validation for lineage/runtime identity fields
  - additive `run_validation.json` artifact emitted under `checkpoint_dir`
  - deterministic resume parity regression coverage
  - strict vs lenient mismatch behavior tests for dataset-hash drift

v2.5.0 (done)
- Program contract freeze + readiness expansion for full-platform line:
  - new readiness profile:
    - `enkai readiness check --profile full_platform --json --output <file>`
  - new readiness manifest:
    - `enkai/contracts/readiness_full_platform_v2_5_0.json`
  - machine-readable release blocker matrix:
    - `enkai/contracts/full_platform_release_blockers_v2_5_0.json`
- Expanded non-hardware gate bundle includes:
  - frontend/backend/LLM/DB smoke gates
  - bootstrap mainline + fallback gates
  - benchmark fairness + class-target gates
- Synced readiness docs/checklists:
  - `docs/37_readiness_matrix.md`
  - `docs/RELEASE_CHECKLIST.md`

v2.4.0 (done)
- Minor-line cut after v2.3.9:
  - release checklist synchronization and docs/spec consistency
  - benchmark class-gate reliability on pinned reference environments
  - release evidence integrity hardening for version-scoped archive/SBOM checks
  - GPU evidence runbook continuity (operator-run)

v2.3.9 (done)
- Advanced patch line to `v2.3.9` with additive compatibility and no contract removals.
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_9_rc_pipeline.ps1`
  - `scripts/v2_3_9_rc_pipeline.sh`
- Synced docs/spec/release metadata and version surfaces to `v2.3.9`.

v2.3.8 (done)
- Advanced patch line to `v2.3.8` with additive compatibility and no contract removals.
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_8_rc_pipeline.ps1`
  - `scripts/v2_3_8_rc_pipeline.sh`
- Synced docs/spec/release metadata and version surfaces to `v2.3.8`.

v2.3.7 (done)
- Advanced patch line to `v2.3.7` with additive compatibility and no contract removals.
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_7_rc_pipeline.ps1`
  - `scripts/v2_3_7_rc_pipeline.sh`
- Hardened release evidence and capability reporting:
  - `scripts/collect_release_evidence.py` archives version-scoped dist artifacts.
  - `scripts/generate_capability_report.py` validates version-scoped archive/checksum/SBOM evidence.
- Synced docs/spec/release metadata and version surfaces to `v2.3.7`.

v2.3.6 (done)
- Advanced patch line to `v2.3.6` with additive compatibility and no contract removals.
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_6_rc_pipeline.ps1`
  - `scripts/v2_3_6_rc_pipeline.sh`
- Synced docs/spec/release metadata and version surfaces to `v2.3.6`.

v2.3.5 (done)
- Advanced patch line to `v2.3.5` with additive compatibility and no contract removals.
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_5_rc_pipeline.ps1`
  - `scripts/v2_3_5_rc_pipeline.sh`
- Synced docs/spec/release metadata and version surfaces to `v2.3.5`.

v2.3.4 (done)
- Advanced patch line to `v2.3.4` with additive compatibility and no contract removals.
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_4_rc_pipeline.ps1`
  - `scripts/v2_3_4_rc_pipeline.sh`
- Synced docs/spec/release metadata and version surfaces to `v2.3.4`.

v2.3.3 (done)
- Advanced patch line to `v2.3.3` with additive compatibility and no contract removals.
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_3_rc_pipeline.ps1`
  - `scripts/v2_3_3_rc_pipeline.sh`
- Synced docs/spec/release metadata and version surfaces to `v2.3.3`.

v2.3.2 (done)
- Advanced patch line to `v2.3.2` with additive compatibility and no contract removals.
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_2_rc_pipeline.ps1`
  - `scripts/v2_3_2_rc_pipeline.sh`
- Hardened benchmark class-gate reliability by stabilizing VM compute workload envelope.

v2.3.1 (done)
- Advanced patch line to `v2.3.1` with additive compatibility and no contract removals.
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_1_rc_pipeline.ps1`
  - `scripts/v2_3_1_rc_pipeline.sh`
- Stabilized VM compute benchmark case for class-gate consistency:
  - `bench/enkai/kernel_numeric.enk`
  - `bench/python/kernel_numeric.py`
  - `bench/suites/official_v2_3_0_vm_compute.json`

v2.3.0 (done)
- Advanced workspace + contract version line to `v2.3.0`.
- Promoted official bounded benchmark suite for the new line:
  - `bench/suites/official_v2_3_0_matrix.json`
  - machine profiles pinned to `official_v2_3_0_matrix`.
- Hardened CI release discipline for evidence integrity:
  - `release-pipeline` CI lane now runs full package gates (not skip-package mode)
  - strict evidence archive + strict capability report generation are executed in CI
  - release evidence artifacts are uploaded from CI.
- Added explicit model serving lifecycle controls and multi-model runtime pinning:
  - `enkai model load|unload|loaded`
  - `enkai serve --multi-model --registry <dir>`
  - deterministic request-level selector/load enforcement (`missing_model_selector`, `model_not_loaded`).

v2.2.1 (in progress)
- Added production-readiness baseline + gate wiring:
  - `enkai readiness check --profile production --json --output <file>`
  - machine-readable readiness manifest:
    - `enkai/contracts/readiness_production_v2_3_0.json`
  - readiness matrix doc:
    - `docs/37_readiness_matrix.md`
- Added bootstrap release one-shot lane:
  - `enkai litec release-ci <corpus_dir> [--triage-dir <dir>]`
  - deterministic triage summary:
    - `litec_release_ci_report.json`
- Added deploy contract validator command:
  - `enkai deploy validate <project_dir> --profile <backend|fullstack> --strict`
- Added v2.3 benchmark target suite:
  - class-based suites:
    - `bench/suites/official_v2_3_0_vm_compute.json`
    - `bench/suites/official_v2_3_0_native_bridge.json`
    - `bench/suites/official_v2_3_0_cli_workflows.json`
    - `bench/suites/official_v2_3_0_ai_data_workflows.json`
    - `bench/suites/official_v2_3_0_matrix.json`
  - class targets:
    - `bench/suites/official_v2_3_0_targets.json`
  - workload-equivalence contract:
    - `bench/contracts/workload_equivalence_v1.json`

v2.1.9 (done)
- Completed v2.1 stability-cut evidence hardening:
  - promoted official bounded claim suite for the stability cut:
    - `bench/suites/official_v2_1_9.json`
  - release pipelines now run official benchmark target gates and emit benchmark evidence into `dist/`.
  - RC pipelines now archive expanded evidence categories (`dist`, `selfhost`, `contracts`, optional `gpu`).
  - RC pipelines now generate capability-complete reports:
    - `artifacts/release/v<version>/capability_complete.json`
    - `artifacts/release/v<version>/capability_complete.md`
  - strict RC sign-off enforces required archive/checksum/SBOM/benchmark/self-host/contract evidence.

v2.1.8 (done)
- Completed performance/efficiency hardening for the v2.1.x line:
  - added VM arithmetic fast paths for int-int add/sub/mul/div/mod and direct int comparisons.
  - added deterministic arithmetic safety errors for divide/modulo by zero.
- Hardened benchmark contract + gates:
  - new official bounded claim suite: `bench/suites/official_v2_1_8.json`
  - benchmark target enforcement now supports suite-median contract:
    - `--enforce-target` validates median speedup/memory targets
    - `--enforce-all-cases` enforces strict per-case targets
  - machine profiles now pin `official_v2_1_8`.
- Added CI regression blocker lanes for benchmark targets:
  - `benchmark-target-gate` runs on Linux + Windows with release binaries
  - requires `--target-speedup 5 --target-memory 5 --enforce-target`
  - uploads per-platform benchmark evidence artifacts.

v2.1.7 (done)
- Completed bootstrap mainline integration hardening:
  - added `enkai litec mainline-ci <corpus_dir> [--triage-dir <dir>]`
  - `mainline-ci` composes:
    - `litec selfhost-ci --no-compare-stage0`
    - `litec replace-check --no-compare-stage0`
  - added deterministic triage artifact support:
    - `litec selfhost-ci ... --triage-dir <dir>` -> `litec_selfhost_ci_report.json`
    - `litec replace-check ... --triage-dir <dir>` -> `litec_replace_check_report.json`
    - `litec mainline-ci ... --triage-dir <dir>` -> `litec_mainline_ci_report.json`
- Added CI lane split for self-hosting:
  - `selfhost-mainline`: Enkai-built compiler default path + triage artifact upload
  - `selfhost-stage0-fallback`: mandatory Stage0 comparison lane retained
- Updated release pipeline gates to run:
  - self-host mainline lane with triage output
  - self-host Stage0 fallback lane

v2.1.6 (done)
- Completed fullstack platform freeze/hardening:
  - expanded scaffold matrix:
    - `enkai new service`
    - `enkai new llm-backend`
    - `enkai new llm-fullstack`
  - deployment env contract artifacts in generated backends:
    - `contracts/deploy_env.snapshot.json`
    - `.env.example`
    - `scripts/validate_env_contract.py`
  - migration assets shipped in generated backends:
    - `migrations/001_conversation_state.sql`
    - `migrations/002_conversation_state_index.sql`
  - persistence durability hardening:
    - dual-write `conversation_state.json` + `conversation_state.backup.json`
    - startup migration reads primary/fallback state files
  - end-to-end generated fullstack compatibility tests now include:
    - contract snapshot parity checks for deployment env snapshot
    - force-rescaffold version upgrade contract checks
    - persistence migration validation during stream/chat flows

v2.1.5 (done)
- Completed additive algorithm-development stack hardening:
  - expanded `std::algo` software primitives (priority/top-k, sorted merge, hash-map merge)
  - streaming transforms (`window_sum`, `window_mean`, `cumulative_sum`)
  - ML utility metrics/eval helpers (`mae`, `rmse`, `precision_recall_f1`)
  - deterministic split helper + linear warmup scheduler utility
- Added correctness corpus coverage:
  - native unit tests for new `std::algo` FFI functions
  - runtime integration golden tests in `enkairt/tests/ffi_modules.rs`
- Added complexity/perf baseline suite:
  - `bench/suites/algorithm_kernels.json` with Enkai/Python parity kernels.

v2.1.3 (done)
- Hardened serving/runtime contract without syntax changes:
  - deterministic JSON error taxonomy for malformed/runtime/invalid-response paths
  - request correlation propagation (`x-enkai-correlation-id`) and stable response metadata headers
  - queue/inflight/model telemetry headers and JSONL observability fields
- Added serving controls:
  - backpressure middleware (`http.middleware("backpressure", ...)`) with deterministic `503 backpressure_overloaded`
  - model-version header enforcement (`missing_model_version`, `model_version_mismatch`, `model_name_mismatch`)
  - expanded rate-limit keying (`tenant`, `model`, `tenant_model`) for tenant/model-scoped quotas
- Added regression coverage for model header enforcement, backpressure handling, correlation roundtrip,
  structured internal errors, observability headers, and tenant/model rate-limit isolation.

v2.1.0 (done)
- Added benchmark foundation:
  - `enkai bench run`
  - deterministic suite harness in `bench/`
  - machine profiles + structured benchmark artifacts (`bench/results/*.json`)
- Added model lifecycle CLI foundation:
  - `enkai model register|list|promote|retire|rollback`
  - active-version and checkpoint-pointer based serve selection support
- Added data/algorithm stdlib foundation:
  - `std::analysis` (CSV/JSONL + schema/filter/project/group/describe/histogram)
  - `std::algo` (sort/search/path + ML metrics)
- Added CI benchmark smoke lane (`benchmark-smoke`) for reproducible suite execution.

v2.0.0 (done)
- Enforced strict train/eval contract checks by default:
  - `enkai train <config> --strict-contracts`
  - `enkai eval <config> --strict-contracts`
  - default behavior is strict in v2.0.0
  - temporary legacy recovery is explicit: `--lenient-contracts` with `ENKAI_ALLOW_LEGACY_CONTRACTS=1`
- Added compatibility wrappers for this release line:
  - `scripts/v2_0_0_rc_pipeline.ps1`
  - `scripts/v2_0_0_rc_pipeline.sh`
- Added strict checkpoint-meta verification mode:
  - `enkai migrate checkpoint-meta-v1 <checkpoint_dir> --verify --strict-contracts`
- Hardened doctor command for machine and operator workflows:
  - strict-by-default readiness scan
  - `--json` output mode
  - optional `--lenient` downgrade mode for transition audits.

v1.9.8 (done)
- Added RC pipeline gates with GPU evidence requirement by default:
  - `scripts/rc_pipeline.ps1`
  - `scripts/rc_pipeline.sh`
  - wrappers: `scripts/v1_9_8_rc_pipeline.ps1/.sh`
- Added release evidence archival tooling:
  - `scripts/collect_release_evidence.py` (writes `artifacts/release/v<version>/manifest.json`)
- Published v2.0 RC notes and migration guide:
  - `docs/31_v2_rc_notes.md`
  - `docs/32_v2_migration_guide.md`
- Locked RC freeze discipline:
  - no syntax expansion
  - bootstrap maintenance-only
  - migration + reliability only in RC cycle.

v1.9.7 (done)
- Added deterministic, script-driven release packaging:
  - `scripts/package_release.py`
  - `scripts/verify_release_artifact.py`
- Added version-neutral release pipeline scripts:
  - `scripts/release_pipeline.ps1`
  - `scripts/release_pipeline.sh`
- Kept backward-compatible wrappers:
  - `scripts/v1_9_release_pipeline.ps1`
  - `scripts/v1_9_release_pipeline.sh`
- Added provenance/security tooling:
  - `scripts/license_audit.py`
  - `scripts/generate_sbom.py`
- Updated CI/release workflows for cross-platform package checksum verification and SBOM artifacts.

v1.9.6 (done)
- Froze serve/frontend compatibility with explicit contract snapshots:
  - `backend/contracts/backend_api.snapshot.json`
  - `backend/contracts/conversation_state.schema.json`
  - `frontend/contracts/sdk_api.snapshot.json`
- Added CI/release pipeline contract snapshot gate:
  - `frontend::tests::contract_snapshots_match_reference_files`
- Hardened generated backend persistence contract:
  - `conversation_state.json` now schema-versioned (`schema_version: 1`)
  - startup migration hook for legacy v0-style conversation state.
- Expanded generated backend/SDK contract for WebSocket streaming route:
  - `GET /api/<version>/chat/ws`
  - SDK `streamChatWs(...)` helper.

v1.9 (done)
- Added stage1 execution command:
  - `enkai litec run <input.enk>`
- Added master pipeline smoke test (`master_pipeline_cpu_smoke`) covering train/eval + frontend scaffold + self-host checks.
- Added GPU evidence verification scripts:
  - `scripts/verify_gpu_gates.ps1`
  - `scripts/verify_gpu_gates.sh`
- Added consolidated v1.9 release pipeline scripts:
  - `scripts/v1_9_release_pipeline.ps1`
  - `scripts/v1_9_release_pipeline.sh`
- Updated CI with v1.9 release pipeline lane.
- Added replacement-readiness fixed-point command:
  - `enkai litec replace-check <corpus_dir> [--no-compare-stage0]`
- Added explicit distributed runtime opt-in gate:
  - `ENKAI_ENABLE_DIST=1` required for multi-rank mode.

v1.8 (done)
- Added compatibility/deprecation policy documentation (`docs/29_compatibility_policy.md`).
- Added self-host day-to-day workflow + fallback guide (`docs/28_selfhost_workflow.md`).
- Added v1.8 release pipeline scripts:
  - `scripts/v1_8_release_pipeline.ps1`
  - `scripts/v1_8_release_pipeline.sh`
- Added compatibility tests for:
  - legacy train config without `config_version`
  - legacy checkpoint metadata without `format_version`
- Added runtime warning path for legacy config parsing in train/eval.

v1.7 (done)
- Added bootstrap self-host beta command:
  - `enkai litec selfhost <corpus_dir>`
- Added staged frontend command surface:
  - `enkai litec stage <parse|check|codegen> <input.enk> [--out <program.bin>]`
- Added self-host CI validation command:
  - `enkai litec selfhost-ci <corpus_dir> [--no-compare-stage0]`
- Expanded bootstrap-core subset for self-host corpus validation:
  - allow `use`, `type`, `enum`, `impl`
  - allow non-capturing lambda expressions
- Added CI self-host lane for `litec selfhost` and `litec selfhost-ci` coverage.

v1.6 (done)
- `enkai litec` bootstrap-core command surface:
  - `enkai litec check <input.enk>`
  - `enkai litec compile <input.enk> --out <program.bin>`
  - `enkai litec verify <input.enk>`
- Added runtime/compiler `compiler` module (`parse_subset`, `check_subset`, `emit_subset`).
- Added stage0/stage1 bytecode equivalence verification and bootstrap-core tests.

v1.5 (done)
- Bootstrap-lite command surface:
  - `enkai fmt-lite`
  - `enkai lint-lite`
  - `enkai tokenizer-lite`
  - `enkai dataset-lite`
- Enkai-scripted bootstrap tooling pipeline with deterministic parity tests against Rust paths.
- Bootstrap subset specification finalized (`docs/bootstrap_subset.md`).

v1.4 (done)
- Added frontend developer stack commands:
  - `enkai new backend`
  - `enkai new frontend-chat`
  - `enkai new fullstack-chat`
- Added typed SDK generation:
  - `enkai sdk generate <output_file> [--api-version <v>]`
- Added end-to-end contract coverage for generated backend/frontend streaming flows and API-version pinning.
- Added persisted conversation flow in backend scaffold.

v1.3 (done)
- Added serving/backend runtime and CLI:
  - `enkai serve`
  - routed HTTP + middleware + SSE/WebSocket streaming token flow
- Added auth/rate-limit middleware baseline and structured request/response metadata.
- Added model registry version-pinning helpers.
- Added stdlib modules for backend integration:
  - `std::db` (SQLite + Postgres)
  - `std::tls`

v1.2 (done)
- Added scale/runtime training controls:
  - multi-rank config wiring
  - grad accumulation
  - grad clipping
  - AMP config path
- Added ranked checkpoint manifests and compatibility handling.
- Added dataset prefetch and packing-efficiency metrics.
- Added best-effort GPU metrics sampling (`gpu_mem_mb`, `gpu_util`) for CUDA configs.
- Added deterministic resolver + lockfile + build cache via `enkai build`.
- Added stdlib systems modules:
  - `std::env`, `std::path`, `std::process`, `std::time`, `std::io`, `std::log`

v1.1 (done)
- Implemented runtime semantics for `type` / `enum` / `impl` method dispatch.
- Implemented runtime semantics for `tool` / `agent` / `prompt` / `model` / `memory`.
- Added first-class ML stdlib:
  - `std::nn`
  - `std::loss`
  - `std::optim`
- Added deterministic seed wiring and checker/runtime tensor-related surface coverage.

v1.0 (done)
- Froze grammar/CLI compatibility to v0.9.3 baseline.
- Enforced train/eval config schema v1 (`config_version: 1`).
- Enforced checkpoint metadata format v1 (`format_version: 1`) with legacy fallback.
- Locked training path to TinyLM CE forward.
- Formalized release validation process (`VALIDATION.md`, `docs/RELEASE_CHECKLIST.md`).

v0.1 (done)
- Lexer/parser/AST with :: blocks
- Tree-walk interpreter
- Minimal types + control flow
- Minimal stdlib stubs
- CLI run/fmt/test stubs

v0.2 (done)
- Modules + use resolution
- Enkai run . with Enkai.toml + src/main.enk
- Better diagnostics (line/col + snippet)
- Runtime stack traces
- Minimal formatter + validation
- Policy enforcement MVP (default deny + allow rules)

v0.3 (done)
- Module exports/import rules (pub/private, re-export)
- Policy filters enforcement (domains, path_prefix)
- Diagnostics with labeled spans
- Local path dependencies in Enkai.toml
- Expand stdlib: strings + fs (policy-gated)
- Keep AI primitives as stubs unless testable



