Enkai Roadmap

Note:
- Historical milestones below capture the path that led to current releases.
- Current production release line is v2.1.0.
- v2.1.x is additive/integration work in progress (no contract-breaking removals).
- Use `docs/Enkai.spec` as the source of truth for current language behavior.

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



