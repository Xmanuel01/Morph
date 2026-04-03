ENKAI PROGRAMMING LANGUAGE
[![CI](https://github.com/Xmanuel01/Enkai/actions/workflows/ci.yml/badge.svg)](https://github.com/Xmanuel01/Enkai/actions/workflows/ci.yml)

Overview
Enkai is a programming language with block structure defined by :: tokens, a clean
assignment operator (:=), and an AI-native roadmap (tools, agents, memory, policy).
This repository contains the v2.9.0 implementation in Rust.

Status (v2.9.0)
- Bytecode VM + globals + type-checking
- Module system with public/private exports
- CLI: run/bench/readiness/deploy/model/serve/new/sdk/check/fmt/fmt-lite/lint-lite/tokenizer-lite/dataset-lite/litec/build/test/train/pretrain/eval/migrate/doctor
- FFI runtime + native std modules (fsx/zstd/hash/db/tls)
- Core simulation std/runtime modules:
- `std::sparse`
- `std::event`
- `std::pool`
- `std::sim`
- `std::spatial`
- `std::snn`
- `std::agent`
- Coroutine-facing simulation APIs under `std::sim`:
  - `sim.coroutine`
  - `sim.coroutine_with`
  - `sim.coroutine_args`
  - `sim.emit`
  - `sim.next`
  - `sim.join`
  - `sim.done`
- `std::sparse`, `std::event`, and `std::pool` now keep the same public API while using
  native-backed acceleration through `enkai_native` when available; deterministic runtime
  fallbacks remain active when acceleration is unavailable or disabled with `ENKAI_SIM_ACCEL=0`
- Adam-0 reference stack and release evidence:
  - deterministic baseline: `examples/adam0_100.enk`
  - bounded reference suite: `examples/adam0_reference.enk`
  - suite definition: `bench/suites/adam0_reference_v2_7_1.json`
  - smoke evidence: `scripts/readiness_adam0_smoke.py`
  - smoke verification: `scripts/verify_adam0_evidence.py`
  - suite evidence: `scripts/readiness_adam0_reference_suite.py`
  - suite verification: `scripts/verify_adam0_reference_suite.py`
- SNN + agent environment kernel baseline and release evidence:
  - `examples/snn_agent_kernel.enk`
  - `scripts/readiness_snn_agent_kernel_smoke.py`
  - `scripts/verify_snn_agent_kernel_evidence.py`
- Simulation CLI surfaces:
  - `enkai sim run`
  - `enkai sim profile`
  - `enkai sim replay`
- Additive data/algorithm std modules:
  - `std::analysis` (CSV/JSONL ingest + typed schema inference/validation + filter/project/join/group aggregates + describe/histogram/quantiles/rolling/pipeline)
  - `std::algo` (sort/search/path + priority/merge/window/cumulative transforms + ML metrics/eval/scheduler + deterministic split helpers)
- Tokenizer + dataset streaming + checkpoints
- Backend serving stack: routing, middleware/auth/rate-limit, HTTP + gRPC + SSE/WebSocket streaming, TLS/SQLite/Postgres/MySQL helpers
- Serving hardening: request correlation IDs, queue/inflight telemetry headers, deterministic JSON error codes, backpressure control, and model-version enforcement hooks
- Model lifecycle registry hardening:
  - local registry lifecycle (`register|list|load|unload|promote|retire|rollback`)
  - remote registry sync (`push|pull|promote-remote|retire-remote|rollback-remote|verify-signature`)
  - immutable remote artifact manifests + optional signature verification
  - append-only model lifecycle audit log (`audit.log.jsonl`)
  - additive artifact kinds:
    - `checkpoint`
    - `simulation`
    - `environment`
    - `native-extension`
  - simulation lineage/snapshot manifests can be registered and remotely signed through the same lifecycle flow
- Frontend stack: React/TypeScript scaffolds + typed SDK generation
- Serve/frontend contract snapshots and compatibility freeze gates for generated backend + SDK
- Schema-versioned conversation persistence (`schema_version: 1`) with startup migration hook for legacy scaffold state
- Fullstack platform completion:
  - expanded scaffolds: `service`, `llm-backend`, `llm-fullstack`
  - deployment env contract snapshot + validator scaffold (`contracts/deploy_env.snapshot.json`, `scripts/validate_env_contract.py`)
  - migration assets scaffold (`migrations/001_conversation_state.sql`, `migrations/002_conversation_state_index.sql`)
  - generated fullstack upgrade contract tests with persistence migration verification
- Bootstrap-lite/core toolchain path with `litec` stage0/stage1 bytecode equivalence checks, phase staging (`litec stage`), self-host CI corpus validation (`litec selfhost-ci`), and consolidated release lane (`litec release-ci`)
- Self-host mainline CI lane with deterministic triage artifacts (`litec mainline-ci --triage-dir <dir>`) plus mandatory Stage0 fallback lane
- Self-host replacement-readiness gate with Stage1/Stage2 fixed-point checks (`litec replace-check`)
- Compatibility/deprecation governance and self-host fallback workflow docs for v2.9.0 release readiness
- Version-neutral release pipeline, deterministic packaging, checksum verification, SBOM generation, and RC evidence-archive gates for v2.9.0 sign-off
- Full-platform simulation smoke evidence integrated into release sign-off:
  - `scripts/readiness_sim_smoke.py`
  - `artifacts/readiness/sim_smoke.json`
  - `artifacts/readiness/sim_evidence_verify.json`
  - `artifacts/sim/smoke_run.json`
  - `artifacts/sim/smoke_profile.json`
  - `artifacts/sim/smoke_replay.json`
- Full-platform SNN/agent kernel evidence integrated into release sign-off:
  - `artifacts/readiness/snn_agent_kernel_smoke.json`
  - `artifacts/readiness/snn_agent_kernel_evidence_verify.json`
  - `artifacts/sim/snn_agent_kernel_run.json`
  - `artifacts/sim/snn_agent_kernel_profile.json`
- Adam-0 reference suite evidence integrated into release sign-off:
  - `artifacts/readiness/adam0_reference_suite.json`
  - `artifacts/readiness/adam0_reference_suite_verify.json`
  - `artifacts/sim/adam0_baseline_100_run.json`
  - `artifacts/sim/adam0_stress_1000_run.json`
  - `artifacts/sim/adam0_target_10000_run.json`
- Registry convergence evidence integrated into release sign-off:
  - `scripts/readiness_registry_convergence.py`
  - `scripts/verify_registry_convergence.py`
  - `artifacts/readiness/model_registry_convergence.json`
  - `artifacts/readiness/model_registry_convergence_verify.json`
  - `artifacts/registry/sim_lineage.json`
  - `artifacts/registry/sim_snapshot.manifest.json`
- Multi-node simulation cluster scale evidence integrated into release sign-off:
  - `scripts/readiness_cluster_scale_smoke.py`
  - `scripts/verify_cluster_scale_evidence.py`
  - `artifacts/readiness/cluster_scale_smoke.json`
  - `artifacts/readiness/cluster_scale_evidence_verify.json`
  - `artifacts/cluster_scale/run.json`
- Registry degraded-mode fallback evidence integrated into release sign-off:
  - `scripts/readiness_registry_degraded_smoke.py`
  - `scripts/verify_registry_degraded_evidence.py`
  - `artifacts/readiness/registry_degraded_smoke.json`
  - `artifacts/readiness/registry_degraded_evidence_verify.json`
  - `artifacts/registry_degraded/cache/audit.log.jsonl`
- gRPC runtime evidence integrated into release sign-off:
  - `scripts/readiness_grpc_smoke.py`
  - `scripts/verify_grpc_evidence.py`
  - `artifacts/readiness/grpc_smoke.json`
  - `artifacts/readiness/grpc_evidence_verify.json`
  - `artifacts/grpc/probe.json`
  - `artifacts/grpc/server.jsonl`
- Real GPU host preflight for final hardware sign-off:
  - `scripts/gpu_preflight.py`
  - `scripts/gpu_preflight.ps1`
  - `scripts/gpu_preflight.sh`
  - `docs/47_gpu_operator_preflight.md`
- Capability-complete release report generated from archived evidence:
  - `scripts/collect_release_evidence.py --strict`
  - `scripts/generate_capability_report.py --strict`
- Production readiness matrix + report command:
  - `docs/37_readiness_matrix.md`
  - `enkai readiness check --profile production --json --output artifacts/readiness/production.json`
- Benchmark foundation for bounded Enkai-vs-Python claims:
  - `enkai bench run --suite official_v2_3_0_matrix ...`
  - `enkai bench run --suite algorithm_kernels ...`
  - fairness guardrail:
    - `enkai bench run --suite official_v2_3_0_matrix --fairness-check-only ...`
  - per-case profiling:
    - `enkai bench profile --case <id> --output bench/results/profiles/<id>.json`
  - target policy controls:
    - `--enforce-target` (suite median target gate)
    - `--enforce-all-cases` (strict per-case target gate)
    - `--enforce-class-targets --class-targets bench/suites/official_v2_3_0_targets.json` (class matrix gate)
  - deterministic suites under `bench/suites/`
  - machine profile manifests under `bench/machines/`
  - structured result artifacts under `bench/results/*.json`
- Strict-contract enforcement in v2.9.0:
  - `enkai train` / `enkai eval` enforce contract checks by default
  - explicit legacy recovery is gated: `--lenient-contracts` + `ENKAI_ALLOW_LEGACY_CONTRACTS=1`
  - readiness audit: `enkai doctor --json [--strict-contracts|--lenient]`
- Additive distributed orchestration controls for multi-rank training:
  - config fields: `dist.topology`, `dist.rendezvous`, `dist.retry_budget`, `dist.device_map`
  - additive multi-node host/simulation partition controls:
    - `dist.hosts`
    - `dist.host_map`
    - `workload = "simulation"`
    - `simulation.target`, `simulation.partition_count`, `simulation.total_steps`
    - `simulation.step_window`, `simulation.snapshot_interval`, `simulation.recovery_dir`, `simulation.route_policy`
  - cluster supervision commands:
    - `enkai cluster validate <config.enk> [--json] [--output <file>]`
    - `enkai cluster plan <config.enk> [--json] [--output <file>]`
    - `enkai cluster run <config.enk> [--dry-run] [--json] [--output <file>]`
  - `enkai cluster run` now executes bounded supervised simulation workloads with snapshot/replay recovery; train multi-node execution remains operator-managed
- Pretraining lifecycle metadata (additive):
  - `enkai pretrain <config.enk>` shares the train/eval config contract
  - writes `run_state.json`, `runs/index.jsonl`, and `checkpoint_lifecycle.json` under `checkpoint_dir`

Workspace structure
- enkaic: compiler front-end (lexer/parser/AST + production type-checking)
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









