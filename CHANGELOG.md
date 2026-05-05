# Changelog

## Unreleased

### Breaking changes
- None.

## v3.7.0

### Highlights
- Started the global self-host AI runtime foundation line.
- Added the first concrete `v3.7.0` implementation tranche:
  - global self-host AI runtime foundation
- Added the second `v3.7.0` tranche for performance deltas.
- Added the third `v3.7.0` tranche for threaded acceleration determinism.
- Added the fourth `v3.7.0` tranche for broader model-shape coverage and explicit latency baselines.
- Added the first distributed-runtime design tranche for `v3.7.0` without widening execution scope.
- Added the fifth `v3.7.0` tranche for explicit checkpoint/eval throughput regression gates.
- Added the first executable distributed-runtime tranche for `v3.7.0` as a rank-sharded `enkai_accel` preview.
- Added the sixth `v3.7.0` tranche for synchronized-gradient distributed preview and checkpoint merge/replay proof.
- Added the seventh `v3.7.0` tranche for synchronized distributed shape-envelope expansion and throughput gates.
- Added the networked multi-process rendezvous design-freeze tranche for `v3.7.0`.
- Added the executable networked rendezvous tranche for `v3.7.0` with bounded barrier/retry fault injection.
- Added the realistic AI workload benchmark matrix tranche for `v3.7.0`.
- Added the adversarial input corruption tranche for `v3.7.0`.
- Added the AI runtime security and fault baseline tranche for `v3.7.0`.
- Added the bounded AI runtime QA floor tranche for `v3.7.0`.
- Added the larger-world-size networked rendezvous design-freeze tranche for `v3.7.0`.
- Added the world-size 4 networked rendezvous execution tranche for `v3.7.0`.
- Added tcp:// networked gradient exchange and adversarial transport coverage for `v3.7.0`.
- Added adversarial peer-behavior coverage for the `v3.7.0` networked rendezvous surface.
- Added longer-context synchronized distributed workload coverage for `v3.7.0`.
- Added networked long-context execution proof for `v3.7.0`.
- Added networked throughput regression gates for `v3.7.0`.
- Added the full `v3.7.0` closure verifier.
- Added the bounded `enkai_accel` backend class for:
  - `enkai train`
  - `enkai pretrain`
  - `enkai eval`
- Replaced the original tiny benchmark with a stronger pinned benchmark suite and archived explicit deltas against Python, the CPU scalar path, and the current `native` comparison path.
- Promoted `enkai_accel` to a deterministic multithreaded execution path and archived repeated-run report/checkpoint hash stability proof.
- Added bounded residual-stack model-shape coverage plus checkpoint-resume and eval-only latency baselines.
- Added a broader bounded transformer-shape frontier with explicit `silu` activation coverage and throughput gates.
- Added bounded proof artifacts for:
  - functional train/pretrain/eval execution on the frozen suite
  - deterministic checkpoint save/load and runtime reporting
  - benchmark comparison against Python and the current `native` backend
  - bounded memory/safety/security validation

## v3.5.0

### Highlights
- Closed the `v3.5.0` bounded post-closure objective-definition line.
- Completed the defined `v3.5.0` machine-verifiable scopes:
  - release-line start baseline
  - objective-set freeze baseline
  - evidence continuity baseline

## v3.4.0

### Highlights
- Closed the `v3.4.0` post-closure baseline line after the `v3.3.0` strict-selfhost shipped-surface release.
- Completed the defined `v3.4.0` machine-verifiable scopes:
  - release-line normalization baseline
  - capability-reporting normalization baseline
  - release-evidence/dashboard historical consistency surface
  - zero-Rust next-step baseline
  - cross-host install proof matrix baseline
  - compatibility-only storage/data path baseline
  - accelerated native/tensor backend baseline
  - historical non-shipped compatibility-path closure baseline

## v3.3.0

### Highlights
- Closed the strict-selfhost shipped-surface objective set for the `v3.3.0` line.
- Completed all six CLI/system ownership surfaces:
  - `project_entrypoints`
  - `readiness_and_validation`
  - `serve_and_transports`
  - `worker_runtime`
  - `model_and_registry_ops`
  - `deploy_and_cluster_ops`
- Completed all six strict-selfhost dependency components:
  - `compiler_frontend`
  - `runtime_core`
  - `systems_and_cli`
  - `native_std_and_accel`
  - `tensor_backend`
  - `data_registry_protocols`
- The strict-selfhost source-of-record artifacts now show:
  - `artifacts/readiness/strict_selfhost_cli_system_slices.json`: all surfaces `done`
  - `artifacts/readiness/strict_selfhost_dependency_inventory.json`: all components `done`, `strict_selfhost_cpu_complete = true`
- Added the `v3.3.0` release-candidate wrapper scripts and bumped the release line to `3.3.0`.

## v3.2.1

### Highlights
- Closed the `v3.2.1` installable zero-Rust toolchain tranche for the current
  shipped path by adding versioned bundle/install contracts and executable proof:
  - `enkai/contracts/install_bundle_v3_2_1.json`
  - `enkai/contracts/zero_rust_closure_v3_2_1.json`
  - `enkai/contracts/install_flow_v3_2_1.json`
  - `enkai/contracts/install_flow_v3_2_1_windows.json`
  - `enkai/contracts/install_flow_v3_2_1_linux.json`
- Added concrete install lifecycle proof for staged bundles:
  - deterministic archive packaging
  - install
  - upgrade
  - uninstall
  - installed `run`/`check`/`build`/`test` self-host entrypoint execution
  - install and bundle manifest validation through `enkai install-diagnostics`
- Added release-archive proof to the install-flow gate:
  - `scripts/package_release.py`
  - `scripts/verify_install_bundle.py`
  - `scripts/verify_install_flows.py`
  - `scripts/verify_release_artifact.py`
- Promoted install-flow proof into strict self-host readiness/blockers for the
  shipped toolchain path while keeping Linux host execution truthfully separate
  from the Windows-host evidence produced in-repo.
- Added `v3.2.0` release-candidate wrapper scripts for the finalized runtime
  tranche and `v3.2.1` wrapper scripts for the installable-toolchain tranche.

## v3.2.0

### Highlights
- Completed the `v3.2.0` self-host runtime-core proof tranche:
  - full shipped-example runtime parity proof
  - full shipped-example runtime instruction coverage proof
  - task/channel/scheduler runtime proof
  - repeated event/coroutine/replay determinism proof
  - stable runtime error taxonomy proof
- Moved readiness profile execution and release blocker verification behind
  contract-driven executors in:
  - `enkai/src/readiness.rs`
  - `enkai/contracts/strict_selfhost_readiness_validation_slices_v3_2_0.json`
- Added runtime determinism verification and promoted it into strict self-host
  release blockers:
  - `scripts/verify_selfhost_runtime_determinism.py`
  - `artifacts/readiness/selfhost_runtime_determinism_verify.json`
- Tightened strict self-host readiness so `v3.2.0` now requires:
  - runtime task/channel parity
  - runtime instruction-surface coverage over shipped examples
  - runtime example parity over shipped examples
  - repeated deterministic runtime behavior for the audited simulation/event corpus
  - runtime error taxonomy stability
- Kept zero-Rust closure truthfully out of scope for this cut:
  - `v3.2.0` closes the self-host runtime-core tranche
  - remaining Rust-owned shipped dependencies remain tracked for `v3.2.1+`

## v3.1.2

### Highlights
- Completed the remaining `v3.1.1` frontend-proof gap by adding a curated
  audited-surface bundled verifier:
  - `enkai/contracts/selfhost_audited_surface_v3_1_1.json`
  - `scripts/verify_selfhost_audited_surface.py`
  - `artifacts/readiness/selfhost_audited_surface_verify.json`
- Tightened strict self-host readiness so the completed `v3.1.1` tranche now requires:
  - frontier proof
  - shipped examples proof
  - bootstrap compiler source proof
  - curated negative semantic proof
  - curated audited executable surface proof through `frontend-audit`,
    `selfhost-ci`, `replace-check`, and `mainline-ci`
- Added the `v3.1.0` strict self-host contract freeze:
  - `strict_selfhost` readiness profile
  - strict self-host blocker manifest
  - machine-readable dependency board and generated inventory
  - release dashboard visibility for remaining Rust-owned shipped dependencies
- Added the `v3.1.1` full frontend frontier audit tranche:
  - `enkai litec frontend-audit <corpus_dir>`
  - deterministic `litec_frontend_audit_report.json`
  - stage0/stage2 acceptance, bytecode parity, and runtime parity reporting
  - package-aware bootstrap compiler intrinsics:
    - `compiler.parse_subset_file`
    - `compiler.check_subset_file`
    - `compiler.emit_subset_file`
  - expanded bootstrap subset declaration support for:
    - `native::import`
    - `tool`
    - `prompt`
    - `model`
    - `agent`
  - widened strict self-host proof so the shipped `examples/` corpus is
    release-checked via:
    - `enkai/contracts/selfhost_examples_v3_1_1.json`
    - `artifacts/readiness/selfhost_examples_verify.json`
  - widened strict self-host proof again so bootstrap compiler sources and a
    curated negative semantic corpus are release-checked via:
    - `enkai/contracts/selfhost_bootstrap_v3_1_1.json`
    - `enkai/contracts/selfhost_negative_v3_1_1.json`
    - `artifacts/readiness/selfhost_bootstrap_verify.json`
    - `artifacts/readiness/selfhost_negative_verify.json`
  - added `enkai litec negative-audit <corpus_dir|file.enk>` for deterministic
    stage0/self-host rejection auditing on curated semantic-failure corpora
  - hardened frontier verification to match relative file paths, not only
    basenames
  - moved another concrete bootstrap frontend slice into Enkai:
    - `compiler.describe_subset`
    - `compiler.describe_subset_file`
    - `compiler.describe_subset_package_file`
    - `compiler.describe_program_file`
    - `compiler.check_subset_raw`
    - `compiler.check_subset_raw_file`
    - `compiler.emit_subset_raw`
    - `compiler.emit_subset_raw_file`
  - the staged bootstrap compiler in `enkai/tools/bootstrap/enkai_lite.enk`
    now performs non-recursive subset-shape validation over structural compiler
    output before invoking raw typecheck/codegen paths
  - the staged bootstrap compiler now also validates imported package modules
    on the self-host path before invoking raw package-aware typecheck/codegen
  - the staged bootstrap compiler now owns another semantic slice:
    - duplicate top-level symbol rejection
    - duplicate import binding rejection
    - import binding/type collisions
    - impl target/type existence checks
    - constructor/local/imported arity checks
  - the staged bootstrap compiler now validates emitted bytecode summaries on
    the self-host path before accepting codegen output

### Pending release blockers
- Real GPU/operator evidence is still required for final hardware sign-off:
  - single GPU evidence
  - multi-GPU parity evidence
  - 4-GPU soak evidence

## v3.0.0

### Highlights
- Advanced the release line to `v3.0.0` with CPU-complete proof gates, publication assets, and final QA documentation.
- Reworked strict capability reporting so non-hardware production claims are derived from archived proof groups instead of individual path presence alone.
- Added a release dashboard that makes CPU-complete vs GPU-pending status explicit for operators.
- Published the final QA findings document for the `v2.9.1` -> `v3.0.0` quality-recovery program.

### Fixes
- Added `scripts/generate_release_dashboard.py` and wired it into the release and RC pipelines.
- Aggregated CPU correctness, determinism, runtime safety, and Adam-0 proof suites into strict capability checks.
- Published `artifacts/release/v<version>/release_dashboard.json` and `artifacts/release/v<version>/release_dashboard.md` as operator-facing summaries.
- Added CI and docs consistency coverage for the release dashboard generation path.
- Added `docs/49_v3_0_0_quality_assurance.md` to record the final CPU-side QA results, remaining GPU blocker, and evidence locations.

## v2.9.5

### Highlights
- Advanced the release line to `v2.9.5` with runtime safety hardening and proof-backed FFI fault handling.
- Promoted FFI invalid-handle, double-free, corrupted-replay, and fault-injection coverage into release-blocking evidence.
- Added archived runtime safety summary and verification artifacts to strict release evidence and capability reporting.

### Fixes
- Added `enkai validate ffi-safety` and `examples/validation/ffi_safety.enk`.
- Hardened `enkai_native` opaque-handle tracking so stale, wrong-kind, and double-freed handles are rejected without crashing and counted deterministically.
- Added native/runtime regression tests for invalid handles, double free, null returns, oversized buffers, invalid UTF-8, and corrupted simulation replay.
- Added `scripts/readiness_runtime_safety.py` and `scripts/verify_runtime_safety.py`.
- Promoted `artifacts/validation/ffi_safety.json`, `artifacts/readiness/runtime_safety.json`, and `artifacts/readiness/runtime_safety_verify.json` into readiness, blocker verification, strict release evidence, and capability reporting.

## v2.9.4

### Highlights
- Advanced the release line to `v2.9.4` with Adam-0 CPU reference proof completion.
- Replaced the weak Adam-0 smoke posture with a bounded validation ladder for 10 / 100 / 1000 / 10000 CPU cases.
- Tightened Adam-0 evidence so native kernel dominance, marshal-copy budgets, and replay/state consistency are machine-checked.

### Fixes
- Added kernel-native counters for sparse/event/pool/spatial/SNN native paths in the runtime bench profile output.
- Added the lighter official fake-Adam package under `examples/adam0_fake/`.
- Added the bounded Adam-0 CPU proof suite definition: `bench/suites/adam0_reference_v2_9_4.json`.
- Tightened `enkai validate adam0-cpu` and `enkai validate perf-baseline` so Adam-0 cases require native-counter minima, kernel-native dominance, and marshal-copy ratio budgets.
- Hardened Adam-0 suite readiness verification to compare replayed state projections, hardware assumptions, and case-level native-kernel proof counters.
- Promoted the 1000-agent and 10000-agent CPU validation/perf artifacts into readiness, blocker verification, strict release evidence, and capability reporting.

## v2.9.3

### Highlights
- Advanced the release line to `v2.9.3` with scheduler, replay, and coroutine proof completion.
- Added release-blocking deterministic coroutine validation and simulation audit payloads.
- Tightened deterministic replay validation so snapshot/replay hash equality is machine-checked.

### Fixes
- Added `sim_coroutines` to the validation manifest and readiness/blocker matrices.
- Added `examples/validation/determinism_sim_coroutines.enk` as the canonical coroutine determinism suite.
- Extended `enkai validate determinism` to emit simulation audits with:
  - seed
  - config hash
  - event log hash
  - snapshot hash
  - replay hash
  - coroutine/task counters
- Extended determinism validation to require coroutine counters when the suite declares them.
- Added interpreter regression coverage for native-vs-VM coroutine equivalence.
- Synced release evidence requirements so coroutine determinism artifacts are archived and verified.

## v2.9.2

### Highlights
- Advanced the release line to `v2.9.2` with proof completion for `std::sparse`, `std::event`, and `std::pool`.
- Tightened validation artifacts so native-backed and VM-fallback behaviors must match exactly.
- Promoted seeded equivalence/property tests into the workspace test gate for the simulation primitives.

### Fixes
- Expanded `ffi_correctness` validation to prove:
  - exact sparse/event/pool outputs
  - native-backed and VM-fallback result equality
  - native path exercise
  - zero leaked native handles after the run
- Expanded deterministic event validation to prove:
  - expected ordering/tie-breaking output
  - 10-run output hash stability
  - native-backed and VM-fallback result/hash equality
  - native queue path exercise
- Expanded `pool_safety` validation to prove:
  - reset semantics
  - plateau/high-watermark behavior
  - native-backed and VM-fallback result equality
  - native pool path exercise
- Added interpreter/native tests for:
  - sparse invalid-index handling
  - sparse dense-length mismatch semantics
  - event `peek/len/is_empty` semantics
  - pool reset semantics
  - seeded native-vs-VM equivalence for sparse/event/pool
- Documented the exact public semantics for `std::sparse`, `std::event`, and `std::pool` in `docs/Enkai.spec`.

## v2.9.1

### Highlights
- Advanced the release line to `v2.9.1` with the first quality-recovery milestone completed.
- Added proof-grade CPU validation commands and archived validation artifacts.
- Tightened readiness, strict evidence, and capability reporting so production claims require proof artifacts.

### Fixes
- Added `enkai validate` subcommands for:
  - FFI correctness
  - determinism
  - performance baselines
  - pool safety
  - Adam-0 CPU validation
- Added manifest-driven CPU validation policy and local/reference machine profiles.
- Added deterministic Adam-0 100-agent determinism evidence into the readiness and release blocker path.
- Added baseline-driven performance validation with median sampling and archived proof artifacts.
- Synced docs/spec/release metadata and version surfaces to `v2.9.1`.

## v2.9.0

### Highlights
- Advanced the release line to `v2.9.0` with additive compatibility and no syntax changes.
- Completed the `v2.9.0` full-platform completion milestone.
- Added real gRPC serve/runtime support alongside the existing HTTP/SSE/WebSocket stack.
- Added release-blocking gRPC smoke and evidence verification to the full-platform readiness path.

### Fixes
- Added a real tonic-backed gRPC server/client runtime behind `enkai serve --grpc-port ...` and `enkai grpc probe ...`.
- Added gRPC persistence/logging evidence artifacts and strict archive verification:
  - `scripts/readiness_grpc_smoke.py`
  - `scripts/verify_grpc_evidence.py`
- Integrated gRPC runtime evidence into:
  - full-platform readiness
  - blocker verification
  - strict release evidence
  - capability reporting
- Completed the remaining `v2.9.0` platform surfaces already staged on the branch:
  - real MySQL runtime support
  - durable worker queue runtime + evidence gates
  - mobile scaffold/deploy validation + evidence gates
- Fixed Windows release-pipeline metadata decoding issues in:
  - `scripts/license_audit.py`
  - `scripts/generate_sbom.py`
- Synced docs/spec/release metadata and version surfaces to `v2.9.0`.

## v2.8.1

### Highlights
- Advanced the release line to `v2.8.1` with additive compatibility and no syntax changes.
- Completed the `v2.8.1` scale, multi-node, and reliability milestone.
- Added bounded simulation cluster supervision and degraded registry fallback evidence to the full-platform release gate.

### Fixes
- Extended `enkai cluster` with additive multi-node simulation fields:
  - `dist.hosts`
  - `dist.host_map`
  - `workload = "simulation"`
  - `simulation.target`
  - `simulation.partition_count`
  - `simulation.total_steps`
  - `simulation.step_window`
  - `simulation.snapshot_interval`
  - `simulation.recovery_dir`
  - `simulation.route_policy`
- Added supervised bounded simulation execution to `enkai cluster run` using windowed `enkai sim run` / `enkai sim replay`, persisted snapshots, and bounded retry/recovery.
- Added additive `--output <file>` support for `enkai cluster validate|plan|run`.
- Added additive `--snapshot-output <file>` support for `enkai sim run|profile|replay`.
- Added release-gated cluster scale smoke and semantic verification:
  - `scripts/readiness_cluster_scale_smoke.py`
  - `scripts/verify_cluster_scale_evidence.py`
- Added release-gated degraded registry fallback smoke and semantic verification:
  - `scripts/readiness_registry_degraded_smoke.py`
  - `scripts/verify_registry_degraded_evidence.py`
- Full-platform readiness, blocker verification, strict evidence archiving, and capability reporting now require archived cluster-scale and degraded-registry artifacts.
- Synced docs/spec/release metadata and version surfaces to `v2.8.1`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_8_1_rc_pipeline.ps1`
  - `scripts/v2_8_1_rc_pipeline.sh`

### Breaking changes
- None.

## v2.8.0

### Highlights
- Advanced the release line to `v2.8.0` with additive compatibility and no syntax changes.
- Completed the `v2.8.0` LLM + AGI data/training/registry convergence milestone.
- Unified the model registry lifecycle across artifact kinds:
  - `checkpoint`
  - `simulation`
  - `environment`
  - `native-extension`

### Fixes
- Added additive simulation lineage and world-snapshot manifest outputs to `enkai sim`:
  - `--lineage-output <file>`
  - `--snapshot-manifest-output <file>`
- Added additive registry metadata for `--artifact-kind`, `--artifact-manifest`, and `--lineage-manifest` on `enkai model register`.
- Added `enkai model verify-signature <registry_dir> <name> <version> --registry <remote_registry_dir>`.
- Added full-platform signed registry convergence smoke and semantic verification:
  - `scripts/readiness_registry_convergence.py`
  - `scripts/verify_registry_convergence.py`
- Full-platform readiness, blocker verification, strict evidence archiving, and capability reporting now require archived registry convergence artifacts under:
  - `artifacts/readiness/model_registry_convergence.json`
  - `artifacts/readiness/model_registry_convergence_verify.json`
  - `artifacts/registry/`
- Synced docs/spec/release metadata and version surfaces to `v2.8.0`.

### Breaking changes
- None.

## v2.7.1

### Highlights
- Advanced the release line to `v2.7.1` with additive compatibility and no syntax changes.
- Completed the `v2.7.1` Adam-0 reference stack milestone.
- Added the bounded Adam-0 reference suite:
  - `examples/adam0_reference.enk`
  - `bench/suites/adam0_reference_v2_9_4.json`

### Fixes
- Added release-gated Adam-0 reference suite generation and semantic verification:
  - `scripts/readiness_adam0_reference_suite.py`
  - `scripts/verify_adam0_reference_suite.py`
- Full-platform readiness, blocker verification, strict evidence archiving, and capability reporting now require archived Adam-0 reference suite artifacts for:
  - `adam0_baseline_100`
  - `adam0_stress_1000`
  - `adam0_target_10000`
- Hardened `enkai sim replay` so large simulation snapshots are compacted before inline restore, avoiding replay failures caused by pretty-printed JSON embedding.
- Synced docs/spec/release metadata and version surfaces to `v2.7.1`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_7_1_rc_pipeline.ps1`
  - `scripts/v2_7_1_rc_pipeline.sh`

### Breaking changes
- None.

## v2.7.0

### Highlights
- Advanced the release line to `v2.7.0` with additive compatibility and no syntax changes.
- Completed the `v2.7.0` SNN runtime + agent environment base milestone.
- Added additive simulation/runtime modules:
  - `std::spatial`
  - `std::snn`
  - `std::agent`
- Added the in-tree SNN/agent environment kernel reference workload:
  - `examples/snn_agent_kernel.enk`

### Fixes
- Added native-backed spatial query hooks behind the stable `std::spatial` interface:
  - `radius`
  - `nearest`
  - `occupancy`
- Added deterministic RNG streams for simulation workloads through `agent.stream`, `agent.next_float`, and `agent.next_int`.
- Added SNN runtime support under `std::snn` with native-backed batched neuron update hooks and stable synapse access through `SparseMatrix`.
- Added agent environment/runtime support under `std::agent` for:
  - registration
  - body/memory/state access
  - reward accounting
  - sensor/action queues
  - spatial neighbor queries
- Added release-gated SNN/agent kernel smoke + evidence verification:
  - `scripts/readiness_snn_agent_kernel_smoke.py`
  - `scripts/verify_snn_agent_kernel_evidence.py`
- Full-platform readiness, blocker verification, strict evidence archiving, and capability reporting now require archived SNN/agent kernel run/profile artifacts.
- Added root `enkai.toml` so in-tree examples resolve project-root modules and `std/` imports consistently.
- Synced docs/spec/release metadata and version surfaces to `v2.7.0`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_7_0_rc_pipeline.ps1`
  - `scripts/v2_7_0_rc_pipeline.sh`

### Breaking changes
- None.

## v2.6.9

### Highlights
- Advanced the release line to `v2.6.9` with additive compatibility and no syntax changes.
- Completed the missing `v2.6.2` coroutine/generator-facing simulation API requirement under `std::sim`.
- Added an in-tree Adam-0 reference prototype for the 100-agent deterministic baseline:
  - `examples/adam0_100.enk`
- Added release-gated Adam-0 evidence generation and verification:
  - `scripts/readiness_adam0_smoke.py`
  - `scripts/verify_adam0_evidence.py`

### Fixes
- Added `SimCoroutine` as a stable runtime/typechecker surface with deterministic task-backed yield/join behavior.
- Added `std::sim` coroutine APIs without changing language syntax:
  - `sim.coroutine`
  - `sim.coroutine_with`
  - `sim.coroutine_args`
  - `sim.world`
  - `sim.state`
  - `sim.emit`
  - `sim.next`
  - `sim.join`
  - `sim.done`
- Added VM bench profile counters for simulation coroutine activity so readiness artifacts can prove the coroutine path was exercised.
- Added release-blocking Adam-0 smoke/evidence artifacts:
  - `artifacts/readiness/adam0_100_smoke.json`
  - `artifacts/readiness/adam0_100_evidence_verify.json`
  - `artifacts/sim/adam0_100_run.json`
  - `artifacts/sim/adam0_100_profile.json`
- Synced docs/spec/release metadata and version surfaces to `v2.6.9`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_6_9_rc_pipeline.ps1`
  - `scripts/v2_6_9_rc_pipeline.sh`

### Breaking changes
- None.

## v2.6.8

### Highlights
- Advanced the release line to `v2.6.8` with additive compatibility and no syntax changes.
- Completed the missing `v2.6.1` native-backed acceleration requirement behind the existing public interfaces for:
  - `std::sparse`
  - `std::event`
  - `std::pool`
- Added internal simulation acceleration bindings that use `enkai_native` when available and fall back to deterministic runtime implementations when unavailable or disabled.

### Fixes
- `sparse.dot` and `sparse.matvec` now use native-backed handles and packed f64 buffers under the same Enkai API surface.
- `event.make`/`event.push`/`event.pop`/`event.peek`/`event.len` now maintain a native heap mirror behind the existing queue interface.
- `pool.make`/`pool.release`/`pool.acquire`/`pool.reset`/`pool.available`/`pool.capacity`/`pool.stats` now maintain native-backed capacity and counter state behind the existing pool interface.
- `scripts/readiness_sim_stdlib_smoke.py` and `scripts/verify_sim_stdlib_evidence.py` now prove that stdlib simulation primitives use the native acceleration path in profile artifacts instead of only the VM fallback path.
- Synced docs/spec/release metadata and version surfaces to `v2.6.8`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_6_8_rc_pipeline.ps1`
  - `scripts/v2_6_8_rc_pipeline.sh`

### Breaking changes
- None.

## v2.6.6

### Highlights
- Advanced the release line to `v2.6.6` with additive compatibility and no syntax changes.
- Hardened the AGI/simulation release path with native FFI smoke verification:
  - new smoke runner:
    - `scripts/readiness_sim_native_smoke.py`
  - new verifier:
    - `scripts/verify_sim_native_evidence.py`
  - new readiness artifacts:
    - `artifacts/readiness/sim_native_smoke.json`
    - `artifacts/readiness/sim_native_evidence_verify.json`

### Fixes
- Full-platform readiness now proves the native `native::import` simulation escape hatch is live before release sign-off.
- Strict release evidence and capability reporting now require archived native FFI simulation smoke artifacts:
  - `artifacts/sim/native_smoke_run.json`
  - `artifacts/sim/native_smoke_profile.json`
- Synced docs/spec/release metadata and version surfaces to `v2.6.6`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_6_6_rc_pipeline.ps1`
  - `scripts/v2_6_6_rc_pipeline.sh`

### Breaking changes
- None.

## v2.6.5

### Highlights
- Advanced the release line to `v2.6.5` with additive compatibility and no syntax changes.
- Hardened simulation release evidence with semantic verification:
  - new verifier:
    - `scripts/verify_sim_evidence.py`
  - new readiness artifact:
    - `artifacts/readiness/sim_evidence_verify.json`

### Fixes
- Full-platform readiness now verifies that simulation smoke reports, replay output, and profile output are mutually consistent before release sign-off.
- Strict release evidence and capability reporting now require the simulation evidence verification artifact in addition to the archived simulation smoke payloads.
- Synced docs/spec/release metadata and version surfaces to `v2.6.5`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_6_5_rc_pipeline.ps1`
  - `scripts/v2_6_5_rc_pipeline.sh`

### Breaking changes
- None.

## v2.6.4

### Highlights
- Advanced the release line to `v2.6.4` with additive compatibility and no syntax changes.
- Integrated simulation runtime smoke into the full-platform readiness and release-evidence contract:
  - new readiness smoke script:
    - `scripts/readiness_sim_smoke.py`
  - new readiness check:
    - `simulation-smoke`
  - archived strict evidence now includes:
    - `artifacts/readiness/sim_smoke.json`
    - `artifacts/sim/smoke_run.json`
    - `artifacts/sim/smoke_profile.json`
    - `artifacts/sim/smoke_replay.json`

### Fixes
- Added blocker-matrix enforcement for simulation smoke artifacts in the full-platform release line.
- Extended strict capability reporting to require archived simulation smoke evidence.
- Synced docs/spec/release metadata and version surfaces to `v2.6.4`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_6_4_rc_pipeline.ps1`
  - `scripts/v2_6_4_rc_pipeline.sh`

### Breaking changes
- None.

## v2.6.3

### Highlights
- Advanced the release line to `v2.6.3` with additive compatibility and no syntax changes.
- Added simulation CLI surfaces:
  - `enkai sim run`
  - `enkai sim profile`
  - `enkai sim replay`
- Added deterministic JSON reporting for simulation runs and replay flows.

### Fixes
- Added VM-profile output support for simulation CLI runs through `ENKAI_BENCH_PROFILE_OUT` / `ENKAI_BENCH_PROFILE_CASE`.
- Added simulation CLI parsing, run, replay, and profile tests.
- Synced docs/spec/release metadata and version surfaces to `v2.6.3`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_6_3_rc_pipeline.ps1`
  - `scripts/v2_6_3_rc_pipeline.sh`

## v2.6.2

### Highlights
- Advanced the release line to `v2.6.2` with additive compatibility and no syntax changes.
- Added core simulation primitives in std/runtime:
  - `std::sparse`
  - `std::event`
  - `std::pool`
- Added deterministic simulation scheduling/runtime support:
  - `std::sim`
  - world creation with seed + bounded event capacity
  - event scheduling / stepping / bounded run
  - snapshot / restore / replay helpers
  - entity set/get/remove/id surfaces

### Fixes
- Implemented deterministic sparse storage, non-zero iteration, dot, and matvec helpers for AGI/simulation workloads.
- Implemented deterministic timestamp ordering with insertion-order tie breaking for event queues.
- Implemented fixed-capacity and growable reusable value pools with explicit stats and no hidden capacity behavior.
- Added stable simulation runtime error codes for:
  - time-order violations
  - event-capacity overflow
  - bounded-run starvation
  - corrupted replay / snapshot restoration
  - unsupported snapshot payload types
- Added runtime/compiler/typecheck/test coverage for `SparseVector`, `SparseMatrix`, `EventQueue`, `Pool`, and `SimWorld`.
- Synced docs/spec/release metadata and version surfaces to `v2.6.2`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_6_2_rc_pipeline.ps1`
  - `scripts/v2_6_2_rc_pipeline.sh`

## v2.6.0

### Highlights
- Advanced the release line to `v2.6.0` with additive compatibility and no syntax changes.
- Hardened `native::import` for AGI/simulation-oriented native acceleration:
  - new FFI `Handle` type
  - optional `Handle?` support
  - automatic opaque handle destruction through `enkai_handle_free`
  - stable ABI-policy support through `enkai_abi_version` + `enkai_symbol_table`
- Extended VM benchmark profiling for FFI-heavy workloads:
  - marshal/copy operation count
  - native handle object count
  - existing native-call timing/byte counters preserved

### Fixes
- Added deterministic FFI error-code mapping for:
  - library load failures
  - symbol resolution failures
  - ABI/symbol-table failures
  - invalid argument and return contracts
  - missing `enkai_free` / `enkai_handle_free`
- Added runtime/compiler/typecheck coverage for handle-based FFI contracts.
- Added native test handle exports and ABI metadata to `enkai_native`.
- Synced docs/spec/release metadata and version surfaces to `v2.6.0`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_6_0_rc_pipeline.ps1`
  - `scripts/v2_6_0_rc_pipeline.sh`

### Breaking changes
- None.

## v2.5.9

### Highlights
- Advanced minor line to `v2.5.9` with additive compatibility (no contract removals).
- Closed the full-platform release evidence loop:
  - strict release evidence now archives `readiness/full_platform_blockers.json`
  - strict capability reporting now validates the archived blocker-verification verdict
- Hardened release and RC pipeline ordering:
  - `scripts/release_pipeline.ps1`
  - `scripts/release_pipeline.sh`
  - `scripts/rc_pipeline.ps1`
  - `scripts/rc_pipeline.sh`
  - blocker verification now runs before archival for bootstrap evidence and again after archive generation for final sign-off

### Fixes
- Removed the circular dependency where blocker verification implicitly depended on the capability report artifact.
- Documented archived blocker-verification requirements across spec, readiness matrix, capability report docs, and release checklist.
- Synced docs/spec/release metadata and version surfaces to `v2.5.9`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_5_9_rc_pipeline.ps1`
  - `scripts/v2_5_9_rc_pipeline.sh`

### Breaking changes
- None.

## v2.5.6

### Highlights
- Advanced minor line to `v2.5.6` with additive compatibility (no contract removals).
- Hardened production deploy validation:
  - additive machine-readable output:
    - `enkai deploy validate ... --json --output <file>`
  - additive migration contract validation:
    - zero-padded migration sequencing checks
    - required SQL fragment checks for generated conversation-state migrations
  - additive deploy-asset contract validation:
    - required env-key coverage in Docker/systemd assets
    - frontend package/SDK fragment validation for fullstack scaffolds

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.5.6`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_5_6_rc_pipeline.ps1`
  - `scripts/v2_5_6_rc_pipeline.sh`

### Breaking changes
- None.

## v2.5.5

### Highlights
- Advanced minor line to `v2.5.5` with additive compatibility (no contract removals).
- Added deterministic readiness filtering:
  - `enkai readiness check ... --skip-check <id>`
  - unknown skipped check ids now fail fast with a machine-readable report mismatch.
- Hardened release pipeline sign-off behavior:
  - `scripts/release_pipeline.ps1`
  - `scripts/release_pipeline.sh`
  - both now treat `full_platform` readiness as the canonical non-hardware release gate.
- Added release pipeline disk-space preflight via `ENKAI_RELEASE_MIN_FREE_GB`.

### Fixes
- Removed redundant standalone self-host readiness execution from release pipelines by skipping
  `selfhost-mainline` and `selfhost-stage0-fallback` in readiness when `litec release-ci` runs separately.
- Aligned archived evidence policy so strict release evidence requires `readiness/full_platform.json`
  while keeping `readiness/production.json` as an optional compatibility artifact.
- Synced docs/spec/release metadata and version surfaces to `v2.5.5`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_5_5_rc_pipeline.ps1`
  - `scripts/v2_5_5_rc_pipeline.sh`

### Breaking changes
- None.

## v2.5.4

### Highlights
- Advanced minor line to `v2.5.4` with additive compatibility (no contract removals).
- Promoted bootstrap mainline compile path for `litec check|compile|stage|run`.
- Added automatic Stage0 emergency fallback when mainline bootstrap build/fixed-point checks fail.
- Added deterministic fallback triage output:
  - `litec_mainline_fallback_report.json`
  - default path `artifacts/selfhost/` (override with `ENKAI_LITEC_TRIAGE_DIR`).
- Tightened bootstrap replacement gate:
  - `enkai litec replace-check` now enforces Stage1/Stage2 runtime parity in addition to bytecode equivalence.

### Fixes
- Added bootstrap regression tests for:
  - automatic fallback + triage emission.
  - Stage1/Stage2 runtime parity report fields.
- Synced docs/spec/release metadata and version surfaces to `v2.5.4`.
- Added current-line RC wrapper scripts:
  - `scripts/v2_5_4_rc_pipeline.ps1`
  - `scripts/v2_5_4_rc_pipeline.sh`

### Breaking changes
- None.

## v2.5.3

### Highlights
- Advanced minor line to `v2.5.3` with additive compatibility (no contract removals).
- Added signed remote model registry sync and lifecycle controls:
  - `enkai model push <registry_dir> <name> <version> --registry <remote_registry_dir> [--sign]`
  - `enkai model pull <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
  - `enkai model promote-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
  - `enkai model retire-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
  - `enkai model rollback-remote <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
- Added immutable remote artifact metadata and optional signature contract:
  - `remote.manifest.json` with deterministic `artifact_digest`
  - `remote.manifest.sig` verified with `ENKAI_MODEL_SIGNING_KEY`
- Added append-only lifecycle audit stream:
  - `audit.log.jsonl` (local and remote sync events, including signature/fallback outcomes)
- Added current-line RC wrapper scripts:
  - `scripts/v2_5_3_rc_pipeline.ps1`
  - `scripts/v2_5_3_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.5.3`.
- Added remote-registry regression tests for remote online and degraded fallback paths.

### Breaking changes
- None.

## v2.5.2

### Highlights
- Advanced minor line to `v2.5.2` with additive compatibility (no contract removals).
- Added distributed orchestration contract fields to train/pretrain/eval config:
  - `dist.topology`, `dist.rendezvous`, `dist.retry_budget`, `dist.device_map`
- Added cluster orchestration CLI surface:
  - `enkai cluster validate <config.enk> [--json]`
  - `enkai cluster plan <config.enk> [--json]`
  - `enkai cluster run <config.enk> [--dry-run] [--json]`
- Added runtime distributed hardening:
  - rank-device mapping enforcement via dist config
  - retry-budgeted native distributed init path (`enkai_dist_config`) with stable machine-parseable error codes (`E_DIST_*`)
  - deterministic fallback to legacy distributed init symbol when config symbol is unavailable
- Added current-line RC wrapper scripts:
  - `scripts/v2_5_2_rc_pipeline.ps1`
  - `scripts/v2_5_2_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.5.2`.
- Added distributed/cluster regression tests for orchestration parsing and planner validation.

### Breaking changes
- None.

## v2.5.1

### Highlights
- Advanced minor line to `v2.5.1` with additive compatibility (no contract removals).
- Added single-node runtime reliability hardening for train/pretrain resume paths:
  - strict resume-time run-state validation across lineage/runtime identity fields
    (`config_hash`, `code_hash`, `dataset_hash`, `seed`, `backend`, `dtype`, `device`, rank/world)
  - additive run validation artifact:
    - `checkpoint_dir/run_validation.json`
  - deterministic resume parity regression coverage for interrupted vs uninterrupted runs
  - strict/lenient resume mismatch behavior coverage for dataset-drift scenarios
- Added current-line RC wrapper scripts:
  - `scripts/v2_5_1_rc_pipeline.ps1`
  - `scripts/v2_5_1_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.5.1`.

### Breaking changes
- None.

## v2.5.0

### Highlights
- Advanced minor line to `v2.5.0` with additive compatibility (no contract removals).
- Added full-platform readiness profile support:
  - `enkai readiness check --profile full_platform --json --output <file>`
- Added v2.5 full-platform readiness contracts:
  - `enkai/contracts/readiness_full_platform_v2_5_0.json`
  - `enkai/contracts/full_platform_release_blockers_v2_5_0.json`
- Updated CI readiness artifacts to publish:
  - `artifacts/readiness/production.json`
  - `artifacts/readiness/full_platform.json`
- Added current-line RC wrapper scripts:
  - `scripts/v2_5_0_rc_pipeline.ps1`
  - `scripts/v2_5_0_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.5.0`.
- Improved readiness command binary resolution with explicit `ENKAI_READINESS_ENKAI_BIN` override support and stable default behavior.

### Breaking changes
- None.

## v2.4.0

### Highlights
- Advanced minor line to `v2.4.0` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_4_0_rc_pipeline.ps1`
  - `scripts/v2_4_0_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.4.0`.

### Breaking changes
- None.

## v2.3.9

### Highlights
- Advanced patch line to `v2.3.9` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_9_rc_pipeline.ps1`
  - `scripts/v2_3_9_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.3.9`.

### Breaking changes
- None.

## v2.3.8

### Highlights
- Advanced patch line to `v2.3.8` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_8_rc_pipeline.ps1`
  - `scripts/v2_3_8_rc_pipeline.sh`

### Fixes
- Hardened release evidence + capability report integrity for version-scoped dist artifacts:
  - `scripts/collect_release_evidence.py` now archives only `enkai-2.3.8-*`/`sbom-2.3.8-*` artifacts (plus benchmark evidence).
  - `scripts/generate_capability_report.py` now validates current-version archive/checksum/SBOM paths.
- Synced docs/spec/release metadata and version surfaces to `v2.3.8`.

### Breaking changes
- None.

## v2.3.7

### Highlights
- Advanced patch line to `v2.3.7` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_7_rc_pipeline.ps1`
  - `scripts/v2_3_7_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.3.7`.
- Hardened release evidence packaging/reporting so capability checks are version-scoped:
  - `scripts/collect_release_evidence.py` now archives only current-version package/SBOM artifacts (plus benchmark evidence).
  - `scripts/generate_capability_report.py` now validates current-version archive/checksum/SBOM paths.

### Breaking changes
- None.

## v2.3.6

### Highlights
- Advanced patch line to `v2.3.6` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_6_rc_pipeline.ps1`
  - `scripts/v2_3_6_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.3.6`.

### Breaking changes
- None.

## v2.3.5

### Highlights
- Advanced patch line to `v2.3.5` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_5_rc_pipeline.ps1`
  - `scripts/v2_3_5_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.3.5`.

### Breaking changes
- None.

## v2.3.4

### Highlights
- Advanced patch line to `v2.3.4` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_4_rc_pipeline.ps1`
  - `scripts/v2_3_4_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.3.4`.

### Breaking changes
- None.

## v2.3.3

### Highlights
- Advanced patch line to `v2.3.3` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_3_rc_pipeline.ps1`
  - `scripts/v2_3_3_rc_pipeline.sh`

### Fixes
- Synced docs/spec/release metadata and version surfaces to `v2.3.3`.

### Breaking changes
- None.

## v2.3.2

### Highlights
- Advanced patch line to `v2.3.2` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_2_rc_pipeline.ps1`
  - `scripts/v2_3_2_rc_pipeline.sh`

### Fixes
- Hardened benchmark gate reliability by stabilizing VM compute benchmark workload envelope for class-target enforcement.
- Synced docs/spec/release metadata and version surfaces to `v2.3.2`.

### Breaking changes
- None.

## v2.3.1

### Highlights
- Advanced patch line to `v2.3.1` with additive compatibility (no contract removals).
- Added current-line RC wrapper scripts:
  - `scripts/v2_3_1_rc_pipeline.ps1`
  - `scripts/v2_3_1_rc_pipeline.sh`

### Fixes
- Stabilized VM compute benchmark case for class-gate consistency:
  - `bench/enkai/kernel_numeric.enk`
  - `bench/python/kernel_numeric.py`
  - `bench/suites/official_v2_3_0_vm_compute.json`
- Synced docs/release metadata to `v2.3.1` baseline.

### Breaking changes
- None.

## v2.3.0

### Highlights
- Added consolidated production readiness command:
  - `enkai readiness check --profile production --json --output <file>`
  - readiness manifest: `enkai/contracts/readiness_production_v2_3_0.json`
- Added bootstrap consolidated release gate:
  - `enkai litec release-ci <corpus_dir> [--triage-dir <dir>]`
  - triage summary artifact: `litec_release_ci_report.json`
- Added deployment validator command:
  - `enkai deploy validate <project_dir> --profile <backend|fullstack> --strict`
- Added class-based official benchmark target suite for v2.3 readiness:
  - `bench/suites/official_v2_3_0_matrix.json`
  - `bench/suites/official_v2_3_0_vm_compute.json`
  - `bench/suites/official_v2_3_0_native_bridge.json`
  - `bench/suites/official_v2_3_0_cli_workflows.json`
  - `bench/suites/official_v2_3_0_ai_data_workflows.json`
  - `bench/suites/official_v2_3_0_targets.json`
- Added workload-equivalence benchmark fairness contract:
  - `bench/contracts/workload_equivalence_v1.json`

### Fixes
- Release/CI workflows now emit readiness-report artifacts.
- Release pipelines now run `litec release-ci` and produce readiness JSON (`artifacts/readiness/production.json`).
- Release pipeline benchmark runner now pins benchmark Python to the machine profile for deterministic fairness checks.
- Capability evidence tooling now archives/validates readiness + `litec_release_ci_report.json` in strict mode.
- Backend/frontend scaffold contracts now include readiness endpoint `GET /api/<version>/ready` and `service_not_ready` error code.
- Benchmark workloads were aligned for deterministic class-gate enforcement:
  - HTTP serving benchmark steady-state profile
  - numeric VM compute kernel fairness path
  - JSON/hash/db bridge workload equivalence metadata

### Breaking changes
- None.

## v2.2.0

### Highlights
- Advanced the release line to `v2.2.0` with additive contract continuity.
- Added official bounded benchmark suite for this line:
  - `bench/suites/official_v2_2_0.json`
  - machine profiles pinned to `official_v2_2_0`
- Added current-line RC wrappers:
  - `scripts/v2_2_0_rc_pipeline.ps1`
  - `scripts/v2_2_0_rc_pipeline.sh`
- Added explicit serving lifecycle controls for model registry operations:
  - `enkai model load <registry_dir> <name> <version>`
  - `enkai model unload <registry_dir> <name> <version>`
  - `enkai model loaded <registry_dir> [name] [--json]`
- Added multi-model serving mode with per-request selector enforcement:
  - `enkai serve --multi-model --registry <dir>`
  - deterministic selector/load errors: `missing_model_selector`, `model_not_loaded`

### Fixes
- CI `release-pipeline` lane now runs full package gates (no skip-package mode) and publishes strict evidence artifacts.
- CI now runs strict evidence/report generation from release outputs:
  - `scripts/collect_release_evidence.py --strict`
  - `scripts/generate_capability_report.py --strict`
- Updated benchmark/release/spec/docs references to the `v2.2.0` contract line.
- Runtime HTTP metadata + rate-limiter model keying now use request-selected model identity in multi-model mode.

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
