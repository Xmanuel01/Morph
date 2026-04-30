Enkai Roadmap

Note:
- Historical milestones below capture the path that led to current releases.
- Current release line is v3.5.0.
- `v3.3.0` closed the strict-selfhost shipped-surface objective set.
- Next major program line remains `v3.1.0 -> v4.0.0` zero-Rust strict self-hosting, with post-closure scope now moving past the shipped-surface completion proof.
- v2.6.x remains additive/integration work (no contract-breaking removals).
- Use `docs/Enkai.spec` as the source of truth for current language behavior.

v3.5.0 (done)
- Release-line start baseline (done):
  - opened after the closed `v3.4.0` post-closure baseline release line
  - established `v3.5.0` as the next explicit post-closure program line
  - preserved `v3.4.0` as the completed source-of-record baseline for the first post-closure zero-Rust follow-on program
  - emits:
    - `enkai/contracts/v3_5_0_release_line_start_baseline.json`
    - `artifacts/readiness/v3_5_0_release_line_start_baseline.json`
- Objective-set freeze baseline (done):
  - turns `v3.5.0` from an empty start-line into an explicit bounded objective set
  - records the current `v3.5.0` objectives as:
    - accelerated native/tensor global replacement program boundary
    - compatibility-only storage/data global replacement program boundary
    - broader non-shipped compatibility-path closure program boundary
  - emits:
    - `enkai/contracts/v3_5_0_objective_set_freeze.json`
    - `artifacts/readiness/v3_5_0_objective_set_freeze.json`
- Evidence continuity baseline (done):
  - proves the completed shipped strict-selfhost line and the completed `v3.4.0` post-closure baselines remain green while `v3.5.0` records the next program boundary
  - anchors the line to:
    - `artifacts/readiness/strict_selfhost.json`
    - `artifacts/readiness/strict_selfhost_blockers.json`
    - `artifacts/readiness/strict_selfhost_dependency_inventory.json`
    - `artifacts/readiness/v3_4_0_closure.json`
  - emits:
    - `enkai/contracts/v3_5_0_evidence_continuity_baseline.json`
    - `artifacts/readiness/v3_5_0_evidence_continuity_baseline.json`
- Version closure state:
  - all defined `v3.5.0` tranches are closed and machine-verifiable
  - `v3.5.0` is complete as a bounded post-closure objective-definition release line

v3.4.0 (done)
- Post-closure release-line normalization baseline (done):
  - establishes `v3.4.0` as the active development line after the `v3.3.0` strict-selfhost shipped-surface closure
  - makes archived release dashboard/reporting explicitly distinguish:
    - archived release line
    - active development line
    - strict-selfhost shipped-surface closure line
    - historical hardware sign-off program origin
  - updates the strict self-host contract text so shipped-surface closure is recorded as complete rather than still described as blocked
  - emits:
    - `enkai/contracts/v3_4_0_release_line_baseline.json`
    - `artifacts/readiness/v3_4_0_release_line_baseline.json`
  - exit state for this tranche:
    - the repository has one explicit machine-verifiable `v3.4.0` implementation scope closed instead of only a version-bump start line
- Post-closure zero-Rust next-step baseline (done):
  - establishes the first explicit broader-than-shipped-surface follow-on scope after the `v3.3.0` closure
  - records the next zero-Rust work as:
    - cross-host install proof expansion beyond the current Windows-host execution evidence
    - replacement of compatibility-only storage/data paths still tolerated outside the shipped strict-selfhost blockers
    - replacement of globally accelerated native/tensor backends beyond the current runtime-owned fallback boundary
    - broader zero-Rust closure of historical non-shipped compatibility paths before the eventual `v4.0.0` target
  - emits:
    - `enkai/contracts/v3_4_0_zero_rust_next_step_baseline.json`
    - `artifacts/readiness/v3_4_0_zero_rust_next_step_baseline.json`
  - exit state for this tranche:
    - post-closure zero-Rust work is no longer implicit; it is recorded as an explicit machine-verifiable `v3.4.0` scope with concrete next-step categories
- Cross-host install proof matrix baseline (done):
  - turns the first zero-Rust next-step category into an explicit implementation tranche
  - records the current install-proof matrix as:
    - Windows-host install/upgrade/uninstall lifecycle proof is executed and green
    - Linux-host install/upgrade/uninstall lifecycle proof is now executed and archived
  - anchors the matrix to:
    - `enkai/contracts/install_flow_v3_3_0.json`
    - `enkai/contracts/install_flow_v3_3_0_windows.json`
    - `enkai/contracts/install_flow_v3_3_0_linux.json`
    - `artifacts/install_bundle_smoke/install_flow_proof.json`
    - `artifacts/install_bundle_smoke/install_flow_proof_linux.json`
  - emits:
    - `enkai/contracts/v3_4_0_install_host_matrix_baseline.json`
    - `artifacts/readiness/v3_4_0_install_host_matrix_baseline.json`
  - exit state for this tranche:
    - cross-host install proof is no longer implied by one Windows artifact; the repo now records an explicit host-matrix baseline with executed Windows-host and Linux-host lifecycle evidence
- Compatibility-only storage/data path baseline (done):
  - turns the next zero-Rust post-closure category into an explicit implementation tranche
  - records the current compatibility-only storage/data state as:
    - the shipped strict-selfhost surface remains complete
    - `sqlite_binding` is the explicitly tracked compatibility-only storage/data path outside shipped strict-selfhost release blockers
    - the broader global replacement of SQLite-backed compatibility paths remains future roadmap work rather than an implicit gap
  - anchors the baseline to:
    - `artifacts/readiness/strict_selfhost_dependency_inventory.json`
    - `artifacts/readiness/strict_selfhost_data_registry_protocols_surface.json`
    - `artifacts/readiness/strict_selfhost.json`
    - `artifacts/readiness/strict_selfhost_blockers.json`
  - emits:
    - `enkai/contracts/v3_4_0_compatibility_storage_data_baseline.json`
    - `artifacts/readiness/v3_4_0_compatibility_storage_data_baseline.json`
  - exit state for this tranche:
    - compatibility-only storage/data work is no longer implicit; the repo now records the exact tolerated path and the proof boundary for future global replacement
- Accelerated native/tensor backend baseline (done):
  - turns the globally accelerated native/tensor follow-on category into an explicit implementation tranche
  - records the current accelerated backend boundary as:
    - the shipped strict-selfhost surface remains complete
    - runtime-owned fallback and proof boundaries are closed for the shipped line
    - broader global replacement of accelerated Rust/native tensor backends remains future roadmap work rather than an implicit blocker in `v3.4.0`
  - anchors the baseline to:
    - `artifacts/readiness/strict_selfhost_tensor_backend_surface.json`
    - `artifacts/readiness/strict_selfhost_native_std_and_accel_surface.json`
    - `artifacts/readiness/strict_selfhost_dependency_inventory.json`
    - `artifacts/readiness/strict_selfhost.json`
    - `artifacts/readiness/strict_selfhost_blockers.json`
  - emits:
    - `enkai/contracts/v3_4_0_accelerated_native_tensor_baseline.json`
    - `artifacts/readiness/v3_4_0_accelerated_native_tensor_baseline.json`
  - exit state for this tranche:
    - accelerated native/tensor follow-on work is no longer implicit; the repo now records the exact shipped proof boundary and the remaining broader replacement scope
- Historical non-shipped compatibility-path closure baseline (done):
  - turns the final broader post-closure category into an explicit implementation tranche
  - records the current non-shipped compatibility closure boundary as:
    - the shipped strict-selfhost surface remains complete
    - broader historical compatibility paths outside the shipped release gate are explicitly tracked as future roadmap work toward `v4.0.0`
    - those paths no longer exist as an implicit unresolved category inside `v3.4.0`
  - anchors the baseline to:
    - `artifacts/readiness/v3_4_0_zero_rust_next_step_baseline.json`
    - `artifacts/readiness/v3_4_0_install_host_matrix_baseline.json`
    - `artifacts/readiness/v3_4_0_compatibility_storage_data_baseline.json`
    - `artifacts/readiness/v3_4_0_accelerated_native_tensor_baseline.json`
  - emits:
    - `enkai/contracts/v3_4_0_non_shipped_compatibility_closure_baseline.json`
    - `artifacts/readiness/v3_4_0_non_shipped_compatibility_closure_baseline.json`
  - exit state for this tranche:
    - the broader non-shipped compatibility-path closure category is no longer implicit; it is recorded as an explicit roadmap boundary with machine-verifiable baseline evidence
  - version closure state:
    - all defined `v3.4.0` tranches are closed and machine-verifiable
    - `v3.4.0` is complete as a post-closure baseline release line

v3.1.0 (in progress)
- Strict self-host contract freeze:
  - adds `strict_selfhost` readiness and blocker profiles
  - freezes the zero-Rust dependency inventory in:
    - `enkai/contracts/strict_selfhost_dependency_board_v3_1_0.json`
  - emits:
    - `artifacts/readiness/strict_selfhost.json`
    - `artifacts/readiness/strict_selfhost_dependency_inventory.json`
    - `artifacts/readiness/strict_selfhost_blockers.json`
  - extends the release dashboard with:
    - `strict_selfhost_cpu_complete`
    - `strict_selfhost_gpu_pending`
    - `remaining_rust_dependencies`
  - documents the contract in:
    - `docs/50_strict_selfhost_contract.md`

v3.1.1 (done)
- Full-language frontend migration tranche:
  - adds `enkai litec frontend-audit <corpus_dir>`
  - records Rust/frontend acceptance and stage0/stage2 parity over broader corpora
  - keeps both the frozen declaration frontier and the shipped `examples/`
    corpus green under strict self-host readiness
  - adds strict self-host bootstrap-source and negative semantic corpus audits:
    - `enkai/contracts/selfhost_bootstrap_v3_1_1.json`
    - `enkai/contracts/selfhost_negative_v3_1_1.json`
  - adds curated audited-surface bundled proof:
    - `enkai/contracts/selfhost_audited_surface_v3_1_1.json`
    - `artifacts/readiness/selfhost_audited_surface_verify.json`
  - expands the bootstrap subset to accept:
    - `native::import`
    - `tool`
    - `prompt`
    - `model`
    - `agent`
  - adds package-aware bootstrap compiler intrinsics:
    - `compiler.parse_subset_file`
    - `compiler.check_subset_file`
    - `compiler.emit_subset_file`
  - emits:
    - `litec_frontend_audit_report.json`
    - `litec_negative_audit_report.json`
  - documents the current frontier in:
    - `docs/51_full_frontend_frontier.md`
  - exit state for this tranche:
    - the audited v3.1.1 language surface is self-host source-of-truth with Rust retained only as verifier/fallback for that audited surface

v2.9.1 (done)
- Quality recovery foundation:
  - adds proof-grade CPU validation commands under `enkai validate`
  - archives deterministic validation artifacts under `artifacts/validation/`
  - promotes correctness, determinism, pool safety, and Adam-0 CPU validation into release evidence
  - tightens release-evidence/capability reporting so claims must be backed by archived proof artifacts
  - keeps GPU sign-off as operator evidence, but removes ambiguity around CPU-complete vs GPU-pending state

v2.9.2 (done)
- Quality recovery line:
  - strengthens `std::sparse`, `std::event`, and `std::pool` proof quality
  - requires native-backed and VM-fallback equivalence in validation artifacts
  - verifies deterministic event ordering and pool plateau semantics
  - adds seeded native-vs-VM property tests for sparse/event/pool

v2.9.4 (done)
- Quality recovery line:
  - makes deterministic replay and coroutine behavior machine-auditable
  - adds simulation audit hashes for seed/config/log/snapshot/replay
  - promotes coroutine counters and replay-hash equality into release-blocking validation

v2.9.5 (done)
- Quality recovery line:
  - hardens FFI and runtime safety into release-blocking validation
  - adds invalid-handle, double-free, fault-injection, and corrupted-replay regression coverage
  - archives runtime safety evidence and verification artifacts in strict release evidence

v2.9.6 (done)
- Quality recovery line:
  - promotes release claims from artifact presence to proof-group summaries
  - adds a release dashboard that summarizes CPU completeness, GPU pending state, and operator steps
  - wires the release dashboard into release and RC pipelines for strict evidence publication

v3.0.0 (CPU-complete / GPU sign-off pending)
- Stability cut for the first AGI + LLM platform release:
  - non-hardware full-platform readiness is complete on the current line
  - publication assets staged:
    - `docs/42_agi_runbook.md`
    - `docs/43_llm_runbook.md`
    - `docs/44_native_extension_abi_policy.md`
    - `docs/45_deployment_rollback_runbook.md`
    - `docs/46_benchmark_profiling_guide.md`
    - `docs/47_gpu_operator_preflight.md`
  - operator-ready GPU preflight added:
    - `scripts/gpu_preflight.py`
    - `scripts/gpu_preflight.ps1`
    - `scripts/gpu_preflight.sh`
  - current hard blocker for the final cut:
    - operator-run GPU evidence package required by the RC pipeline and blocker verifier
  - final QA findings documented in:
    - `docs/49_v3_0_0_quality_assurance.md`
  - RC wrapper prepared:
    - `scripts/v3_0_0_rc_pipeline.ps1`
    - `scripts/v3_0_0_rc_pipeline.sh`

v2.9.0 (done)
- Scale, multi-node, and reliability:
  - extended `enkai cluster` with additive multi-node host/planning fields:
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
  - `enkai cluster run` now supervises bounded simulation workloads with windowed `sim run`/`sim replay`, snapshot emission, and bounded retry/recovery
  - added release-gated cluster scale smoke + verification:
    - `scripts/readiness_cluster_scale_smoke.py`
    - `scripts/verify_cluster_scale_evidence.py`
  - added release-gated degraded registry fallback smoke + verification:
    - `scripts/readiness_registry_degraded_smoke.py`
    - `scripts/verify_registry_degraded_evidence.py`
  - full-platform readiness, blocker verification, strict evidence archiving, and capability reporting now require archived cluster-scale and degraded-registry artifacts
  - documented the scale/reliability contract in:
    - `docs/41_scale_reliability.md`

v2.8.0 (done)
- LLM + AGI data/training/registry convergence:
  - added simulation lineage and world-snapshot manifests to `enkai sim`:
    - `--lineage-output <file>`
    - `--snapshot-manifest-output <file>`
  - extended the model registry lifecycle to support additive artifact kinds:
    - `checkpoint`
    - `simulation`
    - `environment`
    - `native-extension`
  - added signed remote verification command:
    - `enkai model verify-signature <registry_dir> <name> <version> --registry <remote_registry_dir>`
  - added full-platform signed registry convergence smoke + verification:
    - `scripts/readiness_registry_convergence.py`
    - `scripts/verify_registry_convergence.py`
  - full-platform readiness now archives and verifies:
    - `artifacts/readiness/model_registry_convergence.json`
    - `artifacts/readiness/model_registry_convergence_verify.json`
    - `artifacts/registry/sim_lineage.json`
    - `artifacts/registry/sim_snapshot.manifest.json`
    - `artifacts/registry/local/registry.json`
    - `artifacts/registry/remote/registry.json`
    - `artifacts/registry/cache/registry.json`
    - `artifacts/registry/remote/adam0-sim/v2.8.0/remote.manifest.json`
    - `artifacts/registry/remote/adam0-sim/v2.8.0/remote.manifest.sig`
  - documented the convergence contract in:
    - `docs/40_registry_convergence.md`

v2.8.0 (done)
- Adam-0 reference stack completion:
  - added bounded Adam-0 reference suite:
    - `examples/adam0_reference.enk`
    - `bench/suites/adam0_reference_v2_9_4.json`
  - added suite evidence generation and semantic verification:
    - `scripts/readiness_adam0_reference_suite.py`
    - `scripts/verify_adam0_reference_suite.py`
  - full-platform readiness now archives and verifies the 100 / 1000 / 10000 agent Adam-0 suite artifacts
  - documented bounded hardware assumptions and suite contract in:
    - `docs/39_adam0_reference_stack.md`

v2.7.0 (done)
- Simulation coroutine, SNN runtime, and agent environment completion:
  - added task-backed coroutine/generator-facing APIs under `std::sim`:
    - `sim.coroutine`
    - `sim.coroutine_with`
    - `sim.coroutine_args`
    - `sim.emit`
    - `sim.next`
    - `sim.join`
    - `sim.done`
  - added `SimCoroutine` runtime/typechecker support without syntax changes
  - added the deterministic 100-agent Adam-0 reference workload:
    - `examples/adam0_100.enk`
  - added the deterministic SNN + agent environment kernel reference workload:
    - `examples/snn_agent_kernel.enk`
  - added additive runtime/std modules:
    - `std::spatial`
    - `std::snn`
    - `std::agent`
  - added release-gated SNN/agent kernel evidence generation and semantic verification:
    - `scripts/readiness_snn_agent_kernel_smoke.py`
    - `scripts/verify_snn_agent_kernel_evidence.py`
  - added release-gated Adam-0 evidence generation and semantic verification:
    - `scripts/readiness_adam0_smoke.py`
    - `scripts/verify_adam0_evidence.py`
  - full-platform readiness now archives and verifies both Adam-0 and SNN/agent kernel smoke/profile artifacts

v2.6.8 (done)
- Native-backed simulation primitive completion:
  - added internal `enkai_native` acceleration bindings for:
    - sparse vector handles
    - sparse matrix handles
    - event queue handles
    - pool handles
  - kept the public Enkai interfaces unchanged while enabling native-backed execution under:
    - `std::sparse`
    - `std::event`
    - `std::pool`
  - preserved deterministic fallback behavior when `enkai_native` is unavailable or `ENKAI_SIM_ACCEL=0`
  - tightened stdlib simulation smoke verification so archived VM profiles now prove the native acceleration path is live

v2.6.6 (done)
- Simulation native FFI evidence hardening:
  - added native smoke workflow via:
    - `scripts/readiness_sim_native_smoke.py`
  - added native smoke evidence verifier via:
    - `scripts/verify_sim_native_evidence.py`
  - full-platform readiness now emits:
    - `artifacts/readiness/sim_native_smoke.json`
    - `artifacts/readiness/sim_native_evidence_verify.json`
    - `artifacts/sim/native_smoke_run.json`
    - `artifacts/sim/native_smoke_profile.json`
  - strict release evidence and capability reporting now require the native FFI verification artifact

v2.6.5 (done)
- Simulation evidence verification hardening:
  - added semantic verification for archived simulation smoke artifacts via:
    - `scripts/verify_sim_evidence.py`
  - full-platform readiness now emits:
    - `artifacts/readiness/sim_evidence_verify.json`
  - strict release evidence and capability reporting now require the verification artifact

v2.6.4 (done)
- Simulation production-readiness integration:
  - full-platform readiness now executes a simulation smoke gate via:
    - `scripts/readiness_sim_smoke.py`
  - archived release evidence now includes:
    - `artifacts/readiness/sim_smoke.json`
    - `artifacts/sim/smoke_run.json`
    - `artifacts/sim/smoke_profile.json`
    - `artifacts/sim/smoke_replay.json`
  - strict capability reporting now requires the archived simulation evidence set

v2.6.3 (done)
- Simulation CLI hardening:
  - `enkai sim run`
  - `enkai sim profile`
  - `enkai sim replay`
  - deterministic JSON run reports for simulation workloads
  - VM profile artifact generation for simulation runs

v2.6.2 (done)
- Deterministic simulation scheduler/runtime helpers:
  - `std::sim`
    - world creation with seed + capacity
    - event scheduling / stepping / bounded run
    - snapshot / restore / replay for simulation state
    - entity set/get/remove/id surfaces
    - stable simulation error codes for overflow, starvation, and corrupted replay state

v2.6.1 (done)
- Core simulation primitives in std/runtime:
  - `std::sparse`
    - sparse vector/matrix storage
    - deterministic non-zero iteration
    - sparse dot/matvec helpers
  - `std::event`
    - deterministic timestamp-ordered event queue
    - tie-breaking by insertion order
  - `std::pool`
    - fixed-capacity and growable reusable value pools
    - explicit capacity/available/stats surfaces

v2.6.0 (done)
- FFI contract hardening for AGI/simulation workloads:
  - additive `Handle` / `Handle?` support in `native::import`
  - automatic opaque handle destruction via `enkai_handle_free`
  - optional native ABI policy support through:
    - `enkai_abi_version`
    - `enkai_symbol_table`
  - deterministic FFI error taxonomy for load/symbol/ABI/free failures
  - VM benchmark profile expansion for marshal/copy count and native handle counts

v2.5.9 (done)
- Release evidence closure for the full-platform line:
  - strict archived evidence now requires both:
    - `artifacts/readiness/full_platform.json`
    - `artifacts/readiness/full_platform_blockers.json`
  - release/RC pipelines bootstrap blocker verification before evidence archival and refresh the archive after final blocker verification
  - strict capability reporting now requires the archived blocker-verification artifact to be present and passing
  - reduced/package-skipped runs still archive the blocker report generated with `--skip-release-evidence`

v2.5.7 (done)
- Full-platform readiness profile expansion:
  - generated backend scaffold deploy-validation smoke
  - generated fullstack scaffold deploy-validation smoke
  - wrapper artifacts:
    - `artifacts/readiness/deploy_backend_smoke.json`
    - `artifacts/readiness/deploy_fullstack_smoke.json`
  - validation reports:
    - `artifacts/readiness/deploy_backend.json`
    - `artifacts/readiness/deploy_fullstack.json`

v2.5.6 (done)
- Deploy-validation hardening for production rollout workflows:
  - additive machine-readable output:
    - `enkai deploy validate ... --json --output <file>`
  - additive contract enforcement for:
    - migration sequencing/content under `migrations/`
    - Docker/systemd deploy assets against required env keys
    - frontend package/SDK fragments for fullstack scaffold validation

v2.5.5 (done)
- Readiness/release hardening for the v2.5 line:
  - additive readiness filter:
    - `enkai readiness check ... --skip-check <id>`
  - unknown skipped check ids fail early and skipped checks are recorded in the readiness JSON report.
- Release pipeline now treats full-platform readiness as the canonical non-hardware gate:
  - `scripts/release_pipeline.ps1`
  - `scripts/release_pipeline.sh`
  - standalone self-host readiness lanes are skipped there because `enkai litec release-ci` runs separately.
- Release pipeline now fails early on low-disk hosts via:
  - `ENKAI_RELEASE_MIN_FREE_GB`
- Strict evidence archive policy now requires:
  - `artifacts/readiness/full_platform.json`
  - `artifacts/readiness/production.json` remains optional compatibility evidence.

v2.5.4 (done)
- Bootstrap mainline promotion and emergency fallback automation:
  - `enkai litec check|compile|stage|run` now prefer Enkai-built mainline compiler path.
  - automatic Stage0 emergency fallback is triggered when mainline bootstrap build/fixed-point checks fail.
  - deterministic fallback triage bundle emitted:
    - `litec_mainline_fallback_report.json`
    - default path `artifacts/selfhost/` (override via `ENKAI_LITEC_TRIAGE_DIR`).
- Tightened release corpus equivalence gate:
  - `enkai litec replace-check` now enforces Stage1/Stage2 bytecode equivalence and Stage1/Stage2 runtime-output parity.
  - report entries include `stage1_result`, `stage2_result`, and `stage1_stage2_runtime_equivalent`.

v2.5.3 (done)
- Signed remote model registry sync and lifecycle operations:
  - additive model command surface:
    - `enkai model push <registry_dir> <name> <version> --registry <remote_registry_dir> [--sign]`
    - `enkai model pull <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
    - `enkai model promote-remote|retire-remote|rollback-remote ...`
  - immutable remote artifact metadata:
    - `remote.manifest.json` with deterministic `artifact_digest`
    - optional `remote.manifest.sig` verified with `ENKAI_MODEL_SIGNING_KEY`
  - append-only model lifecycle audit stream:
    - `audit.log.jsonl` for local lifecycle + remote sync outcomes
  - regression coverage for remote-online and remote-degraded fallback paths.

v2.5.2 (done)
- LLM distributed orchestration and failure-mode hardening:
  - additive train config distributed orchestration fields:
    - `dist.topology`, `dist.rendezvous`, `dist.retry_budget`, `dist.device_map`
  - runtime distributed configuration now maps rank->device explicitly and supports
    retry-budgeted init with stable machine-parseable error codes (`E_DIST_*`)
  - added cluster command surface:
    - `enkai cluster validate <config.enk> [--json]`
    - `enkai cluster plan <config.enk> [--json]`
    - `enkai cluster run <config.enk> [--dry-run] [--json]`
  - added unit/regression coverage for distributed orchestration parsing and cluster planner behavior.

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






