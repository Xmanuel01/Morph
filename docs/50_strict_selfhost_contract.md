# 50. Strict Self-Host Contract

This document freezes the `v3.1.0 -> v4.0.0` zero-Rust self-hosting target.

## Goal

The strict self-host line means the shipped Enkai platform must not require:

- Rust binaries
- Rust crates in the shipped toolchain/runtime path
- Rust-owned native modules as required production dependencies

This contract is stronger than the current `v3.0.0` release state. `v3.0.0`
remains CPU-complete and GPU-pending, but it is still a Rust workspace and is
not yet a zero-Rust shipped platform.

## Source of Truth

Machine-readable contract inputs live in:

- `enkai/contracts/readiness_strict_selfhost_v3_1_0.json`
- `enkai/contracts/strict_selfhost_release_blockers_v3_1_0.json`
- `enkai/contracts/strict_selfhost_dependency_board_v3_1_0.json`
- `enkai/contracts/selfhost_frontier_v3_1_1.json`
- `enkai/contracts/selfhost_examples_v3_1_1.json`
- `enkai/contracts/selfhost_bootstrap_v3_1_1.json`
- `enkai/contracts/selfhost_negative_v3_1_1.json`
- `enkai/contracts/selfhost_audited_surface_v3_1_1.json`

Generated inventory output lives in:

- `artifacts/readiness/strict_selfhost_dependency_inventory.json`
- `artifacts/readiness/selfhost_frontier_verify.json`
- `artifacts/readiness/selfhost_examples_verify.json`
- `artifacts/readiness/selfhost_bootstrap_verify.json`
- `artifacts/readiness/selfhost_negative_verify.json`
- `artifacts/readiness/selfhost_audited_surface_verify.json`

## Current Interpretation

- `strict_selfhost` readiness originally froze the dependency inventory and the blocker model.
- `v3.3.0` shipped-surface strict-selfhost line is complete.
- It requires both the frozen declaration frontier and the shipped `examples/`
  corpus to stay green under the self-host frontend audit.
- It also requires bootstrap compiler sources and a curated negative semantic
  corpus to stay green under the self-host frontend audit contract.
- It requires a curated audited executable surface to pass `frontend-audit`,
  `selfhost-ci`, `replace-check`, and `mainline-ci` proof through one bundled
  verification artifact.
- Validation examples that depend on the repository `examples/` package layout
  remain part of the shipped examples audit, but are not forced through the
  bundled audited-surface materialization.
- For the shipped strict-selfhost surface, there are currently no unresolved
  blocking subsystems.
- The broader `v3.1.0 -> v4.0.0` zero-Rust target remains open beyond the
  shipped-surface closure proof and can still add future replacement work.

## Historical Blocking Subsystems (closed in `v3.3.0`)

The shipped-surface blockers that were frozen and then closed in `v3.3.0` were:

- compiler frontend
- runtime core
- systems / CLI orchestration
- native std / acceleration layer
- tensor backend
- data / registry layer

These no longer block the shipped strict-selfhost line. They remain useful as
historical decomposition for broader future zero-Rust work outside the already
closed `v3.3.0` shipped surface.

## Post-Closure Next-Step Baseline (`v3.4.0`)

After the `v3.3.0` shipped-surface closure, the next zero-Rust work is broader
than the strict-selfhost shipped release gate. The current `v3.4.0` baseline
tracks these follow-on categories explicitly:

- cross-host install-flow proof beyond the current Windows-host execution evidence
- compatibility-only storage/data paths that remain tolerated outside shipped
  strict-selfhost blockers
- globally accelerated native/tensor backend replacement beyond the current
  runtime-owned fallback boundary
- broader zero-Rust closure of historical non-shipped compatibility paths on
  the way to the eventual `v4.0.0` target

These are roadmap categories for the broader zero-Rust program. They do not
reopen the already closed `v3.3.0` strict-selfhost shipped surface.

## Compatibility-Only Storage/Data Baseline (`v3.4.0`)

The first concrete storage/data post-closure tranche makes the currently
tolerated compatibility path explicit instead of leaving it implied:

- `sqlite_binding` is the compatibility-only storage/data path still tracked
  outside shipped strict-selfhost release blockers
- the shipped strict-selfhost data/registry surface remains complete
- broader future work to replace SQLite-backed compatibility paths globally
  remains roadmap work on the way to `v4.0.0`

The source-of-record proof boundary for this tranche is:

- `artifacts/readiness/strict_selfhost_dependency_inventory.json`
- `artifacts/readiness/strict_selfhost_data_registry_protocols_surface.json`
- `artifacts/readiness/v3_4_0_compatibility_storage_data_baseline.json`

## Accelerated Native/Tensor Backend Baseline (`v3.4.0`)

The next concrete post-closure tranche makes the accelerated backend boundary
explicit instead of leaving it implied:

- the shipped strict-selfhost tensor and native std/accel surfaces remain complete
- runtime-owned fallback and shipped proof boundaries are closed for the shipped
  line
- broader global replacement of accelerated Rust/native tensor backends remains
  roadmap work on the way to `v4.0.0`

The source-of-record proof boundary for this tranche is:

- `artifacts/readiness/strict_selfhost_tensor_backend_surface.json`
- `artifacts/readiness/strict_selfhost_native_std_and_accel_surface.json`
- `artifacts/readiness/strict_selfhost_dependency_inventory.json`
- `artifacts/readiness/v3_4_0_accelerated_native_tensor_baseline.json`

## Historical Non-Shipped Compatibility Closure Baseline (`v3.4.0`)

The final post-closure baseline makes the broader non-shipped compatibility
boundary explicit:

- the shipped strict-selfhost release surface remains complete
- broader historical compatibility paths outside the shipped release gate are
  explicitly tracked as future roadmap work toward `v4.0.0`
- these paths no longer exist as an implicit unresolved category inside the
  `v3.4.0` line

The source-of-record proof boundary for this tranche is:

- `artifacts/readiness/v3_4_0_zero_rust_next_step_baseline.json`
- `artifacts/readiness/v3_4_0_install_host_matrix_baseline.json`
- `artifacts/readiness/v3_4_0_compatibility_storage_data_baseline.json`
- `artifacts/readiness/v3_4_0_accelerated_native_tensor_baseline.json`
- `artifacts/readiness/v3_4_0_non_shipped_compatibility_closure_baseline.json`

## Objective-Definition Closure Baseline (`v3.5.0`)

The `v3.5.0` line does not claim to replace those broader global paths yet.
Instead, it closes a bounded objective-definition release line:

- the shipped strict-selfhost surface remains closed
- the completed `v3.4.0` post-closure baselines remain the source-of-record
- the broader global replacement categories are explicitly frozen as the next
  program boundary rather than left implicit

The source-of-record proof boundary for this tranche is:

- `artifacts/readiness/v3_5_0_release_line_start_baseline.json`
- `artifacts/readiness/v3_5_0_objective_set_freeze.json`
- `artifacts/readiness/v3_5_0_evidence_continuity_baseline.json`
- `artifacts/readiness/v3_5_0_closure.json`

## Objective-Definition Closure Baseline (`v3.6.0`)

The `v3.6.0` line keeps the same discipline and closes another bounded
objective-definition release line:

- the shipped strict-selfhost surface remains closed
- the completed `v3.4.0` and `v3.5.0` post-closure baselines remain the
  source-of-record
- the broader global replacement categories remain explicitly frozen as the next
  program boundary rather than left implicit

The source-of-record proof boundary for this tranche is:

- `artifacts/readiness/v3_6_0_release_line_start_baseline.json`
- `artifacts/readiness/v3_6_0_objective_set_freeze.json`
- `artifacts/readiness/v3_6_0_evidence_continuity_baseline.json`
- `artifacts/readiness/v3_6_0_closure.json`

## Global Self-Host AI Runtime Foundation (v3.7.0)

`v3.7.0` starts the first broader implementation tranche after the bounded
post-closure baseline lines. The shipped strict-selfhost surface remains closed;
this line moves the global self-host program into a bounded AI runtime frontier.

The current bounded frontier is:

- single-node training
- single-node pretraining
- deterministic eval
- checkpoint save/load and resume
- dataset ingest for the frozen suite
- no distributed training in this tranche

The implementation boundary for this tranche is:

- `enkai_accel` becomes the explicit backend class for the bounded suite
- the current Rust/native tensor backend remains available only as
  migration-time comparison/fallback
- benchmark claims are bounded to the frozen suite and machine profile
- memory, safety, fallback, and backend-selection evidence must be archived in
  deterministic proof artifacts

The source-of-record proof boundary for this tranche is:

- `bench/suites/v3_7_0_ai_runtime_foundation.json`
- `enkai/contracts/v3_7_0_ai_runtime_foundation.json`
- `artifacts/readiness/v3_7_0_ai_runtime_foundation.json`
- `artifacts/readiness/v3_7_0_ai_runtime_foundation_verify.json`

## Performance Delta Gates (v3.7.0)

The second concrete `v3.7.0` tranche does not widen the runtime frontier. It
freezes the stronger benchmark and regression view for the same bounded suite:

- the original tiny benchmark is replaced by a stronger pinned throughput suite
- Enkai `enkai_accel` is compared against Python, the CPU scalar path, and the
  current `native` comparison/fallback path
- peak memory and checkpoint overhead are guarded against regression relative to
  the CPU scalar baseline

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_ai_runtime_perf_deltas.json`
- `artifacts/readiness/v3_7_0_ai_runtime_foundation.json`
- `artifacts/readiness/v3_7_0_ai_runtime_perf_deltas.json`

## Threaded Acceleration Determinism (v3.7.0)

The next `v3.7.0` tranche upgrades `enkai_accel` from a distinct kernel to a
deterministic multithreaded execution path for the bounded suite:

- repeated bounded-suite runs must keep the same worker-count choice
- train-report payloads and checkpoint payloads must remain stable across runs
- deterministic losses and checkpoint sizes must remain stable even with
  multithreaded execution

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_ai_runtime_threaded_determinism.json`
- `artifacts/readiness/v3_7_0_ai_runtime_foundation.json`
- `artifacts/readiness/v3_7_0_ai_runtime_threaded_determinism.json`

## Model-Shape Frontier and Latency Baselines (v3.7.0)

The next `v3.7.0` tranche broadens the bounded `enkai_accel` frontier without
claiming unrestricted model coverage:

- additional bounded residual-stack model shapes are part of the proof frontier
- checkpoint-resume latency and eval-only latency are archived as explicit
  baseline evidence
- this remains a bounded single-node surface, not a distributed runtime claim

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_ai_runtime_shape_frontier.json`
- `artifacts/readiness/v3_7_0_ai_runtime_shape_frontier.json`

## Distributed Runtime Design Freeze (v3.7.0)

The first distributed-runtime tranche is intentionally design-only:

- distributed execution remains out of scope for the bounded runtime proofs
- the proof boundary requires the foundation artifact to keep
  `distributed_training = false`
- the design surface is frozen in contract form before execution is widened

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_distributed_runtime_design.json`
- `artifacts/readiness/v3_7_0_distributed_runtime_design.json`

## Checkpoint/Eval Throughput Regression Gates (v3.7.0)

The next bounded `v3.7.0` tranche makes checkpoint/eval throughput an explicit
contract instead of inferring it from latency ratios alone:

- checkpoint resume throughput is gated directly from archived checkpoint bytes
- eval-only throughput is gated directly from processed token volume
- these remain bounded-suite gates, not global performance claims

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_ai_runtime_throughput_regressions.json`
- `artifacts/readiness/v3_7_0_ai_runtime_shape_frontier.json`
- `artifacts/readiness/v3_7_0_ai_runtime_throughput_regressions.json`

## Executable Distributed Runtime Preview (v3.7.0)

The next bounded `v3.7.0` tranche widens the distributed work from pure design
freeze to the first executable preview:

- `enkai_accel` can now run in a bounded rank-sharded preview mode
- the preview is explicitly single-host and does not claim synchronized
  gradient exchange
- execution proof requires each rank to run and emit deterministic runtime
  evidence under explicit distributed opt-in

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_distributed_runtime_exec.json`
- `artifacts/readiness/v3_7_0_distributed_runtime_exec.json`
- `artifacts/readiness/v3_7_0_distributed_runtime_exec_verify.json`

## Synchronized Distributed Preview and Checkpoint Merge/Replay (v3.7.0)

The next bounded `v3.7.0` tranche widens the executable distributed preview
into the first synchronized-gradient preview without overstating it as general
multi-rank training:

- `enkai_accel` can now execute a bounded single-host synchronized-gradient
  preview mode
- gradient exchange is explicitly synchronized and merged across ranks before
  parameter application
- repeated ranks must converge to identical checkpoint payloads under the frozen
  preview surface
- distributed checkpoint merge/replay proof is required before claiming broader
  multi-rank training behavior

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_distributed_runtime_sync.json`
- `artifacts/readiness/v3_7_0_distributed_runtime_sync.json`
- `artifacts/readiness/v3_7_0_distributed_runtime_sync_verify.json`

## Synchronized Distributed Shape Envelope and Throughput Gates (v3.7.0)

The next bounded `v3.7.0` tranche widens the synchronized preview without
claiming unconstrained distributed training:

- larger hidden-size/layer transformer envelopes are part of the synchronized
  preview proof surface
- determinism remains frozen through identical checkpoint semantics and merged
  replay requirements across ranks
- explicit distributed throughput gates are required for:
  - combined train throughput
  - combined eval throughput
  - checkpoint merge throughput

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_distributed_runtime_throughput.json`
- `artifacts/readiness/v3_7_0_distributed_runtime_sync.json`
- `artifacts/readiness/v3_7_0_distributed_runtime_throughput.json`

## Networked Multi-Process Rendezvous Design Freeze (v3.7.0)

The next bounded `v3.7.0` tranche freezes the first networked distributed
rendezvous surface before any execution widening:

- networked rendezvous remains design-only
- the frozen design surface covers:
  - multi-process topology
  - multi-node rendezvous
  - bounded rank registration and synchronization scope
- this tranche does not claim executable networked training yet

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_networked_rendezvous_design.json`
- `artifacts/readiness/v3_7_0_networked_rendezvous_design.json`

## Executable Networked Rendezvous and Barrier/Retry Fault Injection (v3.7.0)

The next bounded `v3.7.0` tranche executes the first networked rendezvous
surface after the earlier design freeze:

- `enkai_accel` runs with `dist.topology = "multi-node"` and a frozen
  `tcp://<host>:<port>` rendezvous
- the execution surface remains bounded to a preview, not a general distributed
  training claim
- barrier/retry fault injection is required and must prove retry-path recovery
  before widening beyond the initial networked world size

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_networked_rendezvous_exec.json`
- `artifacts/readiness/v3_7_0_networked_rendezvous_exec.json`
- `artifacts/readiness/v3_7_0_networked_rendezvous_exec_verify.json`

## Realistic AI Workload Benchmark Matrix (v3.7.0)

The next bounded `v3.7.0` tranche broadens performance evidence into a small
realistic workload matrix without claiming unrestricted model or production
coverage:

- the workload matrix freezes representative categories for:
  - instruction-style supervised fine-tuning
  - retrieval-style question answering with compact context windows
  - code-completion style operator snippets
  - longer-context incident/timeline summarization
  - longer-context policy/retrieval chaining
- each workload must archive:
  - `enkai_accel` train/eval evidence
  - CPU scalar comparison
  - Python comparison
  - native comparison/fallback evidence
  - memory and checkpoint regression gates

Artifacts:

- `bench/suites/v3_7_0_ai_runtime_realistic_workloads.json`
- `enkai/contracts/v3_7_0_ai_runtime_realistic_workloads.json`
- `artifacts/readiness/v3_7_0_ai_runtime_realistic_workloads.json`
- `artifacts/readiness/v3_7_0_ai_runtime_realistic_workloads_verify.json`

## Adversarial Input Corruption Coverage (v3.7.0)

The next bounded `v3.7.0` tranche adds explicit adversarial/corruption evidence
instead of relying only on nominal-path runtime proofs:

- deterministic failure coverage is required for:
  - malformed config payloads
  - invalid multi-node rendezvous settings
  - malformed preview modes
  - corrupted checkpoint payloads

Artifacts:

- `enkai/contracts/v3_7_0_ai_runtime_adversarial_inputs.json`
- `artifacts/readiness/v3_7_0_ai_runtime_adversarial_inputs.json`
- `artifacts/readiness/v3_7_0_ai_runtime_adversarial_inputs_verify.json`

## Bounded AI Runtime QA Floor (v3.7.0)

The next bounded `v3.7.0` tranche stops treating QA as a vague claim for the
covered AI runtime surface and instead computes a contract-defined floor:

- the QA floor requires green evidence for:
  - realistic workload matrix
  - threaded determinism
  - shape frontier
  - synchronized distributed proof
  - distributed throughput proof
  - networked rendezvous execution proof
- the score is explicitly bounded to the `v3.7.0` AI runtime surface; it is
  not a claim about the entire repository or future unimplemented runtime work

Artifacts:

- `enkai/contracts/v3_7_0_ai_runtime_quality_floor.json`
- `artifacts/readiness/v3_7_0_ai_runtime_quality_floor.json`

## AI Runtime Security and Fault Baseline (v3.7.0)

The next bounded `v3.7.0` tranche strengthens the QA floor by freezing a
machine-verifiable security/fault surface for the covered AI runtime work:

- memory-safety evidence must stay green for:
  - invalid backend selection
  - OOM budget enforcement
  - corrupted checkpoint handling
  - invalid runtime state handling
- adversarial input/corruption coverage must stay green
- bounded compliance evidence must stay green for:
  - deterministic validation outputs
  - backend selection archival
  - fallback behavior archival
  - no hidden Rust requirement on the shipped strict-selfhost path
- networked rendezvous fault injection must:
  - exercise a retry path
  - observe the injected fault
  - preserve checkpoint semantics

Artifacts:

- `enkai/contracts/v3_7_0_ai_runtime_security_fault_baseline.json`
- `artifacts/readiness/v3_7_0_ai_runtime_security_fault_baseline.json`

## Larger-World-Size Networked Rendezvous Design Freeze (v3.7.0)

The next bounded `v3.7.0` tranche freezes wider-than-two-rank networked
rendezvous before any execution widening:

- execution remains frozen
- the design surface now explicitly covers:
  - `multi-node` `tcp://` rendezvous
  - `world_size >= 3`
  - multi-process preview topology
  - precondition that the existing 2-rank executable proof remains green first

Artifacts:

- `enkai/contracts/v3_7_0_networked_rendezvous_scale_design.json`
- `artifacts/readiness/v3_7_0_networked_rendezvous_scale_design.json`

## World-Size 4 Networked Rendezvous Execution (v3.7.0)

The next bounded `v3.7.0` tranche executes the frozen wider rendezvous
surface:

- `world_size = 4` is executed on the frozen `tcp://` rendezvous contract
- baseline and retry fault-injection cases must both pass
- ranks must exchange gradient payloads over the `tcp://` rendezvous transport
- checkpoint semantics must remain identical across all ranks
- merged checkpoint replay and train/eval/checkpoint throughput gates must pass
- this tranche still does not claim general distributed training

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_networked_rendezvous_scale_exec.json`
- `artifacts/readiness/v3_7_0_networked_rendezvous_scale_exec.json`
- `artifacts/readiness/v3_7_0_networked_rendezvous_scale_exec_verify.json`

## Networked Gradient Exchange and Adversarial Transport (v3.7.0)

The networked synchronized preview now moves gradient aggregation onto the
`tcp://` transport surface:

- rank 0 receives gradient payloads from peers over TCP
- peers receive the merged gradient payload over TCP
- the single-host file-backed synchronization path remains limited to
  `synchronized-grad-preview`
- adversarial transport cases must fail deterministically:
  - peer disconnect during gradient exchange
  - stale step payload
  - wrong tensor length
  - duplicate rank payload
  - timeout during aggregation

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_networked_gradient_adversarial.json`
- `artifacts/readiness/v3_7_0_networked_gradient_adversarial.json`
- `artifacts/readiness/v3_7_0_networked_gradient_adversarial_verify.json`

## Networked Long-Context Execution (v3.7.0)

The final networked execution tranche before closure proves that `tcp://`
gradient exchange is not limited to the shortest benchmark shape:

- `world_size = 4` remains the bounded networked execution size
- sequence length must be at least 32
- baseline and retry fault-injection cases must both pass
- merged checkpoint replay must remain deterministic
- train/eval/checkpoint/gradient throughput gates must remain green

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_networked_long_context_exec.json`
- `artifacts/readiness/v3_7_0_networked_long_context_exec.json`
- `artifacts/readiness/v3_7_0_networked_long_context_exec_verify.json`

## Networked Throughput Regression Gates (v3.7.0)

Networked throughput is now a parent proof surface rather than only an
implementation detail inside execution reports:

- the `world_size = 4` execution proof must pass train/eval/checkpoint/gradient gates
- the networked long-context proof must pass the same gate classes
- gradient payload exchange must report positive networked byte counts

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_networked_throughput_regressions.json`
- `artifacts/readiness/v3_7_0_networked_throughput_regressions.json`

## Full v3.7.0 Closure

`v3.7.0` is closed only when the closure verifier confirms:

- version metadata is aligned to `3.7.0`
- AI runtime foundation, performance, determinism, realistic workload, safety,
  distributed, networked, and throughput artifacts are all green
- the networked long-context and throughput-regression gates are included in
  the release proof set

The source-of-record proof boundary for this closure is:

- `enkai/contracts/v3_7_0_closure.json`
- `artifacts/readiness/v3_7_0_closure.json`

## Networked Rendezvous Adversarial Peer Behavior (v3.7.0)

The networked rendezvous fault baseline now includes hostile peer behavior,
not only cooperative peers and delayed listener retry:

- malformed peer JSON must fail deterministically
- metadata mismatch must fail deterministically
- out-of-range peer ranks must be rejected at the barrier boundary
- duplicate peer ranks must be rejected at the barrier boundary

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_networked_rendezvous_peer_adversarial.json`
- `artifacts/readiness/v3_7_0_networked_rendezvous_peer_adversarial.json`
- `artifacts/readiness/v3_7_0_networked_rendezvous_peer_adversarial_verify.json`

## Longer-Context Synchronized Distributed Workloads (v3.7.0)

Before widening the networked execution envelope further, the synchronized
distributed preview carries longer-context workload evidence:

- sequence lengths are widened beyond the short-context floor
- deterministic synchronized checkpoint semantics remain required
- merged checkpoint replay remains required
- throughput gates are explicit for the longer-context proof surface

The source-of-record proof boundary for this tranche is:

- `enkai/contracts/v3_7_0_distributed_runtime_long_context_sync.json`
- `artifacts/readiness/v3_7_0_distributed_runtime_long_context_sync.json`
- `artifacts/readiness/v3_7_0_distributed_runtime_long_context_sync_verify.json`
