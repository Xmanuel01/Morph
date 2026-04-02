# 40. Registry Convergence (v2.8.1)

This document defines the `v2.8.1` convergence contract between LLM run artifacts,
Adam-0/simulation artifacts, environment assets, and native extension bundles.

## Goal

Use one auditable lifecycle model for:
- LLM checkpoints
- simulation/world-state snapshots
- environment assets
- native extension bundles

without changing Enkai language syntax.

## CLI Surface

Additive model-registry commands in `v2.8.1`:

- `enkai model register <registry_dir> <name> <version> <artifact_path> [--activate] [--artifact-kind <checkpoint|simulation|environment|native-extension>] [--artifact-manifest <file>] [--lineage-manifest <file>]`
- `enkai model push <registry_dir> <name> <version> --registry <remote_registry_dir> [--sign]`
- `enkai model pull <registry_dir> <name> <version> --registry <remote_registry_dir> [--verify-signature] [--fallback-local]`
- `enkai model promote-remote|retire-remote|rollback-remote ...`
- `enkai model verify-signature <registry_dir> <name> <version> --registry <remote_registry_dir>`

## Evidence

Full-platform readiness now archives and verifies:

- `artifacts/readiness/model_registry_convergence.json`
- `artifacts/readiness/model_registry_convergence_verify.json`
- `artifacts/registry/sim_lineage.json`
- `artifacts/registry/sim_snapshot.manifest.json`
- `artifacts/registry/local/registry.json`
- `artifacts/registry/remote/registry.json`
- `artifacts/registry/cache/registry.json`
- `artifacts/registry/remote/adam0-sim/v2.8.1/remote.manifest.json`
- `artifacts/registry/remote/adam0-sim/v2.8.1/remote.manifest.sig`

## Lineage Contract

Simulation runs now emit machine-verifiable lineage and snapshot manifests through the
`enkai sim` CLI by pairing:

- `--lineage-output <file>`
- `--snapshot-manifest-output <file>`

Those artifacts carry:
- config hash
- environment hash
- source hash
- run identity
- command identity
- snapshot hash

This matches the existing LLM-side run-state/checkpoint lineage model closely enough for
one shared registry/audit flow.

## Sign-Off

`v2.8.1` is complete only when:
- simulation lineage manifests are emitted and archived
- non-checkpoint artifact kinds register/push/pull/verify successfully
- remote signed manifests verify cleanly
- local, remote, and cache registries agree on artifact identity and kind
