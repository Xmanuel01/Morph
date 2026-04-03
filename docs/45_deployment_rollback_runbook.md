# 45. Deployment And Rollback Runbook

This runbook defines the bounded deployment and rollback flow for backend,
fullstack, mobile-supporting, and worker-backed Enkai applications.

## Preconditions

- deploy validation passes:
  - `enkai deploy validate <project_dir> --profile backend --strict --json --output artifacts/readiness/deploy_backend.json`
  - `enkai deploy validate <project_dir> --profile fullstack --strict --json --output artifacts/readiness/deploy_fullstack.json`
  - `enkai deploy validate <project_dir> --profile mobile --strict --json --output artifacts/readiness/deploy_mobile.json`
- release readiness is green:
  - `artifacts/readiness/full_platform.json`
- if using workers:
  - `artifacts/readiness/worker_queue_smoke.json`
  - `artifacts/readiness/worker_queue_evidence_verify.json`

## Deployment Order

1. Validate config and migration state.
2. Apply migrations.
3. Start backend services.
4. Start worker processes if required.
5. Start frontend/mobile clients.
6. Verify readiness and smoke evidence.

## Rollback Rules

- Never roll back schema blindly.
- Roll back in this order:
  1. stop new traffic
  2. pin previous backend/model version
  3. pin previous frontend/mobile package version
  4. restore previous worker image if needed
  5. verify readiness before re-enabling traffic

## Artifact Checks

- package checksum must match
- SBOM must be present
- archived blocker report must pass
- backend HTTP/gRPC contracts must still match the pinned version

## Failure Cases

- migration drift:
  - block rollout
- registry/signature failure:
  - use local cache only
- worker dead-letter growth:
  - drain queue and inspect retry policy before re-enable
- gRPC readiness failure:
  - do not promote the backend

