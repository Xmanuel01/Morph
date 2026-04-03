# 43. LLM Runbook

This runbook defines the supported operational path for LLM training, registry,
and serving on Enkai.

## Scope

This runbook covers:

- `enkai train|pretrain|eval`
- `enkai serve`
- `enkai model`
- local and remote model registry workflows
- checkpoint manifests and resume lineage

## Supported Envelope

- Single-node train/eval/serve
- Multi-node orchestration planning and bounded recovery contracts
- Local filesystem registry plus signed remote registry sync
- HTTP + SSE/WebSocket + gRPC serving

## Required Preconditions

- `enkai readiness check --profile full_platform --json --output artifacts/readiness/full_platform.json`
- registry convergence evidence verifier-clean:
  - `artifacts/readiness/model_registry_convergence.json`
  - `artifacts/readiness/model_registry_convergence_verify.json`
- degraded registry fallback verifier-clean if remote registry is in use:
  - `artifacts/readiness/registry_degraded_smoke.json`
  - `artifacts/readiness/registry_degraded_evidence_verify.json`
- backend serving contract evidence verifier-clean:
  - `artifacts/readiness/grpc_smoke.json`
  - `artifacts/readiness/grpc_evidence_verify.json`

## Daily Operation

1. Validate config and contracts.
   - `enkai doctor --json --strict-contracts`
2. Run train or pretrain.
   - `enkai train <config.enk>`
   - `enkai pretrain <config.enk>`
3. Verify checkpoint and lineage outputs.
   - `checkpoint_dir/run_state.json`
   - `checkpoint_dir/runs/index.jsonl`
   - `checkpoint_dir/checkpoint_lifecycle.json`
4. Register or sync artifacts.
   - `enkai model register ...`
   - `enkai model push ...`
   - `enkai model pull ...`
   - `enkai model verify-signature ...`
5. Start serving.
   - `enkai serve --host 0.0.0.0 --port 8080 --grpc-port 9090 .`

## Serving Controls

- HTTP/SSE/WebSocket and gRPC must expose the same pinned API version contract
- version-pinned model selection must be explicit
- request correlation IDs and structured errors must be preserved
- rate limiting and auth policy must be consistent across protocols

## Failure Handling

- Resume failure:
  - reject the resume if config/code/dataset/seed lineage does not match
- Registry failure:
  - fall back to local cache only
  - suspend promotion/retire flows until remote verification is healthy
- Serve failure:
  - drain traffic
  - inspect `artifacts/grpc/` and HTTP logs
  - restart only after readiness is green again

## Release Sign-Off

LLM sign-off for the current line requires:

- full-platform non-hardware readiness green
- registry convergence verifier-clean
- degraded registry verifier-clean
- backend HTTP/gRPC readiness evidence verifier-clean
- strict archived evidence bundle present under `artifacts/release/v<version>/`

