# 54. LLM Package Registry

This document defines the first stable Enkai LLM model/package registry tranche for the `v4.0` native training runtime line.

## Scope

The goal is a deterministic package ecosystem for local and remote LLM artifacts. This is not a claim that Enkai already has a public package index at PyPI/Hugging Face scale. The closed scope is:

- stable package manifest schema: `enkai.llm.package.v1`
- stable lockfile schema: `enkai.llm.lock.v1`
- exact dependency resolution
- file-level SHA-256 integrity checks
- model card and license requirements
- deterministic tamper rejection
- deterministic dependency conflict rejection
- explicit no-PyTorch core runtime dependency
- explicit ban on native postinstall hooks

## Package Manifest

Every package version has an `enkai.llm.json` manifest:

```json
{
  "schema": "enkai.llm.package.v1",
  "schema_version": 1,
  "package": "enkai/tiny-llm",
  "version": "1.0.0",
  "license": "Apache-2.0",
  "family": "decoder-only-transformer",
  "architecture": "tiny-transformer",
  "tasks": ["causal-lm", "text-generation"],
  "runtime": {
    "devices": ["cpu", "cuda:0"],
    "dtypes": ["fp32", "fp16", "bf16"],
    "requires_pytorch": false,
    "pytorch_reference_only": true
  },
  "security": {
    "native_postinstall_hooks": false,
    "network_on_install": false,
    "requires_signature_for_remote_install": true
  },
  "dependencies": [
    {"package": "enkai/tokenizer-bpe", "version": "1.0.0"}
  ],
  "files": [
    {"path": "model_card.md", "role": "model_card", "media_type": "text/markdown", "bytes": 100, "sha256": "..."}
  ],
  "manifest_digest": "..."
}
```

Rules:

- `package` must be lowercase `namespace/name`.
- `version` must be semantic version format.
- every file must be relative, unique, present, byte-counted, and SHA-256 pinned.
- every installable package must include a `model_card` file.
- package runtime must not require PyTorch; PyTorch is allowed only as a reference benchmark/correctness baseline.
- native postinstall hooks and network-on-install are forbidden.

## Lockfile

Every deterministic install claim requires `enkai.lock.json`:

```json
{
  "schema": "enkai.llm.lock.v1",
  "schema_version": 1,
  "root": "enkai/tiny-llm@1.0.0",
  "resolver": "exact-v1",
  "packages": [
    {
      "package": "enkai/tokenizer-bpe",
      "version": "1.0.0",
      "manifest_digest": "...",
      "files": [{"path": "tokenizer.json", "sha256": "...", "bytes": 64}]
    }
  ],
  "lock_digest": "..."
}
```

The first resolver is intentionally exact-version only. Wider semver ranges can be added later, but production installs should remain lockfile-pinned.

## Standard Helpers

`std::llm_registry` exposes small manifest-fragment helpers for Enkai code:

- `llm_registry.package(name, version, family, architecture)`
- `llm_registry.dependency(name, version)`
- `llm_registry.file(path, role, media_type, bytes, sha256)`
- `llm_registry.runtime_cpu()`
- `llm_registry.runtime_cpu_cuda()`
- `llm_registry.security()`

These helpers do not replace verifier-side hashing or lock resolution. The verifier/installer boundary is still responsible for trust decisions.

## Evidence

The tranche is proven by:

```text
python scripts/readiness_v4_0_0_llm_package_registry.py
python scripts/verify_v4_0_0_llm_package_registry.py
```

Expected artifacts:

- `artifacts/registry/llm_package_ecosystem/registry.index.json`
- `artifacts/registry/llm_package_ecosystem/packages/enkai/tiny-llm/1.0.0/enkai.llm.json`
- `artifacts/registry/llm_package_ecosystem/packages/enkai/tiny-llm/1.0.0/enkai.lock.json`
- `artifacts/readiness/v4_0_0_llm_package_registry.json`
- `artifacts/readiness/v4_0_0_llm_package_registry_verify.json`

Green evidence means the bounded package ecosystem is stable and deterministic. It does not mean Enkai has a public hosted package marketplace yet.
