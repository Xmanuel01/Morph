# Release Checklist

Use this checklist for production releases.

## 1) Prep

- [ ] Pull latest `main`
- [ ] Update versions (workspace crates as needed)
- [ ] Update `CHANGELOG.md`
- [ ] Ensure release notes are ready

## 2) Build + QA

- [ ] `cargo fmt`
- [ ] `cargo clippy -- -D warnings`
- [ ] `cargo test`
- [ ] Complete `VALIDATION.md` gates for the target release

## 3) Tag

- [ ] `scripts/release.sh vX.Y.Z` (creates annotated tag)
- [ ] Push branch + tag:
  - `git push origin main`
  - `git push origin vX.Y.Z`

## 4) GitHub Release

- [ ] Verify CI release workflow ran for tag
- [ ] Confirm assets:
  - `enkai-<version>-windows-x86_64.zip`
  - `enkai-<version>-linux-x86_64.tar.gz`
  - `enkai-<version>-linux-aarch64.tar.gz`
  - `enkai-<version>-macos-x86_64.tar.gz`
  - `enkai-<version>-macos-aarch64.tar.gz`
  - `enkai-setup-<version>.exe`
  - `.sha256` for each asset

## 5) Post-release

- [ ] Validate install scripts against the new release
- [ ] Smoke test:
  - `enkai --version`
  - `enkai run examples/...`

