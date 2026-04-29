# Enkai Installation Guide

This document covers the full flow from download → install → first program.

## 1) Download Enkai

Enkai is distributed as prebuilt binaries via GitHub Releases.

### Release assets (expected names)

- `enkai-<version>-windows-x86_64.zip`
- `enkai-<version>-linux-x86_64.tar.gz`
- `enkai-<version>-linux-aarch64.tar.gz`
- `enkai-<version>-macos-x86_64.tar.gz`
- `enkai-<version>-macos-aarch64.tar.gz`
- `enkai-setup-<version>.exe` (Windows installer)

Each asset also ships with a `.sha256` checksum.

## 2) Install Enkai

### Option A: Install script (recommended)

Linux/macOS:

```
curl -fsSL https://.../install.sh | sh
```

Windows (PowerShell):

```
iwr -useb https://.../install.ps1 | iex
```

These scripts:
- detect OS + CPU
- download the right binary
- verify checksum (if available)
- install to a safe folder
- add that folder to PATH
- do not require `cargo` or `rustc` on the target machine
- support local-bundle proof mode via:
  - PowerShell: `-BundlePath <zip> -InstallDir <dir> -NoPathUpdate`
  - shell: `--bundle-path <tar.gz> --install-dir <dir> --no-path-update`
- support uninstall proof mode via:
  - PowerShell: `-Uninstall -InstallDir <dir>`
  - shell: `--uninstall --install-dir <dir>`

### Option B: Manual install

1) Download the correct release asset
2) Extract `enkai` / `enkai.exe`
3) Move it to a PATH folder:
   - Linux/macOS: `~/.local/bin/enkai`
   - Windows: `C:\Users\Name\.enkai\bin\enkai.exe`
4) Verify:

```
enkai --version
enkai install-diagnostics --json
```

## 3) Write your first program

Create `hello.enk`:

```
fn main() ::
  print("Hello from Enkai!")
::
```

Run it:

```
enkai run hello.enk
```

## 4) What happens when you run

`enkai run` performs:

1) Parse → AST
2) Type-check
3) Compile to bytecode
4) Execute in the VM

The installed bundle is intended to be operational on a machine that does not have Rust installed.
Rust remains a repo-side build/bootstrap dependency only while the zero-Rust self-host line is still closing.
`enkai install-diagnostics` reports whether the bundle layout is intact, whether the bundled stdlib/examples are present, and whether `cargo`/`rustc` are visible on the current PATH.
The install proof for the `v3.3.0` release line is contract-driven:
- `enkai/contracts/install_bundle_v3_3_0.json`
- `enkai/contracts/zero_rust_closure_v3_3_0.json`
and emits:
- `artifacts/install_bundle_smoke/install_bundle_manifest.json`
- `artifacts/install_bundle_smoke/install_bundle_smoke.json`
- `artifacts/install_bundle_smoke/zero_rust_closure.json`
- `artifacts/install_bundle_smoke/install_flow_proof.json`
Release archives also embed `bundle_manifest.json`, and `scripts/verify_release_artifact.py`
checks that manifest against the archive checksum, target OS, architecture, and version.

## 5) Building from source (developers)

```
git clone https://github.com/Xmanuel01/Enkai.git
cd Enkai
cargo build -p enkai --release
```

Then run:

```
target/release/enkai --version
```

## 6) Windows installer

The installer (`enkai-setup-<version>.exe`) installs Enkai into
`C:\Users\Name\Enkai` and adds it to your user PATH.

## 7) Windows EV code signing (guidance)

For a production-grade Windows installer, use an EV code signing certificate.

Recommended approach:

1) Obtain an EV code signing certificate (PFX).
2) Store the PFX as GitHub Actions secrets:
   - `WINDOWS_PFX_BASE64` (base64-encoded PFX)
   - `WINDOWS_PFX_PASSWORD`
3) The release workflow uses `signtool.exe` to sign:
   - `enkai.exe`
   - Inno Setup EXE
   - WiX MSI

If the secrets are not set, signing is skipped.

## 8) Notes on Native Extensions

Optional native-backed modules require the matching companion library in the install directory or on PATH.
The base language toolchain (`enkai --version`, compile, run, bundled stdlib/examples) is validated as an install bundle without requiring the Rust toolchain on the target machine.


