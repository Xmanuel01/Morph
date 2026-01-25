# Morph Installation Guide

This document covers the full flow from download → install → first program.

## 1) Download Morph

Morph is distributed as prebuilt binaries via GitHub Releases.

### Release assets (expected names)

- `morph-<version>-windows-x86_64.zip`
- `morph-<version>-linux-x86_64.tar.gz`
- `morph-<version>-linux-aarch64.tar.gz`
- `morph-<version>-macos-x86_64.tar.gz`
- `morph-<version>-macos-aarch64.tar.gz`
- `morph-setup-<version>.exe` (Windows installer)

Each asset also ships with a `.sha256` checksum.

## 2) Install Morph

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

### Option B: Manual install

1) Download the correct release asset
2) Extract `morph` / `morph.exe`
3) Move it to a PATH folder:
   - Linux/macOS: `~/.local/bin/morph`
   - Windows: `C:\Users\Name\.morph\bin\morph.exe`
4) Verify:

```
morph --version
```

## 3) Write your first program

Create `hello.morph`:

```
fn main() ::
  print("Hello from Morph!")
::
```

Run it:

```
morph run hello.morph
```

## 4) What happens when you run

`morph run` performs:

1) Parse → AST
2) Type-check
3) Compile to bytecode
4) Execute in the VM

## 5) Building from source (developers)

```
git clone https://github.com/Xmanuel01/Morph.git
cd Morph
cargo build -p morph --release
```

Then run:

```
target/release/morph --version
```

## 6) Windows installer

The installer (`morph-setup-<version>.exe`) installs Morph into
`C:\Users\Name\Morph` and adds it to your user PATH.

## 7) Windows EV code signing (guidance)

For a production-grade Windows installer, use an EV code signing certificate.

Recommended approach:

1) Obtain an EV code signing certificate (PFX).
2) Store the PFX as GitHub Actions secrets:
   - `WINDOWS_PFX_BASE64` (base64-encoded PFX)
   - `WINDOWS_PFX_PASSWORD`
3) The release workflow uses `signtool.exe` to sign:
   - `morph.exe`
   - Inno Setup EXE
   - WiX MSI

If the secrets are not set, signing is skipped.

## 7) Notes on FFI

Native-backed modules require the native library:

```
cargo build -p morph_native --release
```

Ensure `morph_native.dll` is in the current directory or on PATH when running.
