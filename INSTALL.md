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

### Option B: Manual install

1) Download the correct release asset
2) Extract `enkai` / `enkai.exe`
3) Move it to a PATH folder:
   - Linux/macOS: `~/.local/bin/enkai`
   - Windows: `C:\Users\Name\.enkai\bin\enkai.exe`
4) Verify:

```
enkai --version
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

## 7) Notes on FFI

Native-backed modules require the native library:

```
cargo build -p enkai_native --release
```

Ensure `enkai_native.dll` is in the current directory or on PATH when running.


