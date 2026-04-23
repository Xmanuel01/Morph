Enkai Quickstart

1) Verify installation:
   enkai --version
   enkai install-diagnostics --json

2) Run the hello example:
   enkai run examples/hello/main.enk

3) Create your own program:
   echo "print(\"Hello Enkai\")" > hello.enk
   enkai run hello.enk

4) Optional proof-mode flags for local bundle testing:
   PowerShell:
     powershell -File install.ps1 -BundlePath <zip> -InstallDir <dir> -NoPathUpdate
     powershell -File install.ps1 -Uninstall -InstallDir <dir>
   shell:
     sh install.sh --bundle-path <tar.gz> --install-dir <dir> --no-path-update
     sh install.sh --uninstall --install-dir <dir>

Notes:
- The std/ folder must remain next to the enkai executable.
- The installed bundle does not require cargo or rustc on the target machine.
- If native modules are included, keep the enkai_native library in the same folder.
- The v3.2.1 install proof writes:
  - artifacts/install_bundle_smoke/install_bundle_manifest.json
  - artifacts/install_bundle_smoke/install_bundle_smoke.json
  - artifacts/install_bundle_smoke/zero_rust_closure.json

