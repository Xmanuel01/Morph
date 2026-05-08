# Release Assets

This directory is for local upload-ready release bundles and installers.

Generated archives and installers are intentionally ignored by git. For each closed tag, run:

```powershell
.\scripts\build_windows_release_asset.ps1 -Version 3.8.0
```

The script writes assets under:

```text
release-assets\v<version>\windows-x86_64\
```

Upload these files to the matching GitHub Release:

```text
enkai-<version>-windows-x86_64.zip
enkai-<version>-windows-x86_64.zip.sha256
enkai-<version>-windows-x86_64-installer.exe
enkai-<version>-windows-x86_64-installer.exe.sha256
```

The installer is a per-user Windows installer. It installs to:

```text
%LOCALAPPDATA%\Programs\Enkai
```

It also sets `ENKAI_HOME` and appends the install directory to the user's `PATH`.

To build only an installer from an existing zip:

```powershell
.\scripts\build_windows_installer.ps1 -Version 3.8.0 -Arch x86_64
```

Additional Windows installer links such as 32-bit or ARM64 require matching target bundles first:

```text
release-assets\v<version>\windows-i686\
release-assets\v<version>\windows-aarch64\
```
