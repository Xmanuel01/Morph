[CmdletBinding()]
param(
    [string]$Version,
    [string]$Arch = "x86_64",
    [string]$OutRoot = "release-assets",
    [switch]$SkipBuild,
    [switch]$NoSmoke,
    [switch]$NoInstaller
)

$ErrorActionPreference = "Stop"

function Get-CrateVersion {
    $cargoToml = Join-Path $PSScriptRoot "..\enkai\Cargo.toml"
    $text = Get-Content -Path $cargoToml -Raw
    $match = [regex]::Match($text, '(?m)^version\s*=\s*"([^"]+)"')
    if (-not $match.Success) {
        throw "Failed to parse version from $cargoToml"
    }
    return $match.Groups[1].Value
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    if (-not $Version) {
        $Version = Get-CrateVersion
    }
    $versionTag = if ($Version.StartsWith("v")) { $Version } else { "v$Version" }
    $versionNumber = $Version.TrimStart("v")
    $outDir = Join-Path $OutRoot (Join-Path $versionTag "windows-$Arch")

    if (-not $SkipBuild.IsPresent) {
        cargo build -p enkai --release
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        cargo build -p enkai_native --release
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $enkaiExe = "target\release\enkai.exe"
    if (-not (Test-Path $enkaiExe)) {
        throw "Missing release binary: $enkaiExe"
    }

    $nativeDll = "target\release\enkai_native.dll"
    $packageArgs = @(
        "scripts\package_release.py",
        "--version", $versionNumber,
        "--target-os", "windows",
        "--arch", $Arch,
        "--archive-format", "zip",
        "--bin", $enkaiExe,
        "--out-dir", $outDir,
        "--check-deterministic"
    )
    if (Test-Path $nativeDll) {
        $packageArgs += @("--native", $nativeDll)
    }

    py -3 @packageArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $archive = Join-Path $outDir "enkai-$versionNumber-windows-$Arch.zip"
    $verifyArgs = @(
        "scripts\verify_release_artifact.py",
        "--archive", $archive,
        "--target-os", "windows",
        "--version", $versionNumber,
        "--arch", $Arch
    )
    if (-not $NoSmoke.IsPresent) {
        $verifyArgs += "--smoke"
    }
    py -3 @verifyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $installer = Join-Path $outDir "enkai-$versionNumber-windows-$Arch-installer.exe"
    if (-not $NoInstaller.IsPresent) {
        & (Join-Path $PSScriptRoot "build_windows_installer.ps1") `
            -Version $versionNumber `
            -Arch $Arch `
            -OutRoot $OutRoot `
            -BundleZip $archive
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    Write-Host ""
    Write-Host "Upload these files to the GitHub Release for ${versionTag}:"
    Write-Host "  $archive"
    Write-Host "  $archive.sha256"
    if (-not $NoInstaller.IsPresent) {
        Write-Host "  $installer"
        Write-Host "  $installer.sha256"
    }
}
finally {
    Pop-Location
}
