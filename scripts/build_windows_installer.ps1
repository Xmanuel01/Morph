[CmdletBinding()]
param(
    [string]$Version,
    [string]$Arch = "x86_64",
    [string]$OutRoot = "release-assets",
    [string]$BundleZip
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

function Get-IExpressPath {
    $candidates = @(
        (Join-Path $env:WINDIR "System32\iexpress.exe"),
        (Join-Path $env:WINDIR "SysWOW64\iexpress.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "Missing iexpress.exe. Windows installer generation requires the built-in IExpress tool."
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
    if (-not $BundleZip) {
        $BundleZip = Join-Path $outDir "enkai-$versionNumber-windows-$Arch.zip"
    }
    if (-not (Test-Path $BundleZip)) {
        throw "Missing bundle zip for installer generation: $BundleZip"
    }

    $resolvedOutDir = (Resolve-Path $outDir).Path
    $resolvedBundleZip = (Resolve-Path $BundleZip).Path
    $installerName = "enkai-$versionNumber-windows-$Arch-installer.exe"
    $installerPath = Join-Path $resolvedOutDir $installerName
    $stagingRoot = Join-Path $resolvedOutDir ".installer-staging"
    $stagingDir = Join-Path $stagingRoot "payload"
    $bundleName = Split-Path $resolvedBundleZip -Leaf

    if (Test-Path $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null
    Copy-Item -LiteralPath $resolvedBundleZip -Destination (Join-Path $stagingDir $bundleName) -Force

    $installScript = @"
`$ErrorActionPreference = "Stop"

`$bundleName = "$bundleName"
`$sourceZip = Join-Path `$PSScriptRoot `$bundleName
`$installDir = Join-Path `$env:LOCALAPPDATA "Programs\Enkai"

if (-not (Test-Path `$sourceZip)) {
    throw "Installer payload is missing: `$sourceZip"
}

if (Test-Path `$installDir) {
    Remove-Item -LiteralPath `$installDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path `$installDir | Out-Null
Expand-Archive -LiteralPath `$sourceZip -DestinationPath `$installDir -Force

[Environment]::SetEnvironmentVariable("ENKAI_HOME", `$installDir, "User")

`$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
`$pathEntries = @()
if (`$userPath) {
    `$pathEntries = `$userPath -split ";" | Where-Object { `$_.Trim().Length -gt 0 }
}
if (-not (`$pathEntries | Where-Object { `$_.TrimEnd("\") -ieq `$installDir.TrimEnd("\") })) {
    `$newPath = if (`$userPath) { "`$userPath;`$installDir" } else { `$installDir }
    [Environment]::SetEnvironmentVariable("Path", `$newPath, "User")
}

`$uninstallScript = @'
`$ErrorActionPreference = "Stop"
`$installDir = Join-Path `$env:LOCALAPPDATA "Programs\Enkai"
`$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (`$userPath) {
    `$newPath = ((`$userPath -split ";") | Where-Object {
        `$_.Trim().Length -gt 0 -and `$_.TrimEnd("\") -ine `$installDir.TrimEnd("\")
    }) -join ";"
    [Environment]::SetEnvironmentVariable("Path", `$newPath, "User")
}
[Environment]::SetEnvironmentVariable("ENKAI_HOME", `$null, "User")
if (Test-Path `$installDir) {
    Remove-Item -LiteralPath `$installDir -Recurse -Force
}
Write-Host "Enkai uninstalled."
'@
Set-Content -Path (Join-Path `$installDir "uninstall.ps1") -Value `$uninstallScript -Encoding UTF8

Write-Host ""
Write-Host "Enkai $versionNumber installed to: `$installDir"
Write-Host "A new terminal may be required before 'enkai' is available on PATH."
Write-Host "To uninstall, run: powershell -ExecutionPolicy Bypass -File `"`$installDir\uninstall.ps1`""
"@
    Set-Content -Path (Join-Path $stagingDir "install.ps1") -Value $installScript -Encoding UTF8

    $sedPath = Join-Path $stagingRoot "enkai-installer.sed"
    $sed = @"
[Version]
Class=IEXPRESS
SEDVersion=3

[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=1
HideExtractAnimation=0
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=
DisplayLicense=
FinishMessage=Enkai $versionNumber has been installed.
TargetName=$installerPath
FriendlyName=Enkai $versionNumber ($Arch) Installer
AppLaunched=powershell.exe -NoProfile -ExecutionPolicy Bypass -File install.ps1
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
SourceFiles=SourceFiles

[Strings]
FILE0=install.ps1
FILE1=$bundleName

[SourceFiles]
SourceFiles0=$stagingDir

[SourceFiles0]
%FILE0%=
%FILE1%=
"@
    Set-Content -Path $sedPath -Value $sed -Encoding ASCII

    $iexpress = Get-IExpressPath
    $process = Start-Process `
        -FilePath $iexpress `
        -ArgumentList @("/N", $sedPath) `
        -Wait `
        -PassThru `
        -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "IExpress installer generation failed with exit code $($process.ExitCode)"
    }
    if (-not (Test-Path $installerPath)) {
        throw "Installer generation completed but output was not found: $installerPath"
    }

    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installerPath).Hash.ToLowerInvariant()
    $hashLine = "$hash  $installerName"
    Set-Content -Path "$installerPath.sha256" -Value $hashLine -Encoding ASCII

    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    $ddfPath = Join-Path $resolvedOutDir "~$($installerName.Replace('.exe', '.DDF'))"
    if (Test-Path $ddfPath) {
        Remove-Item -LiteralPath $ddfPath -Force
    }

    Write-Host ""
    Write-Host "Created Windows installer:"
    Write-Host "  $installerPath"
    Write-Host "  $installerPath.sha256"
}
finally {
    Pop-Location
}
