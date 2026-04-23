[CmdletBinding()]
param(
    [string]$Repo = "Xmanuel01/Enkai",
    [string]$Version = "latest",
    [string]$BundlePath,
    [string]$InstallDir = (Join-Path $env:USERPROFILE ".enkai\\bin"),
    [switch]$NoPathUpdate,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

function Get-ManagedEntries {
    return @(
        "enkai.exe",
        "enkai_native.dll",
        "README.txt",
        "bundle_manifest.json",
        "std",
        "examples",
        "install_manifest.json"
    )
}

function Remove-ManagedEntries {
    param([string]$TargetInstallDir)
    foreach ($entry in Get-ManagedEntries) {
        $path = Join-Path $TargetInstallDir $entry
        if (Test-Path $path) {
            Remove-Item -Recurse -Force $path
        }
    }
    if ((Test-Path $TargetInstallDir) -and -not (Get-ChildItem -Force $TargetInstallDir | Select-Object -First 1)) {
        Remove-Item -Force $TargetInstallDir
    }
}

function Write-InstallManifest {
    param(
        [string]$TargetInstallDir,
        [string]$InstalledVersion,
        [string]$SourceType,
        [string]$SourceValue
    )
    $payload = @{
        schema_version = 1
        installed_version = $InstalledVersion
        source_type = $SourceType
        source_value = $SourceValue
        managed_entries = Get-ManagedEntries
    } | ConvertTo-Json -Depth 4
    Set-Content -Path (Join-Path $TargetInstallDir "install_manifest.json") -Value $payload -Encoding UTF8
}

function Test-BundleManifestVersion {
    param(
        [string]$TargetInstallDir,
        [string]$InstalledVersion
    )
    $manifestPath = Join-Path $TargetInstallDir "bundle_manifest.json"
    if (-not (Test-Path $manifestPath)) { return }
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.version -and $manifest.version -ne $InstalledVersion) {
        throw "Bundle manifest version $($manifest.version) does not match installed binary version $InstalledVersion"
    }
}

function Get-InstalledVersion {
    param([string]$TargetInstallDir)
    $output = & (Join-Path $TargetInstallDir "enkai.exe") --version
    if (-not $output) {
        throw "Failed to query installed Enkai version"
    }
    $match = [regex]::Match(($output | Out-String), 'Enkai v([0-9]+\.[0-9]+\.[0-9]+)')
    if (-not $match.Success) {
        throw "Failed to parse installed Enkai version from output: $output"
    }
    return $match.Groups[1].Value
}

function Resolve-BundleArchive {
    param(
        [string]$RepoName,
        [string]$RequestedVersion,
        [string]$ArchiveOverride,
        [string]$TempRoot
    )
    if ($ArchiveOverride) {
        $resolved = Resolve-Path $ArchiveOverride -ErrorAction Stop
        return @{
            Tag = $RequestedVersion
            Version = $RequestedVersion.TrimStart('v')
            ArchivePath = $resolved.Path
            SourceType = "local_bundle"
            SourceValue = $resolved.Path
        }
    }

    if ($RequestedVersion -eq "latest") {
        $api = "https://api.github.com/repos/$RepoName/releases/latest"
    } else {
        $api = "https://api.github.com/repos/$RepoName/releases/tags/$RequestedVersion"
    }

    $release = Invoke-RestMethod -Uri $api -Headers @{ "User-Agent" = "enkai-installer" }
    $tag = $release.tag_name
    if (-not $tag) { throw "Failed to determine release tag" }
    $ver = $tag.TrimStart('v')
    $asset = "enkai-$ver-windows-x86_64.zip"
    $assetObj = $release.assets | Where-Object { $_.name -eq $asset } | Select-Object -First 1
    if (-not $assetObj) { throw "Asset not found: $asset" }
    $checksumObj = $release.assets | Where-Object { $_.name -eq "$asset.sha256" } | Select-Object -First 1

    $zip = Join-Path $TempRoot $asset
    Invoke-WebRequest -Uri $assetObj.browser_download_url -OutFile $zip
    if ($checksumObj) {
        $checksumPath = Join-Path $TempRoot "$asset.sha256"
        Invoke-WebRequest -Uri $checksumObj.browser_download_url -OutFile $checksumPath
        $expected = (Get-Content $checksumPath -Raw).Trim().Split(" ")[0]
        $actual = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
        if ($actual -ne $expected.ToLowerInvariant()) {
            throw "Checksum mismatch for $asset"
        }
    }

    return @{
        Tag = $tag
        Version = $ver
        ArchivePath = $zip
        SourceType = "github_release"
        SourceValue = $assetObj.browser_download_url
    }
}

if ($Uninstall.IsPresent) {
    Remove-ManagedEntries -TargetInstallDir $InstallDir
    Write-Host "Uninstalled Enkai from $InstallDir"
    exit 0
}

$temp = Join-Path $env:TEMP ("enkai-install-" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Force -Path $temp | Out-Null
try {
    $bundle = Resolve-BundleArchive -RepoName $Repo -RequestedVersion $Version -ArchiveOverride $BundlePath -TempRoot $temp

    $extract = Join-Path $temp "extract"
    Expand-Archive -Path $bundle.ArchivePath -DestinationPath $extract -Force

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Remove-ManagedEntries -TargetInstallDir $InstallDir
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

    Copy-Item -Path (Join-Path $extract "enkai.exe") -Destination (Join-Path $InstallDir "enkai.exe") -Force
    if (Test-Path (Join-Path $extract "enkai_native.dll")) {
        Copy-Item -Path (Join-Path $extract "enkai_native.dll") -Destination $InstallDir -Force
    }
    if (Test-Path (Join-Path $extract "std")) {
        Copy-Item -Recurse -Path (Join-Path $extract "std") -Destination (Join-Path $InstallDir "std") -Force
    }
    if (Test-Path (Join-Path $extract "examples")) {
        Copy-Item -Recurse -Path (Join-Path $extract "examples") -Destination (Join-Path $InstallDir "examples") -Force
    }
if (Test-Path (Join-Path $extract "README.txt")) {
    Copy-Item -Path (Join-Path $extract "README.txt") -Destination (Join-Path $InstallDir "README.txt") -Force
}
if (Test-Path (Join-Path $extract "bundle_manifest.json")) {
    Copy-Item -Path (Join-Path $extract "bundle_manifest.json") -Destination (Join-Path $InstallDir "bundle_manifest.json") -Force
}

    $installedVersion = Get-InstalledVersion -TargetInstallDir $InstallDir
    Test-BundleManifestVersion -TargetInstallDir $InstallDir -InstalledVersion $installedVersion
    Write-InstallManifest -TargetInstallDir $InstallDir -InstalledVersion $installedVersion -SourceType $bundle.SourceType -SourceValue $bundle.SourceValue

    if (-not $NoPathUpdate.IsPresent) {
        $env:PATH = "$InstallDir;$env:PATH"
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if (-not $userPath) { $userPath = "" }
        if ($userPath -notmatch [Regex]::Escape($InstallDir)) {
            [Environment]::SetEnvironmentVariable("Path", "$InstallDir;$userPath", "User")
            Write-Host "Added $InstallDir to PATH. Restart your shell."
        }
    }

    & (Join-Path $InstallDir "enkai.exe") --version
    Write-Host "Installed Enkai to $InstallDir"
}
finally {
    if (Test-Path $temp) {
        Remove-Item -Recurse -Force $temp
    }
}
