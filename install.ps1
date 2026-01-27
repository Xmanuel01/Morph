Param(
    [string]$Version = "latest",
    [string]$Repo = "Xmanuel01/Enkai",
    [string]$InstallDir = "$env:USERPROFILE\\.enkai\\bin"
)

$ErrorActionPreference = "Stop"

function Get-Arch {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    switch ($arch) {
        "X64" { return "x86_64" }
        "Arm64" { return "aarch64" }
        default { throw "Unsupported architecture: $arch" }
    }
}

$arch = Get-Arch
$asset = "enkai-$Version-windows-$arch.zip"
if ($Version -eq "latest") {
    $baseUrl = "https://github.com/$Repo/releases/latest/download"
} else {
    $baseUrl = "https://github.com/$Repo/releases/download/$Version"
}
$url = "$baseUrl/$asset"
$checksumUrl = if ($env:ENKAI_CHECKSUM_URL) { $env:ENKAI_CHECKSUM_URL } elseif ($env:enkai_CHECKSUM_URL) { $env:enkai_CHECKSUM_URL } else { "$baseUrl/$asset.sha256" }

Write-Host "Downloading $asset..."
$tmp = New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString())
$zipPath = Join-Path $tmp $asset
Invoke-WebRequest -Uri $url -OutFile $zipPath

if ($env:ENKAI_SKIP_VERIFY -ne "1" -and $env:enkai_SKIP_VERIFY -ne "1") {
    $checksumPath = Join-Path $tmp "$asset.sha256"
    try {
        Invoke-WebRequest -Uri $checksumUrl -OutFile $checksumPath | Out-Null
        $expected = (Get-Content $checksumPath | Select-Object -First 1).Split(" ")[0]
        $actual = (Get-FileHash -Algorithm SHA256 $zipPath).Hash.ToLower()
        if ($expected.ToLower() -ne $actual) {
            throw "Checksum verification failed"
        }
    } catch {
        Write-Warning "Checksum verification skipped or failed: $($_.Exception.Message)"
    }
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Expand-Archive -Path $zipPath -DestinationPath $tmp -Force

$exe = Join-Path $tmp "enkai.exe"
if (!(Test-Path $exe)) {
    throw "enkai.exe not found in archive"
}

Copy-Item -Force $exe (Join-Path $InstallDir "enkai.exe")
if (Test-Path (Join-Path $tmp "enkai.exe")) {
    Copy-Item -Force (Join-Path $tmp "enkai.exe") (Join-Path $InstallDir "enkai.exe")
}
if (Test-Path (Join-Path $tmp "enkai_native.dll")) {
    Copy-Item -Force (Join-Path $tmp "enkai_native.dll") (Join-Path $InstallDir "enkai_native.dll")
}
if (Test-Path (Join-Path $tmp "enkai_native.dll")) {
    Copy-Item -Force (Join-Path $tmp "enkai_native.dll") (Join-Path $InstallDir "enkai_native.dll")
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath.Split(';') -contains $InstallDir) {
    $newPath = if ([string]::IsNullOrWhiteSpace($userPath)) { $InstallDir } else { "$userPath;$InstallDir" }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Added $InstallDir to PATH (User). Restart your terminal."
}

Write-Host "Enkai installed to $InstallDir\\enkai.exe"
Write-Host "Verify: enkai --version"

