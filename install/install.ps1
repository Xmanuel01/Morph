param(
    [string]$Repo = "Xmanuel01/Enkai",
    [string]$Version = "latest"
)

$ErrorActionPreference = "Stop"

if ($Version -eq "latest") {
    $api = "https://api.github.com/repos/$Repo/releases/latest"
} else {
    $api = "https://api.github.com/repos/$Repo/releases/tags/$Version"
}

$release = Invoke-RestMethod -Uri $api -Headers @{ "User-Agent" = "enkai-installer" }
$tag = $release.tag_name
if (-not $tag) { throw "Failed to determine release tag" }
$ver = $tag.TrimStart('v')
$asset = "enkai-$ver-windows-x86_64.zip"
$assetObj = $release.assets | Where-Object { $_.name -eq $asset } | Select-Object -First 1
if (-not $assetObj) { throw "Asset not found: $asset" }

$temp = Join-Path $env:TEMP ("enkai-install-" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Force -Path $temp | Out-Null
$zip = Join-Path $temp $asset
Invoke-WebRequest -Uri $assetObj.browser_download_url -OutFile $zip

$extract = Join-Path $temp "extract"
Expand-Archive -Path $zip -DestinationPath $extract -Force

$installDir = Join-Path $env:USERPROFILE ".enkai\\bin"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $installDir "std")
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $installDir "examples")

Copy-Item -Path (Join-Path $extract "enkai.exe") -Destination (Join-Path $installDir "enkai.exe") -Force
if (Test-Path (Join-Path $extract "enkai.exe")) {
    Copy-Item -Path (Join-Path $extract "enkai.exe") -Destination (Join-Path $installDir "enkai.exe") -Force
}
if (Test-Path (Join-Path $extract "enkai_native.dll")) {
    Copy-Item -Path (Join-Path $extract "enkai_native.dll") -Destination $installDir -Force
}
if (Test-Path (Join-Path $extract "enkai_native.dll")) {
    Copy-Item -Path (Join-Path $extract "enkai_native.dll") -Destination $installDir -Force
}
if (Test-Path (Join-Path $extract "std")) {
    Copy-Item -Recurse -Path (Join-Path $extract "std") -Destination (Join-Path $installDir "std") -Force
}
if (Test-Path (Join-Path $extract "examples")) {
    Copy-Item -Recurse -Path (Join-Path $extract "examples") -Destination (Join-Path $installDir "examples") -Force
}
if (Test-Path (Join-Path $extract "README.txt")) {
    Copy-Item -Path (Join-Path $extract "README.txt") -Destination (Join-Path $installDir "README.txt") -Force
}

$env:PATH = "$installDir;$env:PATH"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }
if ($userPath -notmatch [Regex]::Escape($installDir)) {
    [Environment]::SetEnvironmentVariable("Path", "$installDir;$userPath", "User")
    Write-Host "Added $installDir to PATH. Restart your shell."
}

& (Join-Path $installDir "enkai.exe") --version
Write-Host "Installed Enkai to $installDir"

