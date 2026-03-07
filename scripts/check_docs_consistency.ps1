$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot

function Read-Text([string]$path) {
    Get-Content -Raw -Encoding UTF8 (Join-Path $root $path)
}

$cargoToml = Read-Text "enkai/Cargo.toml"
$versionMatch = [regex]::Match($cargoToml, '^version\s*=\s*"([^"]+)"', [System.Text.RegularExpressions.RegexOptions]::Multiline)
if (-not $versionMatch.Success) {
    throw "failed to parse version from enkai/Cargo.toml"
}
$version = $versionMatch.Groups[1].Value
$tag = "v$version"
$failures = New-Object System.Collections.Generic.List[string]

$mainRs = Read-Text "enkai/src/main.rs"
if ($mainRs.Contains('const LANG_VERSION: &') -and -not $mainRs.Contains('env!("ENKAI_LANG_VERSION")')) {
    $failures.Add("enkai/src/main.rs still hardcodes LANG_VERSION")
}

$readme = Read-Text "README.md"
if (-not $readme.Contains("Status ($tag)")) {
    $failures.Add("README.md missing Status ($tag)")
}
if ($readme.Contains("Distributed stubs:")) {
    $failures.Add("README.md contains outdated distributed stubs claim")
}

$docsReadme = Read-Text "docs/README.md"
if (-not $docsReadme.Contains($tag)) {
    $failures.Add("docs/README.md missing release tag $tag")
}

$spec = Read-Text "docs/Enkai.spec"
if (-not $spec.Contains("v0.1 -> $tag")) {
    $failures.Add("docs/Enkai.spec title is out of sync with crate version")
}
if (-not $spec.Contains("Known Limits in $tag")) {
    $failures.Add("docs/Enkai.spec known limits header is out of sync")
}
if ($spec.Contains("compile to stub functions")) {
    $failures.Add("docs/Enkai.spec still claims tool declarations compile to stubs")
}

$validation = Read-Text "VALIDATION.md"
if (-not $validation.Contains("Validation Matrix")) {
    $failures.Add("VALIDATION.md title should use release-line validation matrix wording")
}

$frontendDocs = Read-Text "docs/27_frontend_stack.md"
if (-not $frontendDocs.Contains("backend_api.snapshot.json")) {
    $failures.Add("docs/27_frontend_stack.md missing backend snapshot reference")
}
if (-not $frontendDocs.Contains("sdk_api.snapshot.json")) {
    $failures.Add("docs/27_frontend_stack.md missing SDK snapshot reference")
}

$snapshotFiles = @(
    "enkai/contracts/backend_api_v1.snapshot.json",
    "enkai/contracts/sdk_api_v1.snapshot.json",
    "enkai/contracts/conversation_state_v1.schema.json"
)
foreach ($snapshot in $snapshotFiles) {
    if (-not (Test-Path (Join-Path $root $snapshot))) {
        $failures.Add("missing contract snapshot file: $snapshot")
    }
}

if ($failures.Count -gt 0) {
    Write-Host "docs consistency check failed:"
    foreach ($failure in $failures) {
        Write-Host "- $failure"
    }
    exit 1
}

Write-Host "docs consistency check passed for $tag"
