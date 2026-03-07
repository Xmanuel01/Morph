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

$releaseChecklist = Read-Text "docs/RELEASE_CHECKLIST.md"
if (-not $releaseChecklist.Contains("scripts/release_pipeline.ps1")) {
    $failures.Add("docs/RELEASE_CHECKLIST.md missing scripts/release_pipeline.ps1")
}
if (-not $releaseChecklist.Contains("scripts/release_pipeline.sh")) {
    $failures.Add("docs/RELEASE_CHECKLIST.md missing scripts/release_pipeline.sh")
}
if (-not $releaseChecklist.Contains("scripts/package_release.py")) {
    $failures.Add("docs/RELEASE_CHECKLIST.md missing scripts/package_release.py")
}
if (-not $releaseChecklist.Contains("scripts/verify_release_artifact.py")) {
    $failures.Add("docs/RELEASE_CHECKLIST.md missing scripts/verify_release_artifact.py")
}
if ($releaseChecklist.Contains("v1.9 consolidated pipeline")) {
    $failures.Add("docs/RELEASE_CHECKLIST.md still references v1.9-specific pipeline wording")
}
if (-not $releaseChecklist.Contains("scripts/rc_pipeline.ps1")) {
    $failures.Add("docs/RELEASE_CHECKLIST.md missing scripts/rc_pipeline.ps1")
}
if (-not $releaseChecklist.Contains("scripts/rc_pipeline.sh")) {
    $failures.Add("docs/RELEASE_CHECKLIST.md missing scripts/rc_pipeline.sh")
}
if (-not $releaseChecklist.Contains("scripts/collect_release_evidence.py")) {
    $failures.Add("docs/RELEASE_CHECKLIST.md missing scripts/collect_release_evidence.py")
}

$versionToken = $version -replace '\.', '_'
$requiredWrappers = @(
    "scripts/v${versionToken}_rc_pipeline.ps1",
    "scripts/v${versionToken}_rc_pipeline.sh"
)
foreach ($wrapper in $requiredWrappers) {
    if (-not (Test-Path (Join-Path $root $wrapper))) {
        $failures.Add("missing RC wrapper for current version: $wrapper")
    }
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

$rcNotesPath = Join-Path $root "docs/31_v2_rc_notes.md"
if (-not (Test-Path $rcNotesPath)) {
    $failures.Add("missing docs/31_v2_rc_notes.md")
} else {
    $rcNotes = Get-Content -Raw -Encoding UTF8 $rcNotesPath
    if (-not $rcNotes.Contains("v2.0.0")) {
        $failures.Add("docs/31_v2_rc_notes.md missing v2.0.0 references")
    }
    if (-not $rcNotes.ToLower().Contains("strict compatibility")) {
        $failures.Add("docs/31_v2_rc_notes.md missing strict compatibility language")
    }
}

$migrationPath = Join-Path $root "docs/32_v2_migration_guide.md"
if (-not (Test-Path $migrationPath)) {
    $failures.Add("missing docs/32_v2_migration_guide.md")
} else {
    $migration = Get-Content -Raw -Encoding UTF8 $migrationPath
    foreach ($token in @(
        "enkai migrate config-v1",
        "enkai migrate checkpoint-meta-v1",
        "enkai doctor"
    )) {
        if (-not $migration.Contains($token)) {
            $failures.Add("docs/32_v2_migration_guide.md missing required command reference: $token")
        }
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
