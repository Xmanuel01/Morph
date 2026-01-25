Param(
    [Parameter(Mandatory = $true)][string]$Version,
    [switch]$SkipTests,
    [switch]$Push,
    [switch]$AllowDirty
)

function Usage {
    Write-Host "Usage:"
    Write-Host "  scripts\\release.ps1 -Version vX.Y.Z [-SkipTests] [-Push] [-AllowDirty]"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  scripts\\release.ps1 -Version v0.8.0"
    Write-Host "  scripts\\release.ps1 -Version v0.8.0 -SkipTests"
    Write-Host "  scripts\\release.ps1 -Version v0.8.0 -Push"
}

if ($Version -notmatch '^v\d+\.\d+\.\d+$') {
    Write-Error "Version must be in vX.Y.Z format (e.g., v0.8.0)"
    Usage
    exit 1
}

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "main") {
    Write-Warning "You are on branch '$branch' (expected 'main')."
}

if (-not $AllowDirty) {
    $status = (git status --porcelain).Trim()
    if ($status.Length -gt 0) {
        Write-Error "Working tree is dirty. Commit or stash changes (or use -AllowDirty)."
        exit 1
    }
}

if (-not $SkipTests) {
    Write-Host "Running format + clippy + tests..."
    cargo fmt
    cargo clippy -- -D warnings
    cargo test
}

git rev-parse $Version | Out-Null 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Error "Tag $Version already exists."
    exit 1
}

$tagMsg = "Release $Version"
git tag -a $Version -m $tagMsg
Write-Host "Created tag $Version"

if ($Push) {
    git push origin $branch
    git push origin $Version
    Write-Host "Pushed $branch and $Version to origin."
} else {
    Write-Host "To push:"
    Write-Host "  git push origin $branch"
    Write-Host "  git push origin $Version"
}
