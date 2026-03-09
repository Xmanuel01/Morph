param(
    [switch]$VerifyGpuEvidence,
    [string]$GpuLogDir = "artifacts/gpu",
    [switch]$SkipPackageCheck
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:CARGO_BUILD_JOBS)) {
    # Default to serialized rustc jobs in release gates for deterministic memory usage.
    $env:CARGO_BUILD_JOBS = "1"
}
if ([string]::IsNullOrWhiteSpace($env:CARGO_INCREMENTAL)) {
    $env:CARGO_INCREMENTAL = "0"
}
$stripDebug = $env:ENKAI_PIPELINE_STRIP_DEBUG
if ([string]::IsNullOrWhiteSpace($stripDebug)) {
    $stripDebug = "0"
}

function Invoke-Gate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "[release] Gate failed: $Name (exit code $LASTEXITCODE)."
    }
}

function Get-EnkaiVersion {
    $cargoPath = Join-Path $PSScriptRoot "..\enkai\Cargo.toml"
    $content = Get-Content -Raw -Encoding UTF8 $cargoPath
    $match = [regex]::Match($content, '^version\s*=\s*"([^"]+)"', [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if (-not $match.Success) {
        throw "failed to parse version from enkai/Cargo.toml"
    }
    return $match.Groups[1].Value
}

function Get-PythonInvocation {
    foreach ($candidate in @("python3", "python", "py")) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            continue
        }
        if ($candidate -eq "py") {
            return @("py", "-3")
        }
        return @($candidate)
    }
    throw "Python interpreter not found (tried python3, python, py)."
}

function Invoke-Python {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    $python = Get-PythonInvocation
    $cmd = $python[0]
    $prefix = @()
    if ($python.Count -gt 1) {
        $prefix = $python[1..($python.Count - 1)]
    }
    & $cmd @prefix @Args
}

Write-Host "[release] Running format gate..."
Invoke-Gate -Name "format" -Command { cargo fmt --all --check }

Write-Host "[release] Running clippy gate..."
Invoke-Gate -Name "clippy" -Command { cargo clippy --workspace --all-targets -- -D warnings }

Write-Host "[release] Running test gate..."
Invoke-Gate -Name "tests" -Command {
    $previousRustflags = $env:RUSTFLAGS
    if ($stripDebug -eq "1") {
        if ([string]::IsNullOrWhiteSpace($previousRustflags)) {
            $env:RUSTFLAGS = "-C debuginfo=0"
        } else {
            $env:RUSTFLAGS = "$previousRustflags -C debuginfo=0"
        }
    }
    try {
        cargo test --workspace -j 1
    } finally {
        $env:RUSTFLAGS = $previousRustflags
    }
}

Write-Host "[release] Running docs contract consistency gate..."
Invoke-Gate -Name "docs-consistency" -Command {
    powershell -ExecutionPolicy Bypass -File scripts/check_docs_consistency.ps1
}

Write-Host "[release] Running serve/frontend contract snapshot gate..."
Invoke-Gate -Name "contract-snapshots" -Command {
    cargo test -p enkai --bin enkai frontend::tests::contract_snapshots_match_reference_files
}

Write-Host "[release] Running self-host mainline gate..."
Invoke-Gate -Name "selfhost-mainline" -Command {
    New-Item -ItemType Directory -Path artifacts/selfhost -Force | Out-Null
    cargo run -p enkai -- litec mainline-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost
}

Write-Host "[release] Running self-host Stage0 fallback gate..."
Invoke-Gate -Name "selfhost-stage0-fallback" -Command {
    cargo run -p enkai -- litec selfhost-ci enkai/tools/bootstrap/selfhost_corpus
}

Write-Host "[release] Running dependency license audit gate..."
Invoke-Gate -Name "license-audit" -Command {
    Invoke-Python scripts/license_audit.py
}

if (-not $SkipPackageCheck) {
    $version = Get-EnkaiVersion

    Write-Host "[release] Building release binaries for package gate..."
    Invoke-Gate -Name "build-release-enkai" -Command { cargo build -p enkai --release }
    Invoke-Gate -Name "build-release-native" -Command { cargo build -p enkai_native --release }

    Write-Host "[release] Building deterministic package and checksum..."
    Invoke-Gate -Name "package-release" -Command {
        Invoke-Python scripts/package_release.py `
            --version $version `
            --target-os windows `
            --arch x86_64 `
            --bin target/release/enkai.exe `
            --native target/release/enkai_native.dll `
            --check-deterministic
    }

    $archive = "dist/enkai-$version-windows-x86_64.zip"
    Write-Host "[release] Verifying package checksum/layout/smoke..."
    Invoke-Gate -Name "verify-release-artifact" -Command {
        Invoke-Python scripts/verify_release_artifact.py `
            --archive $archive `
            --target-os windows `
            --smoke
    }

    Write-Host "[release] Generating SBOM artifact..."
    Invoke-Gate -Name "generate-sbom" -Command {
        Invoke-Python scripts/generate_sbom.py --output "dist/sbom-$version-windows-x86_64.json"
    }
}

if ($VerifyGpuEvidence) {
    Write-Host "[release] Verifying GPU gate evidence..."
    Invoke-Gate -Name "verify-gpu-evidence" -Command {
        powershell -ExecutionPolicy Bypass -File scripts/verify_gpu_gates.ps1 -LogDir $GpuLogDir
    }
}

Write-Host "[release] Release pipeline gates passed."
