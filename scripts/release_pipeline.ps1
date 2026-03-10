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

function Resolve-BenchPython {
    if (-not [string]::IsNullOrWhiteSpace($env:ENKAI_BENCH_PYTHON)) {
        return $env:ENKAI_BENCH_PYTHON
    }
    try {
        $candidate = (py -3.11 -c "import sys; print(sys.executable)" 2>$null).Trim()
        if (-not [string]::IsNullOrWhiteSpace($candidate)) {
            return $candidate
        }
    }
    catch {
    }
    foreach ($name in @("python3.11", "python3", "python")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $cmd) {
            return $cmd.Source
        }
    }
    throw "Unable to resolve a Python interpreter for benchmark lane. Set ENKAI_BENCH_PYTHON."
}

Write-Host "[release] Running consolidated production readiness gate..."
Invoke-Gate -Name "readiness-check" -Command {
    New-Item -ItemType Directory -Path artifacts/selfhost -Force | Out-Null
    New-Item -ItemType Directory -Path artifacts/readiness -Force | Out-Null
    cargo run -p enkai -- readiness check --profile production --json --output artifacts/readiness/production.json
}

Write-Host "[release] Running bootstrap release lane gate..."
Invoke-Gate -Name "selfhost-release-ci" -Command {
    cargo run -p enkai -- litec release-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost
}

Write-Host "[release] Running dependency license audit gate..."
Invoke-Gate -Name "license-audit" -Command {
    Invoke-Python scripts/license_audit.py
}

Write-Host "[release] Building release binaries for benchmark/package gates..."
Invoke-Gate -Name "build-release-enkai" -Command { cargo build -p enkai --release }
Invoke-Gate -Name "build-release-native" -Command { cargo build -p enkai_native --release }

Write-Host "[release] Running benchmark target gate..."
Invoke-Gate -Name "benchmark-target" -Command {
    $benchPython = Resolve-BenchPython
    New-Item -ItemType Directory -Path dist -Force | Out-Null
    cargo run -p enkai --release -- bench run `
        --suite official_v2_3_0_matrix `
        --baseline python `
        --iterations 2 `
        --warmup 1 `
        --machine-profile bench/machines/windows_ref.json `
        --output dist/benchmark_official_v2_3_0_matrix_windows.json `
        --target-speedup 15 `
        --target-memory 5 `
        --enforce-target `
        --enforce-class-targets `
        --class-targets bench/suites/official_v2_3_0_targets.json `
        --python $benchPython
}

if (-not $SkipPackageCheck) {
    $version = Get-EnkaiVersion

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
