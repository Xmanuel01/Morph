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

function Assert-MinFreeSpace {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [double]$RequiredGiB
    )

    $resolved = Resolve-Path -LiteralPath $Path
    $drive = Get-PSDrive -Name $resolved.Drive.Name -ErrorAction Stop
    $requiredBytes = [math]::Floor($RequiredGiB * 1GB)
    if ($drive.Free -lt $requiredBytes) {
        $freeGiB = [math]::Round($drive.Free / 1GB, 2)
        throw "[release] insufficient free disk space on drive $($drive.Name): ${freeGiB} GiB available, ${RequiredGiB} GiB required. Free space or set ENKAI_RELEASE_MIN_FREE_GB to a lower validated threshold."
    }
}

if ([string]::IsNullOrWhiteSpace($env:ENKAI_RELEASE_MIN_FREE_GB)) {
    $env:ENKAI_RELEASE_MIN_FREE_GB = "4"
}
$minFreeGiB = [double]::Parse(
    $env:ENKAI_RELEASE_MIN_FREE_GB,
    [System.Globalization.CultureInfo]::InvariantCulture
)
Assert-MinFreeSpace -Path "." -RequiredGiB $minFreeGiB

Write-Host "[release] Running consolidated production readiness gate..."
Invoke-Gate -Name "readiness-check" -Command {
    New-Item -ItemType Directory -Path artifacts/selfhost -Force | Out-Null
    New-Item -ItemType Directory -Path artifacts/readiness -Force | Out-Null
    cargo run -p enkai -- readiness check --profile full_platform --json --output artifacts/readiness/full_platform.json --skip-check selfhost-mainline --skip-check selfhost-stage0-fallback
}

Write-Host "[release] Running bootstrap release lane gate..."
Invoke-Gate -Name "selfhost-release-ci" -Command {
    cargo run -p enkai -- litec release-ci enkai/tools/bootstrap/selfhost_corpus --triage-dir artifacts/selfhost
}

Write-Host "[release] Running dependency license audit gate..."
Invoke-Gate -Name "license-audit" -Command {
    Invoke-Python scripts/license_audit.py
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

    Write-Host "[release] Bootstrapping blocker verification artifact before evidence archive..."
    $preBlockerArgs = @(
        "run", "-p", "enkai", "--",
        "readiness", "verify-blockers",
        "--profile", "full_platform",
        "--report", "artifacts/readiness/full_platform.json",
        "--json",
        "--output", "artifacts/readiness/full_platform_blockers.json",
        "--skip-release-evidence",
        "--allow-skipped-required-check", "selfhost-mainline",
        "--allow-skipped-required-check", "selfhost-stage0-fallback"
    )
    Invoke-Gate -Name "verify-release-blockers-bootstrap" -Command {
        cargo @preBlockerArgs
    }

    Write-Host "[release] Collecting strict release evidence bundle..."
    Invoke-Gate -Name "collect-release-evidence" -Command {
        Invoke-Python scripts/collect_release_evidence.py --strict
    }

    Write-Host "[release] Verifying release blocker matrix against archived evidence..."
    $finalBlockerArgs = @(
        "run", "-p", "enkai", "--",
        "readiness", "verify-blockers",
        "--profile", "full_platform",
        "--report", "artifacts/readiness/full_platform.json",
        "--json",
        "--output", "artifacts/readiness/full_platform_blockers.json",
        "--allow-skipped-required-check", "selfhost-mainline",
        "--allow-skipped-required-check", "selfhost-stage0-fallback"
    )
    Invoke-Gate -Name "verify-release-blockers" -Command {
        cargo @finalBlockerArgs
    }

    Write-Host "[release] Refreshing strict release evidence bundle with final blocker report..."
    Invoke-Gate -Name "refresh-release-evidence" -Command {
        Invoke-Python scripts/collect_release_evidence.py --strict
    }

    Write-Host "[release] Generating strict capability report..."
    Invoke-Gate -Name "generate-capability-report" -Command {
        Invoke-Python scripts/generate_capability_report.py --strict
    }

    Write-Host "[release] Generating release dashboard..."
    Invoke-Gate -Name "generate-release-dashboard" -Command {
        Invoke-Python scripts/generate_release_dashboard.py --strict
    }
}
else {
    Write-Host "[release] Generating blocker verification artifact for reduced evidence mode..."
    $reducedBlockerArgs = @(
        "run", "-p", "enkai", "--",
        "readiness", "verify-blockers",
        "--profile", "full_platform",
        "--report", "artifacts/readiness/full_platform.json",
        "--json",
        "--output", "artifacts/readiness/full_platform_blockers.json",
        "--skip-release-evidence",
        "--allow-skipped-required-check", "selfhost-mainline",
        "--allow-skipped-required-check", "selfhost-stage0-fallback"
    )
    Invoke-Gate -Name "verify-release-blockers-reduced" -Command {
        cargo @reducedBlockerArgs
    }

    Write-Host "[release] Collecting reduced release evidence bundle (package checks skipped)..."
    Invoke-Gate -Name "collect-release-evidence" -Command {
        Invoke-Python scripts/collect_release_evidence.py
    }

    Write-Host "[release] Generating reduced capability report (package checks skipped)..."
    Invoke-Gate -Name "generate-capability-report" -Command {
        Invoke-Python scripts/generate_capability_report.py
    }

    Write-Host "[release] Generating reduced release dashboard (package checks skipped)..."
    Invoke-Gate -Name "generate-release-dashboard" -Command {
        Invoke-Python scripts/generate_release_dashboard.py
    }
}

if ($VerifyGpuEvidence) {
    Write-Host "[release] Verifying GPU gate evidence..."
    Invoke-Gate -Name "verify-gpu-evidence" -Command {
        powershell -ExecutionPolicy Bypass -File scripts/verify_gpu_gates.ps1 -LogDir $GpuLogDir
    }
}

Write-Host "[release] Release pipeline gates passed."
