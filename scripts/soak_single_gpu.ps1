$ErrorActionPreference = "Stop"

$Config = if ($env:ENKAI_CONFIG) { $env:ENKAI_CONFIG } else { "configs/enkai_50m.enk" }
$MaxWaitSec = if ($env:ENKAI_MAX_WAIT) { [int]$env:ENKAI_MAX_WAIT } else { 28800 }
$FreshRun = if ($env:ENKAI_FRESH_RUN) { $env:ENKAI_FRESH_RUN -ne "0" } else { $true }

function Get-ConfigMode($configPath) {
    if (-not (Test-Path $configPath)) { return "unknown" }
    try {
        $raw = Get-Content -Raw -Path $configPath
        if ($raw -match '"device"\s*:\s*"cpu"' -or
            $raw -match '"backend"\s*:\s*"cpu"' -or
            $raw -match '\\"device\\"\s*:\s*\\"cpu\\"' -or
            $raw -match '\\"backend\\"\s*:\s*\\"cpu\\"') {
            return "cpu"
        }
        if ($raw -match '"device"\s*:\s*"cuda:[0-9]+"' -or
            $raw -match '"backend"\s*:\s*"native"' -or
            $raw -match '\\"device\\"\s*:\s*\\"cuda:[0-9]+\\"' -or
            $raw -match '\\"backend\\"\s*:\s*\\"native\\"') {
            return "gpu"
        }
    } catch { }
    return "unknown"
}

function Get-ConfigCheckpointDir($configPath) {
    if (-not (Test-Path $configPath)) { return $null }
    try {
        $raw = Get-Content -Raw -Path $configPath
        $m = [regex]::Match($raw, '"checkpoint_dir"\s*:\s*"([^"]+)"')
        if ($m.Success) { return $m.Groups[1].Value }
        $m2 = [regex]::Match($raw, '\\"checkpoint_dir\\"\s*:\s*\\"([^\\"]+)\\"')
        if ($m2.Success) { return $m2.Groups[1].Value }
    } catch { }
    return $null
}

$ConfigMode = Get-ConfigMode -configPath $Config
$KillStep = if ($env:ENKAI_KILL_STEP) {
    [int]$env:ENKAI_KILL_STEP
} elseif ($ConfigMode -eq "cpu") {
    20
} else {
    2000
}
$PostResumeSteps = if ($env:ENKAI_POST_RESUME_STEPS) {
    [int]$env:ENKAI_POST_RESUME_STEPS
} elseif ($ConfigMode -eq "cpu") {
    10
} else {
    500
}

if ($ConfigMode -eq "cpu") {
    Write-Host "Single-device soak (CPU mode)"
} else {
    Write-Host "Single-GPU soak on a 4x4090 machine (uses cuda:0 only)"
}
Write-Host "Using config: $Config"
Write-Host "Kill step: $KillStep"
Write-Host "Post-resume steps: $PostResumeSteps"

function Resolve-EnkaiExe {
    $cmd = Get-Command "enkai" -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $localExe = Join-Path $repoRoot "target\debug\enkai.exe"
    if (Test-Path $localExe) {
        return $localExe
    }
    throw "Could not find enkai executable. Build it first with: cargo build -p enkai"
}

function Resolve-EnkaiTensorLib {
    if ($env:ENKAI_TENSOR_PATH -and (Test-Path $env:ENKAI_TENSOR_PATH)) {
        return $env:ENKAI_TENSOR_PATH
    }
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $candidates = @(
        (Join-Path $repoRoot "target\debug\deps\enkai_tensor.dll"),
        (Join-Path $repoRoot "target\debug\enkai_tensor.dll")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "Could not find enkai_tensor.dll. Build it first with: cargo build -p enkai_tensor"
}

function Resolve-TorchLibDir {
    if ($env:TORCH_LIB -and (Test-Path $env:TORCH_LIB)) {
        return $env:TORCH_LIB
    }
    try {
        $py = Get-Command "python" -ErrorAction SilentlyContinue
        if ($py) {
            $out = & python -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $path = $out.Trim()
                if (Test-Path $path) { return $path }
            }
        }
    } catch { }
    try {
        $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
        if ($pyLauncher) {
            $out = & py -3.10 -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $path = $out.Trim()
                if (Test-Path $path) { return $path }
            }
        }
    } catch { }
    return $null
}

function Get-Latest-StepFromLog($logPath) {
    if (-not (Test-Path $logPath)) { return 0 }
    $lines = Get-Content $logPath
    $last = $lines | Select-Object -Last 1
    if (-not $last) { return 0 }
    try {
        $obj = $last | ConvertFrom-Json
        return [int]$obj.step
    } catch {
        return 0
    }
}

function Assert-LogHealth($logPath, $allowResumeFrom = 0) {
    if (-not (Test-Path $logPath)) { throw "Log file not found: $logPath" }
    $prev = 0
    $seenResumeReset = $false
    foreach ($line in Get-Content $logPath) {
        if (-not $line) { continue }
        $obj = $line | ConvertFrom-Json
        $step = [int]$obj.step
        $loss = [double]$obj.loss
        if ([double]::IsNaN($loss) -or [double]::IsInfinity($loss)) {
            throw "Non-finite loss at step $step"
        }
        if ($step -lt $prev) {
            if ($allowResumeFrom -gt 0 -and -not $seenResumeReset -and $step -le $allowResumeFrom) {
                $seenResumeReset = $true
            } else {
                throw "Step went backwards: $step < $prev"
            }
        }
        $prev = $step
    }
}

function Read-Tail($path, $lines = 40) {
    if (-not (Test-Path $path)) { return "" }
    return ((Get-Content $path -Tail $lines) -join "`n")
}

function Get-LatestCheckpointDir($dirPath) {
    if (-not (Test-Path $dirPath)) { return $null }
    $stepDirs = Get-ChildItem -Path $dirPath -Directory -Filter "step_*" -ErrorAction SilentlyContinue
    if (-not $stepDirs -or $stepDirs.Count -eq 0) { return $null }
    return ($stepDirs | Sort-Object Name | Select-Object -Last 1).FullName
}

function Wait-UntilStep($logPath, $targetStep, $timeoutSec, $proc, $stdoutPath, $stderrPath) {
    $start = Get-Date
    while ($true) {
        $step = Get-Latest-StepFromLog $logPath
        if ($step -ge $targetStep) { return $step }
        if ($proc.HasExited) {
            $outTail = Read-Tail -path $stdoutPath -lines 40
            $errTail = Read-Tail -path $stderrPath -lines 40
            throw "Training process exited early with code $($proc.ExitCode) before reaching step $targetStep (last=$step).`nstdout tail:`n$outTail`nstderr tail:`n$errTail"
        }
        Start-Sleep -Seconds 5
        if (((Get-Date) - $start).TotalSeconds -gt $timeoutSec) {
            $outTail = Read-Tail -path $stdoutPath -lines 40
            $errTail = Read-Tail -path $stderrPath -lines 40
            throw "Timeout waiting for step $targetStep (last=$step).`nstdout tail:`n$outTail`nstderr tail:`n$errTail"
        }
    }
}

$checkpointDir = if ($env:ENKAI_CHECKPOINT_DIR) {
    $env:ENKAI_CHECKPOINT_DIR
} else {
    $cfgCheckpoint = Get-ConfigCheckpointDir -configPath $Config
    if ($cfgCheckpoint) { $cfgCheckpoint } else { "checkpoints/enkai_50m" }
}
New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null
$logPath = Join-Path $checkpointDir "train_log.jsonl"
$integrityScript = Join-Path $PSScriptRoot "check_ckpt_integrity.ps1"
$stdoutPath = Join-Path $checkpointDir "soak_train_stdout.log"
$stderrPath = Join-Path $checkpointDir "soak_train_stderr.log"

$pass = $true
$resumedFrom = 0
$lastLoss = $null
$failureReason = ""

if ($FreshRun -and (Test-Path $checkpointDir)) {
    Get-ChildItem -Path $checkpointDir -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer -and $_.Name -like "step_*") {
            Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue
        } elseif (-not $_.PSIsContainer -and $_.Name -in @("train_log.jsonl", "soak_train_stdout.log", "soak_train_stderr.log", "soak_resume_stdout.log", "soak_resume_stderr.log")) {
            Remove-Item -Force $_.FullName -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "Starting training..."
$enkaiExe = Resolve-EnkaiExe
$enkaiTensorLib = Resolve-EnkaiTensorLib
$env:ENKAI_TENSOR_PATH = $enkaiTensorLib
$torchLibDir = Resolve-TorchLibDir
if ($torchLibDir) {
    $env:Path = "$torchLibDir;$env:Path"
    Write-Host "Using torch runtime lib dir: $torchLibDir"
} else {
    Write-Host "Torch runtime lib dir not auto-detected; if load fails, set TORCH_LIB env var."
}
Write-Host "Using enkai executable: $enkaiExe"
Write-Host "Using enkai tensor library: $enkaiTensorLib"
if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force }
if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force }
$proc = Start-Process -FilePath $enkaiExe -ArgumentList @("train", $Config) -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

try {
    $step = Wait-UntilStep -logPath $logPath -targetStep $KillStep -timeoutSec $MaxWaitSec -proc $proc -stdoutPath $stdoutPath -stderrPath $stderrPath
    Write-Host "Reached step $step, killing process to simulate crash..."
    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force
    }
    $resumedFrom = $step
} catch {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    $pass = $false
    $failureReason = $_.Exception.Message
}

if ($pass) {
    Write-Host "Restarting training..."
    $stdoutPath2 = Join-Path $checkpointDir "soak_resume_stdout.log"
    $stderrPath2 = Join-Path $checkpointDir "soak_resume_stderr.log"
    if (Test-Path $stdoutPath2) { Remove-Item $stdoutPath2 -Force }
    if (Test-Path $stderrPath2) { Remove-Item $stderrPath2 -Force }
    $proc2 = Start-Process -FilePath $enkaiExe -ArgumentList @("train", $Config) -PassThru -RedirectStandardOutput $stdoutPath2 -RedirectStandardError $stderrPath2

    try {
        $resumeTarget = $KillStep + $PostResumeSteps
        $step2 = Wait-UntilStep -logPath $logPath -targetStep $resumeTarget -timeoutSec $MaxWaitSec -proc $proc2 -stdoutPath $stdoutPath2 -stderrPath $stderrPath2
        Write-Host "Resume confirmed at step $step2"
        Assert-LogHealth -logPath $logPath -allowResumeFrom $resumedFrom
        $lastLine = Get-Content $logPath | Select-Object -Last 1
        if ($lastLine) {
            $obj = $lastLine | ConvertFrom-Json
            $lastLoss = [double]$obj.loss
        }
    } catch {
        if (-not $proc2.HasExited) { Stop-Process -Id $proc2.Id -Force }
        $pass = $false
        if ([string]::IsNullOrEmpty($failureReason)) { $failureReason = $_.Exception.Message }
    }
}

if ($pass -and (Test-Path $integrityScript)) {
    try {
        $env:ENKAI_CHECKPOINT_DIR = $checkpointDir
        $env:ENKAI_RESUMED_FROM_STEP = "$resumedFrom"
        & $integrityScript | Out-Host
    } catch {
        $pass = $false
        if ([string]::IsNullOrEmpty($failureReason)) { $failureReason = $_.Exception.Message }
    }
}

$latestCkpt = Get-LatestCheckpointDir -dirPath $checkpointDir
$ckptOk = $null -ne $latestCkpt
$nanDetected = $false
if ($lastLoss -ne $null) {
    $nanDetected = [double]::IsNaN($lastLoss) -or [double]::IsInfinity($lastLoss)
}

Write-Host ""
Write-Host "=== SINGLE-GPU SOAK SUMMARY ==="
$status = if ($pass -and -not $nanDetected -and $ckptOk) { "PASS" } else { "FAIL" }
Write-Host ("status: {0}" -f $status)
Write-Host ("last_step: {0}" -f (Get-Latest-StepFromLog $logPath))
Write-Host ("last_loss: {0}" -f $lastLoss)
Write-Host ("resumed_from_step: {0}" -f $resumedFrom)
Write-Host ("nan_or_inf: {0}" -f $nanDetected)
Write-Host ("checkpoint_verified: {0}" -f $ckptOk)
if ($latestCkpt) { Write-Host ("checkpoint_path: {0}" -f $latestCkpt) }
if (-not [string]::IsNullOrEmpty($failureReason)) {
    Write-Host ("failure_reason: {0}" -f $failureReason)
}
Write-Host "================================"

if ($status -ne "PASS") { exit 1 }
Write-Host "Soak test OK"
