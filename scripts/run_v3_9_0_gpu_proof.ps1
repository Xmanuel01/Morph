param(
    [string]$Python = "py -3.11",
    [switch]$InstallPyTorchCuda,
    [string]$TorchVersion = "2.2.0+cu121",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121",
    [switch]$BuildFirstPartyCudaKernels,
    [switch]$RunRocmSourceBuild,
    [switch]$RunMetalSourceBuild,
    [switch]$RunDistributedGpuProof,
    [switch]$RunFourGpuSoak,
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if ($InstallPyTorchCuda) {
    & (Join-Path $PSScriptRoot "setup_pytorch_cuda.ps1") -Python $Python -TorchVersion $TorchVersion -TorchIndexUrl $TorchIndexUrl
}

$pyArgs = $Python -split " "
$pyExe = $pyArgs[0]
$rest = @()
if ($pyArgs.Count -gt 1) { $rest = $pyArgs[1..($pyArgs.Count - 1)] }

if (-not $SkipPreflight) {
    $preflightArgs = @("scripts\preflight_v3_9_0_gpu_test.py", "--workspace", ".", "--python", $Python)
    if ($BuildFirstPartyCudaKernels) {
        $preflightArgs += "--require-nvcc"
    }
    if ($RunDistributedGpuProof) {
        $preflightArgs += "--require-two-gpus"
    }
    if ($RunFourGpuSoak) {
        $preflightArgs += "--require-four-gpus"
    }
    & $pyExe @rest @preflightArgs
}

$features = "torch"
if ($BuildFirstPartyCudaKernels) { $features = "$features,cuda-kernels" }
if ($RunRocmSourceBuild) { $features = "$features,rocm-kernels" }
if ($RunMetalSourceBuild) { $features = "$features,metal-kernels" }

cargo build -p enkai_tensor --features $features
cargo test -p enkai_tensor --features $features --test cuda_kernel_manifest
cargo test -p enkai_tensor --features $features --test cuda_llm_foundation -- --nocapture
cargo build -p enkai
& $pyExe @rest scripts\readiness_v3_9_0_cuda_llm_runtime_foundation.py --workspace . --python $Python
& $pyExe @rest scripts\verify_v3_9_0_cuda_llm_runtime_foundation.py --workspace .

if ($RunDistributedGpuProof) {
    $distFeatures = "$features,dist"
    cargo build -p enkai_tensor --features $distFeatures
    $env:ENKAI_ENABLE_DIST = "1"
    $env:ENKAI_RUN_MULTI_GPU_TESTS = "1"
    $env:ENKAI_SINGLE_GPU_GREEN = "1"
    $distArgs = @("scripts\readiness_v3_9_0_distributed_gpu_execution.py", "--workspace", ".", "--python", $Python, "--run")
    if ($RunFourGpuSoak) {
        $distArgs += "--run-soak4"
    }
    & $pyExe @rest @distArgs
    & $pyExe @rest scripts\verify_v3_9_0_distributed_gpu_execution.py --workspace .
}
