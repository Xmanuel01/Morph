param(
    [string]$Python = "py -3.11",
    [string]$TorchVersion = "2.2.0+cu121",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121",
    [switch]$RunCargoTorchTests
)

$ErrorActionPreference = "Stop"

$parts = $Python -split " "
$pythonExe = $parts[0]
$pythonArgs = @()
if ($parts.Count -gt 1) {
    $pythonArgs = $parts[1..($parts.Count - 1)]
}

& $pythonExe @pythonArgs -m pip install --upgrade pip
& $pythonExe @pythonArgs -m pip install "torch==$TorchVersion" --index-url $TorchIndexUrl

$resolvedPython = & $pythonExe @pythonArgs -c "import sys; print(sys.executable)"
$torchLib = & $pythonExe @pythonArgs -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
$torchStatus = & $pythonExe @pythonArgs -c "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_count': torch.cuda.device_count(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"

if (-not (Test-Path $torchLib)) {
    throw "Torch lib directory not found: $torchLib"
}

$env:LIBTORCH_USE_PYTORCH = "1"
$env:PYTHON_SYS_EXECUTABLE = $resolvedPython
$env:PATH = "$torchLib;$env:PATH"

Write-Output $torchStatus

if ($RunCargoTorchTests) {
    cargo test -p enkai_tensor --features torch --test cuda_llm_foundation -- --nocapture
}
