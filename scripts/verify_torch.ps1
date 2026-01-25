param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cpu",
    [string]$TorchVersion = "2.2.0+cpu",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python -m pip install --upgrade pip
& $Python -m pip install "torch==$TorchVersion" --index-url $TorchIndexUrl

$env:LIBTORCH_USE_PYTORCH = "1"
$env:PYTHON_SYS_EXECUTABLE = $Python
$torchLib = & $Python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
if (-not (Test-Path $torchLib)) {
    throw \"Torch lib directory not found: $torchLib\"
}
$env:PATH = "$torchLib;$env:PATH"

cargo test -p morph_tensor --features torch
