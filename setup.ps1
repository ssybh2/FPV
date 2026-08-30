param(
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path $IsaacPython)) {
    throw "Isaac Lab Python not found: $IsaacPython`nPass it explicitly: .\setup.ps1 -IsaacPython 'E:\path\to\python.exe'"
}

Write-Host "[1/4] Checking Isaac Lab + RSL-RL Python"
& $IsaacPython .\scripts\check_environment.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/4] Installing workspace in editable mode"
& $IsaacPython -m pip install -e . --no-build-isolation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/4] Running pure dynamics/control/RL math tests"
& $IsaacPython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/4] Printing Q250 model summary"
& $IsaacPython .\scripts\print_model_summary.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Setup complete. Fastest next steps:"
Write-Host "  .\smoke_rl.ps1"
Write-Host "  .\train_rl.ps1 -NumEnvs 512 -MaxIterations 300"
Write-Host "  .\tensorboard.ps1"
Write-Host "  .\play_rl.ps1"
