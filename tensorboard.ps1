param(
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe",
    [int]$Port = 6006,
    [string]$Experiment = "q250_gate_racing"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$LogDir = Join-Path ".\logs\rsl_rl" $Experiment
Write-Host "TensorBoard logdir: $LogDir"
& $IsaacPython -m tensorboard.main --logdir $LogDir --port $Port
exit $LASTEXITCODE
