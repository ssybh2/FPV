param(
    [string]$Checkpoint = ".\checkpoints\gate_racing\model_399.pt",
    [int]$NumEnvs = 16,
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
& $IsaacPython .\scripts\verify_v04_transfer.py --checkpoint $Checkpoint --num_envs $NumEnvs --headless
exit $LASTEXITCODE
