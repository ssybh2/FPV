param(
    [int]$NumEnvs = 32,
    [double]$Duration = 2.0,
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
& $IsaacPython .\scripts\smoke_fly_to_point.py --num_envs $NumEnvs --duration $Duration --headless
exit $LASTEXITCODE
