param(
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe",
    [int]$Port = 6006
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
& $IsaacPython -m tensorboard.main --logdir .\logs\rsl_rl\q250_fly_to_point --port $Port
exit $LASTEXITCODE
