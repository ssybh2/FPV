param(
    [string]$Checkpoint = "",
    [int]$NumEnvs = 1,
    [double]$Duration = 30.0,
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$argsList = @(".\scripts\play_fly_to_point.py", "--num_envs", "$NumEnvs", "--duration", "$Duration")
if ($Checkpoint) { $argsList += @("--checkpoint", $Checkpoint) }
& $IsaacPython @argsList
exit $LASTEXITCODE
