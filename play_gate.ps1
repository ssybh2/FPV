param(
    [string]$Checkpoint = "",
    [int]$NumEnvs = 1,
    [double]$Duration = 0.0,
    [ValidateSet(0,1,2)][int]$Stage = 2,
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe",
    [switch]$RealTime
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$argsList = @(".\scripts\play_gate_racing.py", "--num_envs", "$NumEnvs", "--duration", "$Duration", "--stage", "$Stage")
if ($Checkpoint) { $argsList += @("--checkpoint", $Checkpoint) }
if ($RealTime) { $argsList += "--real_time" }
& $IsaacPython @argsList
exit $LASTEXITCODE
