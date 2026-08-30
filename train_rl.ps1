param(
    [int]$NumEnvs = 512,
    [int]$MaxIterations = 300,
    [string]$RunName = "",
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe",
    [switch]$Gui
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$argsList = @(".\scripts\train_fly_to_point.py", "--num_envs", "$NumEnvs", "--max_iterations", "$MaxIterations")
if ($RunName) { $argsList += @("--run_name", $RunName) }
if (-not $Gui) { $argsList += "--headless" }
& $IsaacPython @argsList
exit $LASTEXITCODE
