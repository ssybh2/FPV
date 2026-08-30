param(
    [int]$NumEnvs = 512,
    [int]$MaxIterations = 400,
    [string]$RunName = "",
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe",
    [switch]$Gui
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$argsList = @(".\scripts\train_gate_racing.py", "--num_envs", "$NumEnvs", "--max_iterations", "$MaxIterations")
if ($RunName) { $argsList += @("--run_name", $RunName) }
if (-not $Gui) { $argsList += "--headless" }
& $IsaacPython @argsList
exit $LASTEXITCODE
