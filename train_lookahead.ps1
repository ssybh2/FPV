param(
    [int]$NumEnvs = 512,
    [int]$MaxIterations = 450,
    [string]$Checkpoint = ".\checkpoints\gate_racing\model_399.pt",
    [string]$RunName = "transfer399",
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe",
    [switch]$Gui
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$argsList = @(".\scripts\train_lookahead_racing.py", "--num_envs", "$NumEnvs", "--max_iterations", "$MaxIterations", "--transfer_checkpoint", "$Checkpoint")
if ($RunName) { $argsList += @("--run_name", $RunName) }
if (-not $Gui) { $argsList += "--headless" }
& $IsaacPython @argsList
exit $LASTEXITCODE
