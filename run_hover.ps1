param(
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe",
    [switch]$Headless,
    [switch]$ColdStart
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$ArgsList = @(".\scripts\run_hover.py")
if ($Headless) { $ArgsList += "--headless" }
if ($ColdStart) { $ArgsList += "--cold-start" }
& $IsaacPython @ArgsList
