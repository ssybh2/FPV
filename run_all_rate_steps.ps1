param(
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe",
    [double]$Rate = 100.0,
    [switch]$Headless,
    [switch]$NoPlot
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

foreach ($Axis in @("roll", "pitch", "yaw")) {
    & .\run_rate_step.ps1 `
        -IsaacPython $IsaacPython `
        -Axis $Axis `
        -Rate $Rate `
        -Headless:$Headless `
        -NoPlot:$NoPlot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
