param(
    [string]$IsaacPython = "E:\IsaacWork\env_isaaclab\python.exe",
    [ValidateSet("roll", "pitch", "yaw")]
    [string]$Axis = "roll",
    [double]$Rate = 100.0,
    [double]$Duration = 2.0,
    [double]$StepStart = 0.50,
    [double]$StepEnd = 0.90,
    [switch]$Headless,
    [switch]$NoPlot
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path $IsaacPython)) {
    throw "Isaac Lab Python not found: $IsaacPython"
}

$RateTag = if ($Rate -ge 0) { "p$([math]::Round($Rate))" } else { "m$([math]::Abs([math]::Round($Rate)))" }
$LogPath = Join-Path $Root "logs\rate_steps\${Axis}_${RateTag}dps.csv"

$ArgsList = @(
    ".\scripts\run_rate_step.py",
    "--axis", $Axis,
    "--rate-deg-s", $Rate,
    "--duration", $Duration,
    "--step-start", $StepStart,
    "--step-end", $StepEnd,
    "--log-path", $LogPath
)
if ($Headless) { $ArgsList += "--headless" }

Write-Host "Running Q250 $Axis body-rate pulse at $Rate deg/s"
& $IsaacPython @ArgsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $NoPlot) {
    Write-Host "Generating plots..."
    & $IsaacPython .\scripts\plot_rate_step.py --log $LogPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "CSV: $LogPath"
Write-Host "Rate plot: $([IO.Path]::ChangeExtension($LogPath, $null))_rates.png"
Write-Host "Motor plot: $([IO.Path]::ChangeExtension($LogPath, $null))_motors.png"
