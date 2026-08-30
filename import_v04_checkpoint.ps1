param(
    [string]$Source = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$DestDir = Join-Path $Root "checkpoints\gate_racing"
$Dest = Join-Path $DestDir "model_399.pt"
New-Item -ItemType Directory -Force $DestDir | Out-Null

$candidates = @()
if ($Source) { $candidates += $Source }
$candidates += @(
    "E:\IsaacWork\Q250_UZH_Racing_v0.4.0\checkpoints\gate_racing\model_399.pt",
    "E:\IsaacWork\Q250_UZH_Racing_v0.4.0\logs\rsl_rl\q250_gate_racing\2026-08-30_14-43-08\model_399.pt"
)

$Found = $null
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) { $Found = (Resolve-Path $c).Path; break }
}
if (-not $Found) {
    throw "Could not find model_399.pt. Pass -Source <full path to model_399.pt>."
}
Copy-Item $Found $Dest -Force
Write-Host "[OK] Imported v0.4 checkpoint:" -ForegroundColor Green
Write-Host "     $Found"
Write-Host "  -> $Dest"
Get-Item $Dest | Format-List FullName,Length,LastWriteTime
