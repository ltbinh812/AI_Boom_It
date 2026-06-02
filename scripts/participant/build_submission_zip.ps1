# Build a flat submission.zip for Bomberland (server requires agent.py at zip root).
# Usage (from repo root): .\scripts\participant\build_submission_zip.ps1

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$MyAgent = Join-Path $Root "my_agent"
$Required = @("agent.py", "model.py", "model.pth")
foreach ($name in $Required) {
    $path = Join-Path $MyAgent $name
    if (-not (Test-Path $path)) {
        Write-Error "Missing required file: $path"
    }
}

$Staging = Join-Path $Root "temp_submission"
$ZipPath = Join-Path $Root "submission.zip"

if (Test-Path $Staging) { Remove-Item $Staging -Recurse -Force }
New-Item -ItemType Directory -Path $Staging | Out-Null

foreach ($name in $Required) {
    Copy-Item (Join-Path $MyAgent $name) (Join-Path $Staging $name) -Force
}

if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path (Join-Path $Staging "*") -DestinationPath $ZipPath -Force
Remove-Item $Staging -Recurse -Force

Write-Host "Created $ZipPath (flat: agent.py, model.py, model.pth at root)"
