$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $repoRoot 'frontend'
$releaseDir = Join-Path $frontendDir 'release'
$unpackedDir = Join-Path $releaseDir 'win-unpacked'
$portableZip = Join-Path $releaseDir 'ML-Studio-Portable-1.0.0.zip'
$appExe = Join-Path $unpackedDir 'ML Studio.exe'

if (-not (Test-Path $appExe)) {
  throw "Portable source not found: $appExe"
}

if (Test-Path $portableZip) {
  Remove-Item -LiteralPath $portableZip -Force
}

Compress-Archive -Path (Join-Path $unpackedDir '*') -DestinationPath $portableZip -Force

Write-Host ''
Write-Host 'Portable package created.' -ForegroundColor Green
Write-Host "Portable zip: $portableZip"
Write-Host "App exe: $appExe"
