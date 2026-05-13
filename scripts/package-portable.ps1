$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $repoRoot 'frontend'
$backendExe = Join-Path $repoRoot 'backend\\dist\\api\\api.exe'
$portableZip = Join-Path $frontendDir 'release\\ML-Studio-Portable-1.0.0.zip'

function Assert-LastExitCode {
  param(
    [string]$StepName
  )

  if ($LASTEXITCODE -ne 0) {
    throw "$StepName failed with exit code $LASTEXITCODE."
  }
}

Write-Host ''
Write-Host 'ML Studio portable packaging script' -ForegroundColor Cyan
Write-Host "Project root: $repoRoot"
Write-Host ''

Push-Location $repoRoot

try {
  Write-Host '1/4 Build backend...' -ForegroundColor Yellow
  npm run build:backend
  Assert-LastExitCode 'Build backend'

  if (-not (Test-Path $backendExe)) {
    throw "Backend executable not found: $backendExe"
  }

  Write-Host ''
  Write-Host '2/4 Build frontend...' -ForegroundColor Yellow
  Push-Location $frontendDir
  npm run build
  Assert-LastExitCode 'Build frontend'

  Write-Host ''
  Write-Host '3/4 Create unpacked app...' -ForegroundColor Yellow
  npx electron-builder --win dir --publish never
  Assert-LastExitCode 'Create unpacked app'
  Pop-Location

  Write-Host ''
  Write-Host '4/4 Create portable zip...' -ForegroundColor Yellow
  powershell -ExecutionPolicy Bypass -File (Join-Path $repoRoot 'scripts\\make-portable.ps1')
  Assert-LastExitCode 'Create portable zip'

  Write-Host ''
  Write-Host 'Portable packaging finished.' -ForegroundColor Green
  Write-Host "Portable zip: $portableZip"
}
finally {
  if ((Get-Location).Path -eq $frontendDir) {
    Pop-Location
  }
  Pop-Location
}
