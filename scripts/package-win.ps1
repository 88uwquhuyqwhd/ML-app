$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $repoRoot 'frontend'
$backendExe = Join-Path $repoRoot 'backend\\dist\\api\\api.exe'
$releaseDir = Join-Path $frontendDir 'release'
$portableExe = Join-Path $releaseDir 'win-unpacked\\ML Studio.exe'

function Assert-LastExitCode {
  param(
    [string]$StepName
  )

  if ($LASTEXITCODE -ne 0) {
    throw "$StepName failed with exit code $LASTEXITCODE."
  }
}

Write-Host ''
Write-Host 'ML Studio Windows packaging script' -ForegroundColor Cyan
Write-Host "Project root: $repoRoot"
Write-Host ''

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
  Write-Warning 'Not running as Administrator. If you hit EPERM or Access is denied, reopen PowerShell as Administrator and run this script again.'
}

Push-Location $repoRoot

try {
  Write-Host '1/3 Build backend...' -ForegroundColor Yellow
  npm run build:backend
  Assert-LastExitCode 'Build backend'

  if (-not (Test-Path $backendExe)) {
    throw "Backend executable not found: $backendExe"
  }

  Write-Host ''
  Write-Host '2/3 Build frontend...' -ForegroundColor Yellow
  Push-Location $frontendDir
  npm run build
  Assert-LastExitCode 'Build frontend'

  Write-Host ''
  Write-Host '3/3 Create Windows installer...' -ForegroundColor Yellow
  npx electron-builder --win nsis --publish never
  Assert-LastExitCode 'Create Windows installer'
  Pop-Location

  Write-Host ''
  Write-Host 'Packaging finished.' -ForegroundColor Green
  Write-Host "Installer output: $releaseDir"
}
catch {
  if (Test-Path $portableExe) {
    Write-Warning "Installer step failed, but the unpacked app is available: $portableExe"
  }
  throw
}
finally {
  if ((Get-Location).Path -eq $frontendDir) {
    Pop-Location
  }
  Pop-Location
}
