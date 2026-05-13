$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot

$targets = @(
  (Join-Path $repoRoot 'frontend\release'),
  (Join-Path $repoRoot 'frontend\dist'),
  (Join-Path $repoRoot 'backend\build'),
  (Join-Path $repoRoot 'backend\dist'),
  (Join-Path $repoRoot 'backend\data'),
  (Join-Path $repoRoot 'backend\__pycache__'),
  (Join-Path $repoRoot 'backend\api.spec')
)

Write-Host ''
Write-Host 'ML Studio release cleanup script' -ForegroundColor Cyan
Write-Host "Project root: $repoRoot"
Write-Host ''

$failedTargets = @()

foreach ($target in $targets) {
  $resolved = [System.IO.Path]::GetFullPath($target)
  if (-not $resolved.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove path outside workspace: $resolved"
  }

  if (Test-Path -LiteralPath $resolved) {
    try {
      Remove-Item -LiteralPath $resolved -Recurse -Force
      Write-Host "Removed: $resolved" -ForegroundColor Yellow
    } catch {
      $failedTargets += $resolved
      Write-Warning "Failed to remove: $resolved"
      Write-Warning $_.Exception.Message
    }
  } else {
    Write-Host "Skipped: $resolved" -ForegroundColor DarkGray
  }
}

Write-Host ''
if ($failedTargets.Count -gt 0) {
  Write-Warning 'Cleanup finished with locked or unavailable targets.'
  Write-Host 'Please close ML Studio / Electron and run this script again.' -ForegroundColor Yellow
  Write-Host ''
  Write-Host 'Remaining targets:' -ForegroundColor Yellow
  $failedTargets | ForEach-Object { Write-Host "- $_" }
  exit 1
}

Write-Host 'Cleanup finished.' -ForegroundColor Green
