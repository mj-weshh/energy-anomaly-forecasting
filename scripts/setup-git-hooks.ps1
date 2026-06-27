# Enable project git hooks for this repository clone.
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    git config core.hooksPath .githooks
    Write-Host "Git hooks enabled: core.hooksPath = .githooks"
} finally {
    Pop-Location
}
