param(
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BuildPath = Join-Path $ProjectRoot $BuildDir

if (Test-Path -LiteralPath $BuildPath) {
    Remove-Item -LiteralPath $BuildPath -Recurse -Force
    Write-Host "Removed: $BuildPath"
} else {
    Write-Host "Nothing to clean: $BuildPath"
}

