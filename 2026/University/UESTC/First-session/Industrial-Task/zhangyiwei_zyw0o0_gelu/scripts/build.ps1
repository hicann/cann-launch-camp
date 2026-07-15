param(
    [string]$BuildDir = "build",
    [string]$BuildType = "Release",
    [string]$CannPath = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BuildPath = Join-Path $ProjectRoot $BuildDir

if ($CannPath -ne "") {
    $env:ASCEND_HOME_PATH = $CannPath
    $env:ASCEND_CANN_PACKAGE_PATH = $CannPath
}

if (-not $env:ASCEND_HOME_PATH -and -not $env:ASCEND_CANN_PACKAGE_PATH) {
    Write-Error "CANN environment is not configured. Run set_env first or pass -CannPath."
}

$CMakeArgs = @("-S", $ProjectRoot, "-B", $BuildPath, "-DCMAKE_BUILD_TYPE=$BuildType")
if ($CannPath -ne "") {
    $CMakeArgs += "-DCMAKE_PREFIX_PATH=$CannPath"
}

cmake @CMakeArgs
cmake --build $BuildPath --config $BuildType

Write-Host "Build finished: $BuildPath"
