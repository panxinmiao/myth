@echo off
setlocal enabledelayedexpansion

if not exist "Cargo.toml" (
    echo Error: please run this script from the workspace root.
    exit /b 1
)

set TARGET_NAME=%1
if "%TARGET_NAME%"=="" (
    echo Usage: build_wasm.bat ^<target_name^> [debug^|release]
    exit /b 1
)

shift
set MODE=--release
if /I "%~1"=="debug" (
    set MODE=--debug
    shift
) else if /I "%~1"=="release" (
    shift
)

if not "%~1"=="" (
    echo Warning: extra cargo flags are no longer supported by this compatibility wrapper and will be ignored.
)

if exist "demo_apps\%TARGET_NAME%" (
    cargo xtask build-app %TARGET_NAME% %MODE%
) else if exist "examples\%TARGET_NAME%.rs" (
    cargo xtask build-gallery %MODE% --only %TARGET_NAME%
) else (
    echo Error: '%TARGET_NAME%' is neither a curated example nor an app package.
    exit /b 1
)

if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo Build complete. Dist output is available in .\dist
