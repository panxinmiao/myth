@echo off
REM Build script for WASM target (Windows)
REM Usage: build_wasm.bat [--debug]

setlocal enabledelayedexpansion

REM Check if wasm-bindgen is installed
where wasm-bindgen >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: wasm-bindgen-cli is not installed
    echo Install it with: cargo install wasm-bindgen-cli
    exit /b 1
)

REM Parse arguments
set BUILD_PROFILE=release
set CARGO_FLAGS=--release
if "%1"=="--debug" (
    set BUILD_PROFILE=debug
    set CARGO_FLAGS=
    echo Building in DEBUG mode
) else (
    echo Building in RELEASE mode
)

REM Get script directory and move to project root
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%\..\..\..
set PROJECT_ROOT=%cd%

echo Project root: %PROJECT_ROOT%

REM Create output directory
if not exist "%SCRIPT_DIR%pkg" mkdir "%SCRIPT_DIR%pkg"

REM Step 1: Build the WASM binary with Cargo
echo.
echo Step 1: Building WASM binary with Cargo...
cargo build --example gltf_viewer --target wasm32-unknown-unknown %CARGO_FLAGS%
if %ERRORLEVEL% neq 0 (
    echo Cargo build failed!
    exit /b 1
)

REM Step 2: Generate JS bindings with wasm-bindgen
echo.
echo Step 2: Generating JS bindings with wasm-bindgen...
set WASM_FILE=%PROJECT_ROOT%\target\wasm32-unknown-unknown\%BUILD_PROFILE%\examples\gltf_viewer.wasm

wasm-bindgen ^
    --out-dir "%SCRIPT_DIR%pkg" ^
    --target web ^
    --no-typescript ^
    "%WASM_FILE%"

if %ERRORLEVEL% neq 0 (
    echo wasm-bindgen failed!
    exit /b 1
)

REM Step 3: Optional - Optimize with wasm-opt (if available)
where wasm-opt >nul 2>nul
if %ERRORLEVEL% equ 0 (
    if "%BUILD_PROFILE%"=="release" (
        echo.
        echo Step 3: Optimizing WASM with wasm-opt...
        wasm-opt -Oz -o "%SCRIPT_DIR%pkg\gltf_viewer_bg.wasm" "%SCRIPT_DIR%pkg\gltf_viewer_bg.wasm"
    )
)

echo.
echo ========================================
echo Build complete!
echo ========================================
echo.
echo To run the example:
echo   cd %SCRIPT_DIR%
echo   python -m http.server 8080
echo   # Then open http://localhost:8080 in your browser
echo.
echo Files generated in: %SCRIPT_DIR%pkg\
dir /b "%SCRIPT_DIR%pkg"
