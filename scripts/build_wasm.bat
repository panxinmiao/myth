@echo off
setlocal enabledelayedexpansion

REM 1. Check Environment
where cargo >nul 2>nul
if not exist "Cargo.toml" (
    echo Error: Please run this script from the project root directory.
    exit /b 1
)

set EXAMPLE_NAME=%1
if "%EXAMPLE_NAME%"=="" (
    echo Error: Example name required.
    exit /b 1
)

set MODE=%2
if "%MODE%"=="" set MODE=release

echo Building example '%EXAMPLE_NAME%' in %MODE% mode...

REM 2. Define Paths
set WASM_PATH=target\wasm32-unknown-unknown\%MODE%\examples\%EXAMPLE_NAME%.wasm
set WEB_ROOT=examples\%EXAMPLE_NAME%\web
set OUTPUT_PKG_DIR=%WEB_ROOT%\pkg
set OUTPUT_ASSETS_DIR=%WEB_ROOT%\assets
set SOURCE_ASSETS_DIR=examples\assets

REM 3. Cargo Build
set CARGO_FLAGS=--target wasm32-unknown-unknown --example %EXAMPLE_NAME%
if "%MODE%"=="release" set CARGO_FLAGS=%CARGO_FLAGS% --release

cargo build %CARGO_FLAGS%
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

REM 4. Generate js Bindings
if not exist "%OUTPUT_PKG_DIR%" mkdir "%OUTPUT_PKG_DIR%"
wasm-bindgen "%WASM_PATH%" --out-dir "%OUTPUT_PKG_DIR%" --target web --no-typescript
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

REM Prevent pkg directory from being uploaded
echo * > "%OUTPUT_PKG_DIR%\.gitignore"

REM ==========================================
REM 5. Sync Assets
REM ==========================================
echo Syncing assets...
if exist "%SOURCE_ASSETS_DIR%" (
    if not exist "%OUTPUT_ASSETS_DIR%" mkdir "%OUTPUT_ASSETS_DIR%"
    xcopy "%SOURCE_ASSETS_DIR%" "%OUTPUT_ASSETS_DIR%" /E /I /Y /Q
    
    REM Prevent copied assets directory from being uploaded
    echo * > "%OUTPUT_ASSETS_DIR%\.gitignore"
    
    echo    Done syncing assets.
) else (
    echo    Warning: Shared assets directory not found.
)
REM ==========================================

REM 6. Optimize WASM (release mode only)
if "%MODE%"=="release" (
    where wasm-opt >nul 2>nul && (
        echo Optimizing WASM size...
        wasm-opt -Oz -o "%OUTPUT_PKG_DIR%\%EXAMPLE_NAME%_bg.wasm" "%OUTPUT_PKG_DIR%\%EXAMPLE_NAME%_bg.wasm"
    ) || (
        echo [INFO] wasm-opt not found. Skipping optimization.
    )
)

echo.
echo Build Complete!
echo Run: python -m http.server 8080 --directory %WEB_ROOT%