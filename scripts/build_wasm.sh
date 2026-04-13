#!/bin/bash

set -euo pipefail

if [ ! -f "Cargo.toml" ]; then
    echo "Error: please run this script from the workspace root."
    exit 1
fi

TARGET_NAME=${1:-}
if [ -z "$TARGET_NAME" ]; then
    echo "Usage: ./scripts/build_wasm.sh <target_name> [debug|release]"
    exit 1
fi

shift
MODE="--release"
if [ "${1:-}" = "debug" ]; then
    MODE="--debug"
    shift
elif [ "${1:-}" = "release" ]; then
    shift
fi

if [ "$#" -gt 0 ]; then
    echo "Warning: extra cargo flags are no longer supported by this compatibility wrapper and will be ignored."
fi

if [ -d "demo_apps/$TARGET_NAME" ]; then
    cargo xtask build-app "$TARGET_NAME" "$MODE"
elif [ -f "examples/$TARGET_NAME.rs" ]; then
    cargo xtask build-gallery "$MODE" --only "$TARGET_NAME"
else
    echo "Error: '$TARGET_NAME' is neither a curated example nor an app package."
    exit 1
fi

echo "Build complete. Dist output is available in ./dist"
echo "👉 Run: python -m http.server 8080 --directory ./dist"