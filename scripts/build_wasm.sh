#!/bin/bash
# Usage: ./scripts/build_wasm.sh <example_name> [debug|release]

set -e

# 1. Check Environment
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory."
    exit 1
fi

EXAMPLE_NAME=$1

if [ -z "$EXAMPLE_NAME" ]; then
    echo "❌ Error: Example name required."
    echo "Usage: ./scripts/build_wasm.sh <example_name> [debug|release] [additional cargo flags...]"
    exit 1
fi

# remove first argument (example name)
shift

MODE="release"
if [ "$1" == "debug" ]; then
    MODE="debug"
    shift
elif [ "$1" == "release" ]; then
    MODE="release"
    shift
fi

# 2. Define Paths
TARGET_DIR="target/wasm32-unknown-unknown/$MODE/examples"
WEB_ROOT="examples/$EXAMPLE_NAME/web"
OUTPUT_PKG_DIR="$WEB_ROOT/pkg"
OUTPUT_ASSETS_DIR="$WEB_ROOT/assets"
SOURCE_ASSETS_DIR="examples/assets"

echo "🚀 Building example '$EXAMPLE_NAME' in $MODE mode..."

# 3. Cargo Build
BUILD_FLAGS="--target wasm32-unknown-unknown --example $EXAMPLE_NAME"
if [ "$MODE" == "release" ]; then
    BUILD_FLAGS="$BUILD_FLAGS --release"
fi
cargo build $BUILD_FLAGS "$@"

# 4. Generate js Bindings
echo "📦 Generating JS bindings..."
rm -rf "$OUTPUT_PKG_DIR"
wasm-bindgen "$TARGET_DIR/$EXAMPLE_NAME.wasm" \
  --out-dir "$OUTPUT_PKG_DIR" \
  --target web \
  --no-typescript

# Prevent pkg directory from being uploaded
echo "*" > "$OUTPUT_PKG_DIR/.gitignore"

# ==========================================
# 5. Sync Assets
# ==========================================
echo "📂 Syncing assets..."
rm -rf "$OUTPUT_ASSETS_DIR"
mkdir -p "$OUTPUT_ASSETS_DIR"

if [ -d "$SOURCE_ASSETS_DIR" ]; then
    cp -r "$SOURCE_ASSETS_DIR/"* "$OUTPUT_ASSETS_DIR/"
    
    # Prevent copied assets directory from being uploaded
    echo "*" > "$OUTPUT_ASSETS_DIR/.gitignore"
    
    echo "   ✅ Copied shared assets."
else
    echo "   ⚠️ Warning: Shared assets directory '$SOURCE_ASSETS_DIR' not found."
fi
# ==========================================

# 6. Optimize WASM (release mode only)
if [ "$MODE" == "release" ] && command -v wasm-opt &> /dev/null; then
    echo "✨ Optimizing WASM size..."
    wasm-opt -Oz -o "$OUTPUT_PKG_DIR/${EXAMPLE_NAME}_bg.wasm" "$OUTPUT_PKG_DIR/${EXAMPLE_NAME}_bg.wasm"
fi

echo "✅ Build Complete!"
echo "👉 Run: python3 -m http.server 8080 --directory $WEB_ROOT"