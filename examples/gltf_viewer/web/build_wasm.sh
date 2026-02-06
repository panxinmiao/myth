#!/bin/bash
# Build script for WASM target
# Usage: ./build_wasm.sh [--debug]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo -e "${RED}Error: wasm-pack is not installed${NC}"
    echo "Install it with: cargo install wasm-pack"
    exit 1
fi

# Parse arguments
BUILD_MODE="--release"
if [ "$1" == "--debug" ]; then
    BUILD_MODE="--dev"
    echo -e "${YELLOW}Building in DEBUG mode${NC}"
else
    echo -e "${GREEN}Building in RELEASE mode${NC}"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Project root: $PROJECT_ROOT"

# Build the WASM package
echo -e "${GREEN}Building WASM package...${NC}"
cd "$PROJECT_ROOT"

wasm-pack build \
    --target web \
    $BUILD_MODE \
    --out-dir "examples/gltf_viewer/web/pkg" \
    --out-name gltf_viewer \
    -- --example gltf_viewer

echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "To run the example:"
echo "  cd examples/gltf_viewer/web"
echo "  python -m http.server 8080"
echo "  # Then open http://localhost:8080 in your browser"
