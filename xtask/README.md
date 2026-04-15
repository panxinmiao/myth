# Myth xtask - Web Gallery Build Tool

[中文](README_zh.md)

## Prerequisites

Before running the build scripts, please ensure you have the following toolchains installed in your development environment:

1. **Rust Wasm Target**:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```
2. **wasm-bindgen CLI** (for generating JS/Wasm bindings):
   ```bash
   cargo install wasm-bindgen-cli
   ```
3. **wasm-opt** (Optional, from Binaryen):
   If you are making a `Release` build, the script will automatically invoke `wasm-opt` to further compress and optimize the Wasm file size.

## Usage Guide

You can run these build tasks from anywhere within the Myth workspace using `cargo xtask` followed by the appropriate subcommand.

### Core Commands

#### 1. Build Web Gallery (Examples)
Builds all examples containing the `[gallery]` metadata block. It assembles the frontend UI, Wasm binaries, and assets into a unified distribution directory (defaults to `dist/`):
```bash
cargo xtask build-gallery
```

#### 2. Build All Demo Apps
Builds all standalone demo applications defined in the workspace and packages their web resources and Wasm files:
```bash
cargo xtask build-apps
```

#### 3. Build a Specific Demo App
If you only need to build a single demo application, provide its `id` (or package name) after the command:
```bash
cargo xtask build-app <app_id>
```

### Optional Flags

All of the commands above support the following flags to customize the build behavior:

* **Build Profile**
  * `--release` (Default): Compiles in Release mode and automatically optimizes the output using `wasm-opt`.
  * `--debug`: Compiles in Debug mode and skips `wasm-opt`. Recommended for faster iterations during development.

* **Target Filtering (--only)**
  When batch building, you can use `--only` to compile a specific item, which significantly reduces build time.
  ```bash
  # Only build the "hello_triangle" example
  cargo xtask build-gallery --only hello_triangle
  ```

* **Custom Features (--features)**
  Pass custom Rust features down to the underlying `cargo build` command.
  ```bash
  cargo xtask build-gallery --features "some_feature"
  ```

## Build Pipeline Overview

When you execute a build task, the script automatically performs the following steps:
1. **Metadata Parsing**: Scans `Cargo.toml` and source files for `[gallery]` metadata to collect categories, descriptions, and features.
2. **Directory Prep**: Clears and creates the distribution directory (`dist/` by default).
3. **Asset Sync**: Copies shared static assets (like `assets/`) and frontend templates (`demo_apps/gallery/`) to the output directory.
4. **Wasm Compilation**: Invokes `cargo build --target wasm32-unknown-unknown` to compile the targets.
5. **Binding & Optimization**: Runs `wasm-bindgen` to generate JavaScript bindings. If in release mode, it runs `wasm-opt` for size optimization.
6. **Manifest Generation**: Generates an `examples.json` manifest in the `dist/` folder, which the frontend uses to render the gallery UI dynamically.

## Gallery Metadata Configuration Guide

The `xtask` script relies on metadata to discover, categorize, and build items for the gallery. To add a new example or demo app to the gallery, you need to provide the appropriate metadata.

### 1. Engine Examples
For single-file examples located in the `examples/` directory, add a TOML block using `//!` module doc comments at the top of the `.rs` file (within the first 48 lines):

```rust
//! [gallery]
//! name = "Hello Triangle"
//! category = "Basics"
//! description = "A basic wgpu triangle example."
//! order = 10
//! web = true
//! features = ["some_optional_feature"]

fn main() {
    // ...
}
```
* **name**: The display name in the gallery.
* **category**: The group under which this example will be listed.
* **order** (Optional): Sorting priority (lower numbers appear first).
* **web** (Optional): Whether it supports WebAssembly target (defaults to `true`).
* **features** (Optional): A list of Cargo features required to build this example.

### 2. Demo Apps
For standalone crates in the `demo_apps/` directory, define the metadata in the app's `Cargo.toml`:

```toml
[package.metadata.gallery]
name = "GLTF Viewer"
category = "Tools"
description = "A full-featured GLTF model viewer."
order = 1
web = true

# Optional: If the app features multiple specific showcase items
[[package.metadata.gallery.showcase]]
id = "helmet"
name = "Damaged Helmet"
model = "DamagedHelmet/glTF/DamagedHelmet.gltf"
order = 1
```

### 3. Workspace Configuration
You can customize the directory structures by adding the following block to the root `Cargo.toml`:
```toml
[workspace.metadata.gallery]
dist_dir = "dist"                  # Output directory
frontend_dir = "demo_apps/gallery" # Frontend UI template directory
examples_dir = "examples"          # Path to examples
shared_assets_dir = "examples/assets" # Path to shared assets
```