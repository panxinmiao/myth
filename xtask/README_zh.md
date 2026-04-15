# Myth xtask - Web Gallery 构建工具

## 环境准备

在运行构建脚本之前，请确保您的开发环境已安装以下工具链：

1. **Rust Wasm 目标**：
   ```bash
   rustup target add wasm32-unknown-unknown
   ```
2. **wasm-bindgen CLI**（用于生成 Wasm 绑定）：
   ```bash
   cargo install wasm-bindgen-cli
   ```
3. **wasm-opt**（可选，来自 Binaryen）：
   如果您需要进行 `Release` 构建，脚本会自动调用 `wasm-opt` 进一步压缩和优化 Wasm 文件体积。

## 使用指南

您可以在 Myth 工作区的任意目录下通过 `cargo xtask` 加上对应的子命令来执行构建任务。

### 核心命令

#### 1. 构建 Web Gallery (示例画廊)
构建所有带有 `[gallery]` 元数据的示例，并将前端页面、Wasm 文件和资源组装到统一的分发目录（默认输出到 `dist/`）：
```bash
cargo xtask build-gallery
```

#### 2. 构建所有 Demo Apps
构建工作区中定义的独立 Demo 应用，并将其 Web 资源和 Wasm 文件打包：
```bash
cargo xtask build-apps
```

#### 3. 构建特定的 Demo App
如果您只需要构建某一个特定的 Demo 应用（例如指定的包名），请在命令后提供该应用的 `id`：
```bash
cargo xtask build-app <app_id>
```

### 可选参数

所有上述命令均支持以下附加标志位，以定制构建行为：

* **构建配置 (Profile)**
  * `--release`（默认）：使用 Release 模式编译，并在构建后自动使用 `wasm-opt` 优化产物。
  * `--debug`：使用 Debug 模式编译，跳过 `wasm-opt`，适合开发和调试。

* **指定构建目标 (--only)**
  在批量构建时，可以使用 `--only` 仅编译特定的项目，从而大幅节省时间。
  ```bash
  # 仅构建名为 "hello_triangle" 的示例
  cargo xtask build-gallery --only hello_triangle
  ```

* **指定特性 (--features)**
  将自定义的 Rust features 传递给底层的 `cargo build`。
  ```bash
  cargo xtask build-gallery --features "some_feature"
  ```

## 构建流程简述

当您执行构建任务时，脚本会自动执行以下步骤：
1. **解析元数据**：扫描 `Cargo.toml` 和示例文件中的元数据，收集分类、描述和特性等信息。
2. **准备目录**：清空并重建分发目录（默认 `dist/`）。
3. **资源同步**：将共享静态资源（如 `assets`）和前端模板（`demo_apps/gallery` 等）同步到分发目录。
4. **Wasm 编译**：调用 `cargo build --target wasm32-unknown-unknown` 进行编译。
5. **生成绑定与优化**：通过 `wasm-bindgen` 生成 JS/Wasm 绑定，如果是 Release 模式还会执行 `wasm-opt` 优化。
6. **生成清单**：最终在 `dist/` 目录下生成用于前端渲染的 `examples.json` 数据清单。


## Gallery 元数据配置指南

`xtask` 通过解析文件中的元数据来决定如何构建和展示示例。要将一个新的示例或应用加入 Gallery，您需要按以下格式添加配置：

### 1. 引擎示例 (Examples)
在 `examples/` 目录下的 `.rs` 文件顶部（前48行内），使用 `//!` 文档注释块编写 TOML 格式的配置：

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
* **name**: 示例在页面上显示的名称。
* **category**: 所属分类（将按此分类进行分组）。
* **order** (可选): 排序权重，数字越小越靠前。
* **web** (可选): 是否支持 WebAssembly 构建（默认为 `true`）。
* **features** (可选): 构建此示例时需要激活的 Cargo features 列表。

### 2. 独立演示应用 (Demo Apps)
对于 `demo_apps/` 目录下的独立应用，元数据需配置在应用的 `Cargo.toml` 中：

```toml
[package.metadata.gallery]
name = "GLTF Viewer"
category = "Tools"
description = "A full-featured GLTF model viewer."
order = 1
web = true

# 可选：如果应用包含多个展示子项 (Showcase)
[[package.metadata.gallery.showcase]]
id = "helmet"
name = "Damaged Helmet"
model = "DamagedHelmet/glTF/DamagedHelmet.gltf"
order = 1
```

### 3. 工作区配置 (Workspace Config)
您还可以在根目录的 `Cargo.toml` 中修改构建输出路径和资源路径的默认行为：
```toml
[workspace.metadata.gallery]
dist_dir = "dist"                  # 最终输出目录
frontend_dir = "demo_apps/gallery" # 前端模板目录
examples_dir = "examples"          # 示例源码目录
shared_assets_dir = "examples/assets" # 共享静态资源
```