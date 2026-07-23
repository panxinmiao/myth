# 快速开始

在本节中，我们将使用不到 50 行代码，在屏幕上渲染出一个带有棋盘格纹理、不断旋转的 3D 立方体。

## 1. 安装与依赖

首先，创建一个新的 Rust 项目并添加 Myth 引擎依赖。在你的 `Cargo.toml` 中加入：

```toml
[dependencies]
# 从 crates.io 获取稳定版本
myth-engine = "0.2.0"

# 或从 GitHub 获取最新主分支
# myth-engine = { git = "https://github.com/panxinmiao/myth", branch = "main" }
```

::: info Feature Flags (特性开关)
Myth 采用模块化设计，许多重型能力默认隐藏在 Feature Flag 之后以保持极致轻量。在实际开发中，你可能需要根据需求启用以下特性：

* `gltf`: 加载 glTF 和 GLB 格式的模型资产
* `3dgs`: 启用 3D 高斯溅射（Gaussian Splatting）渲染支持
* `gaussian-npz`: 支持加载压缩的 NPZ 格式高斯点云
* `debug_view`: 开启调试视图和渲染图检查器辅助工具
:::

## 2. 你的第一个 App (Hello World)

Myth 使用了极其直观的 API 设计。创建 `src/main.rs` 并填入以下代码：

```rust
use myth::prelude::*;

struct MyApp;

impl AppHandler for MyApp {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        // 1. 创建并激活场景
        let scene = engine.scene_manager.create_active();

        // 2. 创建带有棋盘格纹理的立方体
        let tex_handle = engine.assets.checkerboard(512, 64);
        let mesh_handle = scene.spawn_box(
            1.0, 1.0, 1.0, 
            PhongMaterial::new(Vec4::new(1.0, 0.76, 0.33, 1.0)).with_map(tex_handle),
            &engine.assets,
        );

        // 3. 设置相机与视口
        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene.node(&cam_node_id).set_position(0.0, 0.0, 5.0).look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);

        // 4. 添加环境光源
        scene.add_light(Light::new_directional(Vec3::ONE, 5.0));

        // 5. 注册每帧更新回调逻辑
        scene.on_update(move |scene, _input, _dt| {
            if let Some(node) = scene.get_node_mut(mesh_handle) {
                let rot_y = Quat::from_rotation_y(0.02);
                let rot_x = Quat::from_rotation_x(0.01);
                node.transform.rotation = node.transform.rotation * rot_y * rot_x;
            }
        });

        Self {}
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Myth-Engine Hello World")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            ..Default::default()
        })
        .run::<MyApp>()
}

```

## 3. 运行你的程序

使用 Cargo 运行项目，体验你的第一个 Myth 3D 场景：

```bash
cargo run --release

```

::: tip 渲染路径 (Render Path) 选择
我们在 `main` 函数中指定了 `RenderPath::HighFidelity`。如果你的场景只需要非常基础的前向渲染，且运行在低端设备上，可以切换为 `RenderPath::BasicForward`。但如果你需要 PBR、泛光(Bloom)、SSAO 或 3DGS 特性，请始终保持在 `HighFidelity` 高保真管线下。
:::

## 4. 运行官方示例

仓库内置了 50+ 个覆盖各类特性的示例。克隆仓库后可直接运行：

```bash
# 运行单个示例（如地球 Demo）
cargo run --example earth --release

# 运行独立 App（如 glTF Viewer）
cargo run -p gltf_viewer --release
```

关于 Web/WASM 示例的构建，请参考 [myth xtask 指南](https://github.com/panxinmiao/myth/blob/main/xtask/README.md)。

## 下一步

- 理解引擎的四层心智模型 → [场景与节点系统](/zh/guide/scene-graph)
- 加载模型并播放动画 → [资产、glTF 与动画](/zh/guide/assets-animation)
- 纵览全部能力 → [核心特性总览](/zh/guide/features)