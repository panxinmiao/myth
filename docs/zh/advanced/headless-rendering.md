# 离屏与无头渲染

Myth 可以在**完全没有窗口**的情况下运行，将渲染结果直接读回 CPU 内存。这使引擎天然适合 **CI/CD 自动化测试、云端渲染、服务端缩略图生成与离线视频合成**等场景。

## 1. 无头初始化

与基于 `App` 的窗口化流程不同，无头渲染直接操作 `Engine`，通过 `init_headless` 在指定分辨率下初始化 GPU（无需 surface）：

```rust
use myth::prelude::*;

fn main() {
    let mut engine = Engine::default();

    let (width, height) = (800, 600);
    // 在无头模式下初始化 GPU —— 没有窗口，没有 surface
    pollster::block_on(engine.init_headless(width, height, None))
        .expect("headless init failed");

    // …构建场景（见下文）
}
```

## 2. 构建场景

无头模式下，场景的构建方式与窗口化完全一致：

```rust
let scene = engine.scene_manager.create_active();

// 棋盘格材质的立方体
let image = engine.assets.images.add(Image::checkerboard(512, 512, 64));
let tex = engine.assets.textures.add(Texture::new_2d(Some("checker"), image));
let _cube = scene.spawn_box(2.0, 2.0, 2.0, UnlitMaterial::new(Vec4::ONE).with_map(tex), &engine.assets);

// 相机
let cam = scene.add_camera(Camera::new_perspective(45.0, width as f32 / height as f32, 0.1));
scene.node(&cam).set_position(0.0, 3.0, 8.0).look_at(Vec3::ZERO);
scene.active_camera = Some(cam);

// 方向光
let light = scene.add_light(Light::new_directional(Vec3::ONE, 5.0));
scene.node(&light).set_position(5.0, 10.0, 5.0).look_at(Vec3::ZERO);
```

## 3. 渲染并读回像素

手动推进一帧、渲染当前活跃场景，然后将帧缓冲读回 CPU：

```rust
// 推进一帧逻辑（dt 以秒为单位）
engine.update(0.016);

// 渲染当前活跃场景
let rendered = engine.render_active_scene();
assert!(rendered, "render_active_scene returned false");

// 读回像素（RGBA8）
let pixels = engine.readback_pixels().expect("readback failed");

// 用 image crate 保存为 PNG
image::save_buffer(
    "output.png",
    &pixels,
    width,
    height,
    image::ColorType::Rgba8,
).expect("failed to save output.png");
```

::: tip 完整示例
仓库中的 [`examples/headless_export.rs`](https://github.com/panxinmiao/myth/blob/main/examples/headless_export.rs) 提供了可直接运行的完整示例：
```bash
cargo run --example headless_export --no-default-features
```
:::

## 4. 高吞吐回读流

对于需要**连续导出大量帧**（如离线视频生成）的场景，引擎内置了**非阻塞的异步 GPU→CPU 回读管线**：

- 采用**环形缓冲 (ring-buffer)** 架构，多帧回读可流水线化进行，避免 GPU 因等待 CPU 读取而停顿。
- 内置**自动背压 (back-pressure)** 机制，当 CPU 消费速度跟不上时自动调节，防止内存无限增长。

这套设计让 Myth 能以接近实时的吞吐率持续产出帧序列，非常适合服务端批量渲染与视频管线。

## 适用场景

- **CI / 回归测试：** 在无 GPU 显示器的服务器上渲染参考图并做像素级比对。
- **云渲染 / 缩略图服务：** 按请求生成 3D 模型的预览图。
- **离线视频生成：** 以固定步长逐帧渲染并编码成视频。

## 下一步

- 了解一帧的完整合成流程 → [渲染路径与帧合成](/zh/architecture/rendering-pipeline)
- 理解底层资源调度 → [Render Graph 渲染图](/zh/architecture/render-graph)
