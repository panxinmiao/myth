[English](README.md)

---
<div align="center">

# Myth

**基于 wgpu 的高性能 Rust 渲染引擎。**

[![CI](https://github.com/panxinmiao/myth/actions/workflows/ci.yml/badge.svg)](https://github.com/panxinmiao/myth/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/panxinmiao/myth/actions/workflows/deploy.yml/badge.svg)](https://github.com/panxinmiao/myth/actions/workflows/deploy.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](LICENSE)
[![WebGPU Ready](https://img.shields.io/badge/WebGPU-Ready-green.svg)](https://gpuweb.github.io/gpuweb/)

[![Myth Engine Hero](https://raw.githubusercontent.com/panxinmiao/myth/main/docs/public/images/hero.jpg)](https://panxinmiao.github.io/myth/zh/)

[**主页**](https://panxinmiao.github.io/myth/zh/) | [**在线体验**](https://panxinmiao.github.io/myth/gallery/) | [**示例代码**](examples/)

</div>

---

## 简介

**Myth** 是一个基于 Rust 和 wgpu 构建的高性能、跨平台 3D 渲染引擎。它旨在提供极致的渲染表现与极简的 API 设计，为开发者带来流畅的编程体验。

> 获取完整的快速入门教程、API 指南、进阶渲染特性以及底层架构设计，请访问 **[Myth Engine 主页](https://panxinmiao.github.io/myth/zh/)**。

> 无需配置环境，**[直接在浏览器中体验 Myth 引擎的渲染能力](https://panxinmiao.github.io/myth/gallery/)**。


## 核心特性

* **一处编写，多端运行：** 借助 `wgpu` 的强大能力，Myth 一次编码即可原生运行于 Windows, macOS, Linux，并无缝编译为 WASM 运行于现代浏览器 (WebGPU)。
* **次世代图形管线：**
    * 完整支持基于物理的渲染 (PBR)。
    * 高效的前向集群光照 (Clustered Forward Lighting)。
    * 包含屏幕空间全局光照 (SSGI)、屏幕空间反射 (SSR) 与 SSAO 等高级后处理矩阵。
    * **内置 3D 高斯溅射 (3DGS) 渲染支持**，实现高保真的前沿视图合成。
* **现代渲染图架构：** 核心采用严格的 SSA（静态单赋值）[渲染图架构](https://panxinmiao.github.io/myth/zh/architecture/render-graph)，自动处理复杂的显存复用、管线屏障 (Barriers) 与依赖同步。
* **丰富的生产力生态：** 完整的 glTF 支持，提供异步资产加载，并原生支持 Python 绑定，满足多种开发流需求。

> 查看完整的特性列表，请访问文档的 [Features 页面](https://panxinmiao.github.io/myth/zh/guide/features)。

## 快速开始

### 安装依赖

在你的 Rust 项目中，通过 Cargo 引入 Myth：

```bash
cargo add myth-engine
```

### "Hello World"

用不到 50 行代码，实现一个带有棋盘格纹理的旋转立方体：

```rust
use myth::prelude::*;

struct MyApp;

impl AppHandler for MyApp {
    fn init(engine: &mut Engine, _: &dyn Window) -> Self {
        // 0. 创建场景
        let scene = engine.scene_manager.create_active();

        // 1. 创建带有棋盘格纹理的立方体网格
        let tex_handle = engine.assets.checkerboard(512, 64);
        let mesh_handle = scene.spawn_box(
            1.0, 1.0, 1.0, 
            PhongMaterial::new(Vec4::new(1.0, 0.76, 0.33, 1.0)).with_map(tex_handle),
            &engine.assets,
        );
        // 2. 设置相机
        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene.node(&cam_node_id).set_position(0.0, 0.0, 5.0).look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);
        // 3. 添加光源
        scene.add_light(Light::new_directional(Vec3::ONE, 5.0));

        // 4. 设置更新回调以旋转立方体
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

fn main() -> myth::Result<()> {
    App::new().with_title("Myth-Engine Demo").run::<MyApp>()
}
```

### 运行环境与示例

Myth 提供了丰富的 Example 供开发者参考。无论你是想在本地 Native 环境运行，还是编译到 Web 端，或者使用 Python 绑定，我们都准备了详尽的指南。

> **[查阅完整的 Quick Start 指南](https://panxinmiao.github.io/myth/zh/guide/quick-start)** 获取详细的构建说明。

### Python 绑定

Myth 引擎还提供了 Python 绑定，用于快速原型设计和科学可视化。
有关安装方法和示例，请参阅 [Python Bindings](https://panxinmiao.github.io/myth/zh/guide/python)。


## 开源协议 (License)

本项目采用 **MIT License** 或 **Apache-2.0 License** 双重授权。