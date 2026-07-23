# 简介与愿景

**Myth Engine** 是一款对开发者友好的、基于 **Rust** 编写的高性能 3D 渲染引擎。

它深受 Three.js 简洁且符合直觉的 API 设计启发，并基于现代图形 API **wgpu** 构建，旨在弥合底层图形 API 与高层交互逻辑之间的鸿沟。

## 为什么选择 Myth？

现代图形 API（如 WebGPU、Vulkan 和 DirectX 12）的功能极其强大，但代价是极其繁琐的样板代码 —— 即使只是绘制一个简单的场景，也需要处理大量的内存屏障和管线状态。

另一方面，当你仅仅需要一个**极致性能的轻量级渲染库**时，像 Bevy 或 Godot 这样包罗万象的重量级商业引擎往往显得过于臃肿（"too heavy"）。

Myth 正是为填补这一空白而生。我们拒绝妥协，通过引入一套严格的底层架构，在保持轻量级的同时，压榨出极致的硬件性能。

## 核心设计哲学

Myth 的引擎底层不仅是一个渲染器，更是一个**图编译器**。

- **基于 SSA 的声明式渲染图 (RenderGraph)：** 我们抛弃了传统的手动状态管理，将渲染过程视为编译器问题。你只需声明渲染阶段的拓扑需求，引擎会自动执行拓扑排序、计算激进的内存别名（Aliasing），并自动剔除死节点（Dead-pass elimination）。**完全无需手动管理资源生命周期和内存屏障。**
- **为下一代 3D 交互打造：** 原生集成 3D Gaussian Splatting (3DGS) 混合管线，并提供强大的 GPU-driven 聚类光照（Clustered Lighting），轻松应对海量动态光源。
- **一套代码，全平台运行：** 基于 Rust 的跨平台特性，支持原生桌面端 (Windows, macOS, Linux)、移动端 (iOS, Android)、以及通过 WebAssembly 无缝导出至 WebGPU 浏览器环境。

::: tip 适用场景
Myth 非常适合需要**高度定制化渲染管线**、**AI Agent 驱动的 3D 数字生命**展示，以及需要嵌入到其他系统中的高性能图形模块。
:::

## 下一步

- 想立刻动手？ → [快速开始](/zh/guide/quick-start)
- 想纵览全部能力？ → [核心特性总览](/zh/guide/features)
- 想理解底层架构？ → [Render Graph 渲染图](/zh/architecture/render-graph)