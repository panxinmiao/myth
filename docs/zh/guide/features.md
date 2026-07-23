# 核心特性总览

Myth 在保持**轻量级**的同时，提供了媲美重型商业引擎的现代渲染能力。本页对引擎的核心能力做一个鸟瞰式梳理，帮助你快速判断它是否契合你的项目。

## 能力矩阵

| 领域 | 能力 |
| :--- | :--- |
| **核心架构** | 基于 SSA 的声明式 RenderGraph 编译器、每帧零分配重建、自动内存别名、死节点剔除 |
| **跨平台** | Windows / macOS / Linux / iOS / Android · WebGPU/WASM · Python 绑定 |
| **后端** | 基于 wgpu，支持 Vulkan / Metal / DX12 / WebGPU |
| **光照** | 聚类前向光照 (Clustered Forward)、IBL 基于图像的光照、CSM 级联阴影、聚光灯阴影 |
| **材质** | 完整 PBR（Clearcoat / Iridescence / Transmission / Sheen / Anisotropy）、Phong、Unlit、自定义材质宏 |
| **屏幕空间特效** | SSAO、SSR、SSGI、SSSS（次表面散射） |
| **后处理** | HDR 管线、Bloom、色彩分级、TAA / FXAA / MSAA、CAS 锐化 |
| **环境** | 程序化天空（Hillaire 2020 大气散射）、日月星辰、昼夜循环、Skybox、HDR 环境贴图 |
| **3DGS** | GPU-Driven 3D 高斯溅射，与 PBR 管线统一融合 |
| **资产** | 完整 glTF 2.0（PBR、动画、Morph Target）、异步资产系统 |
| **工具** | 内嵌 egui Inspector、运行时 RenderGraph 拓扑导出 |
| **离屏** | 无头渲染、非阻塞 GPU→CPU 回读环形缓冲 |

## 设计取舍

Myth 的目标不是成为又一个 "全家桶" 引擎，而是做一个**专注、可嵌入、可定制**的高性能渲染核心。

- **专注渲染：** 我们把精力集中在图形管线本身。物理、音频、网络等系统留给你自由选择，引擎不强加架构。
- **可嵌入：** 引擎可以无窗口运行，也可以嵌入到既有的 winit / egui / 自定义事件循环中，便于做编辑器、可视化工具或服务端渲染。
- **编译器思维：** 渲染过程被建模为一个图编译问题。你声明拓扑需求，编译器负责调度、同步与内存复用——把复杂性交给编译器，把创造力还给开发者。

## 渲染路径一览

引擎提供两条主要的渲染路径，可在创建 `App` 时通过 `RendererSettings` 指定：

- **`RenderPath::HighFidelity`（高保真）：** 完整的 PBR + 后处理 + 屏幕空间特效 + 3DGS 管线。绝大多数特性（Bloom、SSAO、SSR、SSGI、TAA、3DGS 等）都依赖该路径。
- **`RenderPath::BasicForward`（基础前向）：** 极简前向渲染，适合低端设备或仅需基础着色的场景。

详见 [渲染路径与帧合成](/zh/architecture/rendering-pipeline)。

## 下一步

- 想立刻上手？ → [快速开始](/zh/guide/quick-start)
- 想理解引擎心智模型？ → [场景与节点系统](/zh/guide/scene-graph)
- 想深入底层架构？ → [Render Graph 渲染图](/zh/architecture/render-graph)
