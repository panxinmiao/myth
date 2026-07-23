---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Myth Engine"
  text: "Rust 原生的下一代渲染引擎"
  tagline: 轻量且跨平台，为实时 3D 与智能交互提供纯粹动力。
  image:
    src: /images/hero.png
    alt: Myth Engine
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/guide/quick-start
    - theme: alt
      text: 核心特性
      link: /zh/guide/features
    - theme: alt
      text: 在线演示
      link: /gallery/
      target: _self
    - theme: alt
      text: GitHub
      link: https://github.com/panxinmiao/myth

features:
  - icon: 🦀
    title: 纯粹的 Rust 核心
    details: 内存安全、零成本抽象、无 GC 负担。基于 wgpu，覆盖 Vulkan / Metal / DX12 / WebGPU。
  - icon: ⚙️
    title: SSA 渲染图编译器
    details: 声明式 RenderGraph，自动拓扑排序、死节点剔除与内存别名，零手动屏障。
  - icon: ✨
    title: 完整 PBR 与后处理
    details: Clearcoat / Transmission / Sheen 等高级材质，搭配 Bloom、TAA 与 SSAO / SSR / SSGI / SSSS。
  - icon: 💡
    title: GPU-Driven 聚类光照
    details: Compute 驱动的聚类前向光照，轻松驱动成百上千个动态点光源与聚光灯。
  - icon: 🌌
    title: 原生 3D 高斯溅射
    details: 一等公民的 3DGS 混合管线，与 PBR 几何在物理正确的色彩空间下深度融合。
  - icon: 🌐
    title: 一套代码，全平台运行
    details: 原生桌面 / 移动端 + WebGPU/WASM 浏览器 + Python 绑定，并支持无头离屏渲染。
---