---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Myth Engine"
  text: "Next-Gen Rendering in Rust"
  tagline: Lightweight and cross-platform. Powering real-time 3D and intelligent interactions.
  image:
    src: /images/hero.png
    alt: Myth Engine
  actions:
    - theme: brand
      text: Quick Start
      link: /guide/quick-start
    - theme: alt
      text: Feature Overview
      link: /guide/features
    - theme: alt
      text: Live Gallery
      link: /gallery/
      target: _self
    - theme: alt
      text: GitHub
      link: https://github.com/panxinmiao/myth

features:
  - icon: 🦀
    title: Pure Rust Core
    details: Memory-safe, zero-cost abstractions, no GC overhead. Built on wgpu across Vulkan / Metal / DX12 / WebGPU.
  - icon: ⚙️
    title: SSA RenderGraph Compiler
    details: Declarative RenderGraph with automatic topological sort, dead-pass elimination and memory aliasing, zero manual barriers.
  - icon: ✨
    title: Full PBR & Post-Processing
    details: Advanced materials like Clearcoat / Transmission / Sheen, paired with Bloom, TAA and SSAO / SSR / SSGI / SSSS.
  - icon: 💡
    title: GPU-Driven Clustered Lighting
    details: Compute-driven clustered forward lighting for hundreds of dynamic point and spot lights.
  - icon: 🌌
    title: Native 3D Gaussian Splatting
    details: First-class hybrid 3DGS pipeline, deeply fused with PBR geometry in a physically-correct color space.
  - icon: 🌐
    title: One Codebase, Every Platform
    details: Native desktop / mobile + WebGPU/WASM browsers + Python bindings, plus headless offscreen rendering.
---
