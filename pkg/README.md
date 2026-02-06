<div align="center">

# Myth Engine

**The Mythical 3D Engine for Rust.**

[![Crates.io](https://img.shields.io/crates/v/myth-engine.svg)](https://crates.io/crates/myth-engine)
[![Docs.rs](https://docs.rs/myth-engine/badge.svg)](https://docs.rs/myth-engine)
[![Build Status](https://img.shields.io/github/actions/workflow/status/panxinmiao/myth-engine/ci.yml)](https://github.com/panxinmiao/myth-engine/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](LICENSE)
[![WebGPU Ready](https://img.shields.io/badge/WebGPU-Ready-green.svg)](https://gpuweb.github.io/gpuweb/)

![Myth Engine Hero](docs/images/hero_render.jpg)

[**在线体验 Web Demo**](https://panxinmiao.github.io/myth-engine/) | [**文档**](https://docs.rs/myth-engine) | [**示例**](examples/)

</div>

---

## ✨ Introduction

**Myth Engine** (myth-engine) is a developer-friendly, high-performance 3D rendering engine written in **Rust**. 

Inspired by the simplicity of **Three.js** and built on the modern power of **wgpu**, Myth aims to bridge the gap between low-level graphics APIs and high-level game engines. It provides a familiar object-oriented API for rapid development, backed by a **Transient Render Graph** architecture for industrial-grade performance.


## 🚀 Features

* **⚡ Modern Architecture**: Built on **wgpu**, fully supporting **Vulkan**, **Metal**, **DX12**, and **WebGPU**.
* **🎨 Advanced PBR**: Industry-standard Physically Based Rendering pipeline.
    * Metalness/Roughness workflow.
    * **Clearcoat** (car paint, varnished wood).
    * **Transmission** (glass, water).
    * **Iridescence** (soap bubbles, oil films).
    * **HDR** Environment Lighting (IBL).
* **🕸️ Transient Render Graph**: A highly optimized, frame-graph based rendering architecture that minimizes overhead and maximizes flexibility.
* **📦 Asset System**: Asynchronous asset loading with `AssetServer`, built-in **glTF 2.0** support (geometry, materials, animations).
* **🛠️ Tooling Ready**: Includes a powerful `gltf_viewer` example with an embedded **Inspector UI** (based on egui), capable of inspecting scene nodes, materials, and textures at runtime.
* **🌐 Web First**: First-class support for **WASM** and **WebGPU**. Write once, run everywhere.

## 🎮 Online Demo

Experience the engine directly in your browser (Chrome/Edge 113+ required for WebGPU):

👉 **[Launch glTF Viewer & Inspector](https://panxinmiao.github.io/myth-engine/)**

* **Drag & Drop** your own `.glb` / `.gltf` files to view them.
* Inspect node hierarchy and tweak PBR material parameters in real-time.

![Web Editor Preview](docs/images/editor_preview.jpg)

## 📦 Quick Start

Add `myth-engine` to your `Cargo.toml`:

```toml
[dependencies]
myth-engine = "0.0.1"