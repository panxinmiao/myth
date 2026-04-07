# Changelog

## Unreleased

## v0.2.0

Released 2026-04-07

### Major Changes
- Add a dedicated `myth_macros` crate for generating Material and GPU data structures. The previous complex declarative macros have been removed, making the creation of Materials and GPU data structs simpler and more ergonomic, with a more user-friendly API.
- Refactored the asynchronous asset loading system, introducing a “fire-and-forget” style, ergonomic API. All asynchronous loading logic is now fully handled internally by the engine.
- **Headless Rendering Mode**: Added support for offscreen rendering without a window surface (`Renderer.init_headless`). Ideal for server-side rendering, CI/CD testing, and offline video/image generation.
  - **Synchronous GPU Readback**: Introduced `Renderer.readback_pixels()` for simple, one-shot synchronous GPU-to-CPU pixel data extraction.
  - **High-Throughput Asynchronous Readback Stream**: Implemented `ReadbackStream`, a non-blocking ring-buffer pipeline for continuous frame readback. Designed for extreme performance in video recording and AI training data generation without stalling the GPU pipeline.
- Refactored the shader management and templating system. Shader code is now organized based on functional semantics and responsibility boundaries. The API entry point for creating shader programs has been consolidated and unified, and support has been added for loading custom shaders from external files.

### Added
- Added point light shadows, completing the final piece of the basic lighting system.
- Added a “debug_view” feature, enabling real-time inspection of material base textures (albedo, metalness, roughness) as well as in-frame buffers (depth, normal, SSAO, velocity, and more).
- `AssetServer::load_lut_texture` now supports both `.cube` and `.bin` LUT files, with automatic format detection based on file extension.

### Changes
- Updated a number of crates to latest versions.

### Fixed
- Fixed an issue that caused edge jittering of objects in TAA.

## v0.1.1

Released 2021-03-26

#### Changes
- Use `ehttp` instead of `reqwest`.
- Release python bindings package (`myth-py`) on PyPI.
- Update documentation.

## v0.1.0

Released 2026-03-25

### First release of `Myth Engine`.

Myth is a developer-friendly, high-performance 3D rendering engine written in Rust.

Inspired by the ergonomic simplicity of Three.js and built on the modern power of wgpu, Myth aims to bridge the gap between low-level graphics APIs and high-level game engines.

### Features

* **Core Architecture & Platform**
    * **True Cross-platform, One Codebase**: Native (Windows, macOS, Linux, iOS, Android) + WebGPU/WASM + Python bindings.
    * **Modern Backend**: Built on **wgpu**, fully supporting Vulkan, Metal, DX12, and WebGPU.
    * **SSA-based Render Graph**: A declarative, compiler-driven rendering architecture. You declare the topological needs, and the engine handles the rest:
        * **Automatic Synchronization**: Zero manual memory barriers or layout transitions.
        * **Aggressive Memory Aliasing**: Reuses transient high-resolution physical textures perfectly across distinct logical passes.
        * **Dead Pass Elimination**: Automatically culls rendering workloads.
        * **Zero-Allocation Per-Frame Rebuild**: Evaluates and compiles the entire DAG every frame.

* **Advanced Rendering & Lighting**
    * **Physically Based Materials**: Robust PBR pipeline with Clearcoat, Iridescence, Transmission, Sheen, Anisotropy.
    * **Image-Based Lighting (IBL)** + **Dynamic Shadows (CSM)**.
    * **SSAO / SSSS / Skybox**.

* **Post-Processing & FX**
    * **HDR Pipeline** + **Bloom** + **Color Grading** + **TAA / FXAA / MSAA**.

* **Assets & Tooling**
    * **Full glTF 2.0 Support** (PBR, animations, morph targets).
    * **Asynchronous Asset System** + **Embedded egui Inspector**.

## Diffs

- [Unreleased](https://github.com/panxinmiao/myth/compare/0.2.0...HEAD)
- [v0.2.0](https://github.com/panxinmiao/myth/compare/0.1.1...0.2.0)
- [v0.1.1](https://github.com/panxinmiao/myth/compare/0.1.0...0.1.1)
- [v0.1.0](https://github.com/panxinmiao/myth/compare/0.0.1...0.1.0)