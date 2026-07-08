# Changelog

## Unreleased

### Major Changes
- Introduced **SSR (Screen Space Reflections)** as a first-class real-time reflection feature in Myth's high-fidelity renderer.
  - **Hierarchical Screen-Space Tracing:** Uses Hi-Z ray marching against the scene depth pyramid to trace glossy and mirror-like reflections efficiently.
  - **Spatiotemporal Stabilization:** Combines history reprojection, temporal accumulation, and roughness-aware spatial cleanup to reduce shimmer and preserve stable reflection energy.
  - **Scalable Quality Tiers:** Provides Low / Medium / High / Ultra presets that scale trace distance, ray budget, trace resolution, and denoiser cost across different hardware targets.
  - **Scene-Level Controls:** Exposed through `SsrSettings`, allowing per-scene enablement and tuning of intensity, distance, thickness, and denoising behaviour.
  - **Debugging & Validation Workflow:** Includes dedicated debug views together with the `examples/ssr_showroom.rs` reference scene for tuning and regression checks.

- Introduced **SSGI (Screen Space Global Illumination)** as a first-class real-time global illumination feature in Myth's high-fidelity renderer.
  - **Physically Plausible Indirect Lighting:** Adds diffuse bounce lighting to the post-lighting pipeline and merges the result back into HDR scene color through the RenderGraph.
  - **Efficient Screen-Space Ray Tracing:** Uses half-resolution Hi-Z ray marching together with blue-noise driven hemisphere sampling to estimate indirect radiance at low cost.
  - **Spatiotemporal Denoising:** Combines history reprojection, temporal accumulation, and multi-pass A-Trous wavelet filtering to stabilize 1-SPP indirect lighting and remove low-frequency blotches.
  - **Robust Hit Validation:** Includes thickness-aware hit rejection to reduce light leaking on thin surfaces and grazing-angle intersections.
  - **Scalable Quality Tiers:** Provides Low / Medium / High / Ultra presets to scale ray budget and denoiser cost across desktop and constrained hardware targets.

- Introduced a **procedural sky system** powered by a physically-based atmospheric scattering model (Hillaire 2020), enabling high-quality real-time sky rendering with procedural celestial bodies (sun, moon, and stars).
Also includes a `DayNightCycle` component for dynamic time progression, automatically syncing the trajectories of the sun, moon, and star field with scene parameters.

- Overhauled the **RenderGraph** into a more complete typed-resource system: buffers and textures are now first-class SSA resources with unified dependency tracking, zero-cost typed node handles, and transient power-of-two buffer pooling for aggressive VRAM reuse.
Also migrated major compute-heavy paths such as 3D Gaussian Splatting, atmosphere baking, and PMREM/environment processing onto RDG-managed buffer lifetimes, removing ad-hoc side channels around the graph.

- Introduced a unified cached bind-group assembly API across `PrepareContext` and `ExtractContext`, centered around a fluent builder plus `myth_bind_group!`.This removes large amounts of repetitive WGPU boilerplate, unifies static and transient bind-group construction, and guarantees that RDG buffer bindings clamp pooled physical allocations back to their logical resource sizes.

- Introduced **Clustered Shading (Forward+)** to support rendering massive numbers of dynamic lights. Built on a production-oriented, fully GPU-driven compute pipeline:
  - **Precise Light Culling:** High-accuracy view-space frustum vs. sphere intersection testing for reliable light assignment across the logarithmic depth grid.
  - **Dynamic VRAM Allocation:** Utilizes a global atomic allocator to support extreme light densities dynamically, maximizing memory efficiency without artificial per-cluster limits.
  - **High-Performance Compute:** Aggressively leverages shared memory (`var<workgroup>`) and parallel workgroup execution to drastically reduce ALU and bandwidth overhead.
  - **Visual Profiling:** Added a `ClusterHeatmap` debug view to inspect light density and grid allocation in real-time.

- Introduced 3D Gaussian Splatting (3DGS) as a first-class rendering primitive behind the `3dgs` feature flag.
  - High-Performance Sorting: Fast GPU radix sort for depth sorting millions of splats per frame, ensuring correct transparency and blending without CPU overhead.
  - Scene Graph & Post-Processing Integration: Full integration with the render state and scene graph. Compatible with post-processing passes.
  - Asset Support: Load point clouds from compressed `.npz` (via `gaussian-npz`) and standard `.ply` formats via the async asset server.

### Refactored / Changed
- Updated Myth's graphics backend dependency to `wgpu` v30 and adjusted render graph, pipeline, readback, and surface presentation code for the new API.
- Split egui support into a dedicated `myth_egui` crate with a Myth-native renderer and optional `myth/egui` facade, removing the examples' dependency on `egui-wgpu`.
  - Added stable Myth texture registration for egui via `UiPass::texture_id` / `UiPass::register_texture`, allowing `TextureHandle` values to be shown directly in egui while the backend keeps the GPU view and sampler synchronized.

- Removed the `compose_frame` method from `AppHandler` and narrowed its responsibility to providing only a high-level render trigger, with full render graph orchestration delegated to the `Engine`.
  > _Note: This simplifies `AppRunner`, returning control of `RedrawRequested` execution to the user. It also improves the extensibility of headless mode — `FrameComposer` can now be accessed directly to attach custom RenderGraph nodes (e.g., offline data extraction or custom compute passes), without being constrained by the window system lifecycle._

- Updated `RenderCamera` in `Renderer::begin_frame` and `ComposerContext` to be passed by value.
  > _Note: This clarifies the architectural intent of `RenderCamera` data as a transient snapshot and removes borrowing dependencies on local variables._

### Added
- Added `#[myth::main]` macro for ergonomic application entry point definition, unifying entry points across Native and WASM platforms.
- Added "dynamic image/texture" support, via `Image::new_dynamic` and `AssetServer::update_dynamic_texture`. This provides a simple API for real-time updating of texture content from CPU data (without allocation), ideal for video streaming, dynamic UI elements, or procedural textures.
- Added some custom material examples to the Gallery, showcasing the use of custom shader code and material definitions.
- Added some primitive geometry constructors to the API, such as `create_cone`, `create_cylinder`, `create_torus`, etc.
- Added `LIGHT_FLAG_IS_SUN`/`LIGHT_FLAG_IS_MOON` flags to the `Light` struct, allowing the renderer to identify and treat directional lights as celestial bodies for sky rendering and IBL purposes. Now sunlight correctly considers the atmospheric scattering effects from the procedural sky system.

### Fixed
- Fixed an issue of UnlitMaterial UV transform not taking effect.
- Fixed an issue causing conflicts in GPU resource ID allocation.

### Engineering & DX (Developer Experience)
* **[Build]** Replaced legacy shell/batch build scripts with a pure-Rust `cargo xtask` workflow, ensuring cross-platform consistency for WebAssembly compilation and Gallery generation.
* **[Gallery]** Redesigned the Gallery infrastructure to support examples and dynamic, parameter-driven standalone applications. Migrated complex examples such as `showcase` and `gltf_viewer` into standalone apps with independent builds, significantly reducing engine `dev-dependencies`.

## v0.2.0

Released 2026-04-07

### Major Changes
- Added a dedicated `myth_macros` crate for generating Material and GPU data structures. The previous complex declarative macros have been removed, making the creation of Materials and GPU data structs simpler and more ergonomic, with a more user-friendly API.
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
