# Feature Overview

Myth delivers modern rendering capabilities rivaling heavyweight commercial engines while staying **lightweight**. This page gives a bird's-eye view of the engine's core capabilities to help you quickly decide whether it fits your project.

## Capability Matrix

| Domain | Capabilities |
| :--- | :--- |
| **Core Architecture** | SSA-based declarative RenderGraph compiler, zero-allocation per-frame rebuild, automatic memory aliasing, dead-pass elimination |
| **Cross-platform** | Windows / macOS / Linux / iOS / Android · WebGPU/WASM · Python bindings |
| **Backend** | Built on wgpu — Vulkan / Metal / DX12 / WebGPU |
| **Lighting** | Clustered forward lighting, Image-Based Lighting (IBL), Cascaded Shadow Maps (CSM), spot-light shadows |
| **Materials** | Full PBR (Clearcoat / Iridescence / Transmission / Sheen / Anisotropy), Phong, Unlit, custom material macro |
| **Screen-Space FX** | SSAO, SSR, SSGI, SSSS (subsurface scattering) |
| **Post-Processing** | HDR pipeline, Bloom, color grading, TAA / FXAA / MSAA, CAS sharpening |
| **Environment** | Procedural sky (Hillaire 2020 atmospheric scattering), sun/moon/stars, day-night cycle, skybox, HDR environment maps |
| **3DGS** | GPU-driven 3D Gaussian Splatting, unified with the PBR pipeline |
| **Assets** | Full glTF 2.0 (PBR, animation, morph targets), asynchronous asset system |
| **Tooling** | Embedded egui inspector, runtime RenderGraph topology dump |
| **Offscreen** | Headless rendering, non-blocking GPU→CPU readback ring buffer |

## Design Trade-offs

Myth's goal is not to be yet another "batteries-included" engine, but a **focused, embeddable, customizable** high-performance rendering core.

- **Focused on rendering:** We concentrate on the graphics pipeline itself. Physics, audio, and networking are left to your choice — the engine imposes no architecture on you.
- **Embeddable:** The engine can run windowless, or embed into existing winit / egui / custom event loops, making it ideal for editors, visualization tools, or server-side rendering.
- **Compiler mindset:** Rendering is modeled as a graph-compilation problem. You declare topological needs; the compiler handles scheduling, synchronization, and memory reuse — leaving the complexity to the compiler and the creativity to you.

## Render Paths at a Glance

The engine offers two main render paths, selectable via `RendererSettings` when creating an `App`:

- **`RenderPath::HighFidelity`:** The full PBR + post-processing + screen-space FX + 3DGS pipeline. Most features (Bloom, SSAO, SSR, SSGI, TAA, 3DGS, etc.) depend on this path.
- **`RenderPath::BasicForward`:** Minimal forward rendering, suited to low-end devices or scenes that only need basic shading.

See [Render Paths & Frame Composer](/architecture/rendering-pipeline) for details.

## Next Steps

- Ready to dive in? → [Quick Start](/guide/quick-start)
- Want the engine's mental model? → [Scene & Node System](/guide/scene-graph)
- Want to go deep on the architecture? → [Render Graph](/architecture/render-graph)
