# Introduction & Vision

**Myth Engine** is a developer-friendly, high-performance 3D rendering engine written in **Rust**.

Inspired by the ergonomic, intuitive API design of Three.js and built on the modern graphics API **wgpu**, Myth aims to bridge the gap between low-level graphics APIs and high-level interaction logic.

## Why Myth?

Modern graphics APIs (WebGPU, Vulkan, DirectX 12) are extraordinarily powerful — but that power comes at the cost of overwhelming boilerplate. Even drawing a simple scene requires juggling memory barriers and pipeline state.

On the other hand, when all you need is a **lean, high-performance rendering library**, all-in-one heavyweight engines like Bevy or Godot often feel *too heavy*.

Myth was born to fill that gap. We refuse to compromise: through a strict low-level architecture, Myth stays lightweight while squeezing out extreme hardware performance.

## Core Design Philosophy

At its foundation, Myth is not merely a renderer — it is a **graph compiler**.

- **SSA-based Declarative RenderGraph:** We abandon traditional manual state management and treat rendering as a compiler problem. You simply declare the topological needs of each render stage, and the engine performs topological sorting, computes aggressive memory aliasing, and culls dead passes automatically. **No manual resource lifetime or memory-barrier management required.**
- **Built for Next-Gen 3D Interaction:** Natively integrates a 3D Gaussian Splatting (3DGS) hybrid pipeline and provides powerful GPU-driven Clustered Lighting to effortlessly handle massive numbers of dynamic lights.
- **One Codebase, Every Platform:** Thanks to Rust's cross-platform nature, Myth runs on native desktop (Windows, macOS, Linux), mobile (iOS, Android), and seamlessly targets WebGPU browsers via WebAssembly.

::: tip Where Myth Shines
Myth is ideal for projects that need a **highly customizable rendering pipeline**, **AI-agent-driven 3D digital life** showcases, or a high-performance graphics module that embeds into a larger system.
:::

## Next Steps

- Want to get hands-on right away? → [Quick Start](/guide/quick-start)
- Want a bird's-eye view of all capabilities? → [Feature Overview](/guide/features)
- Want to understand the architecture? → [Render Graph](/architecture/render-graph)
