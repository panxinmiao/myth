<div align="center">

# Myth

**A High-Performance, WGPU-Based Rendering Engine for Rust.**

[![CI](https://github.com/panxinmiao/myth/actions/workflows/ci.yml/badge.svg)](https://github.com/panxinmiao/myth/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/panxinmiao/myth/actions/workflows/deploy.yml/badge.svg)](https://github.com/panxinmiao/myth/actions/workflows/deploy.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](LICENSE)
[![WebGPU Ready](https://img.shields.io/badge/WebGPU-Ready-green.svg)](https://gpuweb.github.io/gpuweb/)

![Myth Engine Hero](docs/images/hero.png)

[**Online Web Demo**](https://panxinmiao.github.io/myth/) | [**Examples**](examples/)

</div>

---

> ⚠️ **Warning: Early Development Stage**
>
> Myth is currently in **active alpha development**. APIs are unstable and subject to **drastic breaking changes**. 

## ✨ Introduction

**Myth** is a developer-friendly, high-performance 3D rendering engine written in **Rust**. 

Inspired by the simplicity of **Three.js** and built on the modern power of **wgpu**, Myth aims to bridge the gap between low-level graphics APIs and high-level game engines. It provides a familiar object-oriented API for rapid development, backed by a **Transient Render Graph** architecture for industrial-grade performance.


## 🚀 Features

* **⚡ Modern Architecture**: Built on **wgpu**, fully supporting **Vulkan**, **Metal**, **DX12**, and **WebGPU**.
* **✨ Advanced PBR**: Industry-standard Physically Based Rendering pipeline.
    * Metalness/Roughness workflow.
    * **IBL** (support cubeMap & equirectangular env maps, with auto PMREM generation).
    * **Clearcoat** (car paint, varnished wood).
    * **Iridescence** (soap bubbles, oil films).
    * **Sheen** (cloth-like materials).
    * **Anisotropy** (brushed metals).
    * **Transmission** (glass, water).
* **✨ Full glTF 2.0 Support**: Complete support for glTF 2.0 specification, including PBR materials, animations, and scene hierarchy.
* **🎨 HDR Rendering Pipeline**: Full support for HDR Rendering, various tone mapping mode.
* **🛡️ MSAA**: Built-in Multi-Sample Anti-Aliasing.
* **🕸️ Transient Render Graph**: A highly optimized, frame-graph based rendering architecture that minimizes overhead and maximizes flexibility.
* **📦 Asset System**: Asynchronous asset loading with `AssetServer`, built-in **glTF 2.0** support (geometry, materials, animations).
* **🛠️ Tooling Ready**: Includes a powerful `gltf_viewer` example with an embedded **Inspector UI** (based on egui), capable of inspecting scene nodes, materials, and textures at runtime.
* **🌐 Web First**: First-class support for **WASM** and **WebGPU**. Write once, run everywhere.

## 🎮 Online Demo

Experience the engine directly in your browser (Chrome/Edge 113+ required for WebGPU):

👉 **[Launch glTF Viewer & Inspector](https://panxinmiao.github.io/myth/)**

* **Drag & Drop** your own `.glb` / `.gltf` files to view them.
* Inspect node hierarchy and tweak PBR material parameters in real-time.

![Web Editor Preview](docs/images/inspector.png)


## 📦 Quick Start

Add `myth` to your `Cargo.toml`:

```toml
[dependencies]
myth = { git = "https://github.com/panxinmiao/myth", branch = "main" }

```

### The "Hello World" (Three.js Style)

A spinning cube with a checkerboard texture within less than 50 lines of code.
Notice how similar this feels to the JS equivalent, but statically typed:

```rust
use myth::prelude::*;

struct MyApp;

impl AppHandler for MyApp {
    fn init(engine: &mut Engine, _: &std::sync::Arc<winit::window::Window>) -> Self {
        // 0. Create a Scene
        let scene = engine.scene_manager.create_active();

        // 1. Create a Material
        let material = MeshPhongMaterial::new(Vec4::new(1.0, 0.76, 0.33, 1.0));
        let texture = Texture::create_checkerboard(Some("checker"), 512, 512, 64);
        let tex_handle = engine.assets.textures.add(texture);
        material.set_map(Some(tex_handle));
        let mat_handle = engine.assets.materials.add(material);

        // 2. Create Geometry & Mesh
        let geometry = Geometry::new_box(1.0, 1.0, 1.0);
        let geo_handle = engine.assets.geometries.add(geometry);
        let mesh = Mesh::new(geo_handle, mat_handle);
        let mesh_handle = scene.add_mesh(mesh);
        
        // 3. Setup Camera
        let camera = Camera::new_perspective(45.0, 16.0/9.0, 0.1);
        let cam_node = scene.add_camera(camera);
        // Move camera back
        scene.nodes.get_mut(cam_node).unwrap().transform.position = Vec3::new(0.0, 0.0, 5.0);
        scene.active_camera = Some(cam_node);
        
        // 4. Add Light
        scene.add_light(Light::new_directional(Vec3::ONE, 5.0));

        // 5. Setup update callback to rotate the cube
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

### 🏃 Running Examples

Clone the repository and run the examples directly:

```bash
# Run the PBR Box example
cargo run --example box_pbr

# Run the glTF Viewer (Desktop)
cargo run --example gltf_viewer --release

# Run the glTF Viewer (Web/WASM)
cd examples/gltf_viewer/web
./build_wasm.sh
python3 -m http.server 8080

```

## 📄 License

This project is licensed under the **MIT License** or **Apache-2.0 License**.