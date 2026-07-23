# Quick Start

In this section we'll render a spinning, checkerboard-textured 3D cube on screen in under 50 lines of code.

## 1. Installation & Dependencies

First, create a new Rust project and add the Myth engine dependency. In your `Cargo.toml`:

```toml
[dependencies]
# Stable release from crates.io
myth-engine = "0.2.0"

# Or the latest from GitHub
# myth-engine = { git = "https://github.com/panxinmiao/myth", branch = "main" }
```

::: info Feature Flags
Myth is modular, and many heavyweight capabilities are hidden behind feature flags by default to keep the engine lean. In practice you may need to enable:

* `gltf`: Load glTF and GLB model assets
* `3dgs`: Enable 3D Gaussian Splatting rendering support
* `gaussian-npz`: Load compressed NPZ-format Gaussian point clouds
* `debug_view`: Enable debug views and the render-graph inspector tooling
:::

## 2. Your First App (Hello World)

Myth uses an extremely intuitive API. Create `src/main.rs` and fill in:

```rust
use myth::prelude::*;

struct MyApp;

impl AppHandler for MyApp {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        // 1. Create and activate a scene
        let scene = engine.scene_manager.create_active();

        // 2. Create a cube with a checkerboard texture
        let tex_handle = engine.assets.checkerboard(512, 64);
        let mesh_handle = scene.spawn_box(
            1.0, 1.0, 1.0,
            PhongMaterial::new(Vec4::new(1.0, 0.76, 0.33, 1.0)).with_map(tex_handle),
            &engine.assets,
        );

        // 3. Set up camera and viewport
        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene.node(&cam_node_id).set_position(0.0, 0.0, 5.0).look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);

        // 4. Add a light
        scene.add_light(Light::new_directional(Vec3::ONE, 5.0));

        // 5. Register a per-frame update callback
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

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Myth-Engine Hello World")
        .with_settings(RendererSettings {
            path: RenderPath::HighFidelity,
            ..Default::default()
        })
        .run::<MyApp>()
}
```

## 3. Run Your Program

Run the project with Cargo and experience your first Myth 3D scene:

```bash
cargo run --release
```

::: tip Choosing a Render Path
We specified `RenderPath::HighFidelity` in `main`. If your scene only needs very basic forward rendering and runs on low-end hardware, you can switch to `RenderPath::BasicForward`. But if you need PBR, Bloom, SSAO, or 3DGS features, always stay on the `HighFidelity` path.
:::

## 4. Run the Official Examples

The repository ships 50+ examples covering all kinds of features. After cloning, run them directly:

```bash
# Run a single example (e.g. the Earth demo)
cargo run --example earth --release

# Run a standalone app (e.g. the glTF Viewer)
cargo run -p gltf_viewer --release
```

For building Web/WASM examples, see the [myth xtask Guide](https://github.com/panxinmiao/myth/blob/main/xtask/README.md).

## Next Steps

- Understand the engine's four-layer mental model → [Scene & Node System](/guide/scene-graph)
- Load models and play animations → [Assets, glTF & Animation](/guide/assets-animation)
- Survey all capabilities → [Feature Overview](/guide/features)
