# Headless & Offscreen Rendering

Myth can run with **no window at all**, reading rendered results directly back into CPU memory. This makes the engine a natural fit for **CI/CD automated testing, cloud rendering, server-side thumbnail generation, and offline video composition**.

## 1. Headless Initialization

Unlike the windowed `App`-based flow, headless rendering operates the `Engine` directly, initializing the GPU at a given resolution (without a surface) via `init_headless`:

```rust
use myth::prelude::*;

fn main() {
    let mut engine = Engine::default();

    let (width, height) = (800, 600);
    // Initialize the GPU in headless mode — no window, no surface
    pollster::block_on(engine.init_headless(width, height, None))
        .expect("headless init failed");

    // …build the scene (see below)
}
```

## 2. Building the Scene

In headless mode, the scene is built exactly as in the windowed flow:

```rust
let scene = engine.scene_manager.create_active();

// A cube with a checkerboard material
let image = engine.assets.images.add(Image::checkerboard(512, 512, 64));
let tex = engine.assets.textures.add(Texture::new_2d(Some("checker"), image));
let _cube = scene.spawn_box(2.0, 2.0, 2.0, UnlitMaterial::new(Vec4::ONE).with_map(tex), &engine.assets);

// Camera
let cam = scene.add_camera(Camera::new_perspective(45.0, width as f32 / height as f32, 0.1));
scene.node(&cam).set_position(0.0, 3.0, 8.0).look_at(Vec3::ZERO);
scene.active_camera = Some(cam);

// Directional light
let light = scene.add_light(Light::new_directional(Vec3::ONE, 5.0));
scene.node(&light).set_position(5.0, 10.0, 5.0).look_at(Vec3::ZERO);
```

## 3. Render and Read Back Pixels

Manually advance a frame, render the active scene, then read the framebuffer back to the CPU:

```rust
// Advance one frame of logic (dt in seconds)
engine.update(0.016);

// Render the active scene
let rendered = engine.render_active_scene();
assert!(rendered, "render_active_scene returned false");

// Read back pixels (RGBA8)
let pixels = engine.readback_pixels().expect("readback failed");

// Save as PNG with the image crate
image::save_buffer(
    "output.png",
    &pixels,
    width,
    height,
    image::ColorType::Rgba8,
).expect("failed to save output.png");
```

::: tip Complete Example
The repository's [`examples/headless_export.rs`](https://github.com/panxinmiao/myth/blob/main/examples/headless_export.rs) provides a ready-to-run complete example:
```bash
cargo run --example headless_export --no-default-features
```
:::

## 4. High-Throughput Readback Stream

For scenarios that **continuously export many frames** (e.g. offline video generation), the engine has a built-in **non-blocking asynchronous GPU→CPU readback pipeline**:

- A **ring-buffer** architecture pipelines multi-frame readback, preventing the GPU from stalling while it waits for the CPU to read.
- Built-in **automatic back-pressure** self-regulates when CPU consumption can't keep up, preventing unbounded memory growth.

This design lets Myth sustain near-realtime throughput while continuously producing frame sequences — ideal for server-side batch rendering and video pipelines.

## Use Cases

- **CI / regression testing:** Render reference images on a GPU server with no display and do pixel-level comparison.
- **Cloud rendering / thumbnail services:** Generate preview images of 3D models on demand.
- **Offline video generation:** Render frame-by-frame at a fixed step and encode to video.

## Next Steps

- Understand the full frame composition flow → [Render Paths & Frame Composer](/architecture/rendering-pipeline)
- Understand the underlying resource scheduling → [Render Graph](/architecture/render-graph)
