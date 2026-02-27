# Myth Engine User Guide

A comprehensive guide to building 3D applications with Myth Engine.

---

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Your First Application](#your-first-application)
- [Core Concepts](#core-concepts)
  - [Architecture Overview](#architecture-overview)
  - [Application Lifecycle](#application-lifecycle)
  - [Handle System](#handle-system)
- [Building Scenes](#building-scenes)
  - [Creating a Scene](#creating-a-scene)
  - [Adding Objects](#adding-objects)
  - [Node Hierarchy](#node-hierarchy)
  - [Transforms](#transforms)
  - [Scene Logic](#scene-logic)
- [Materials & Textures](#materials--textures)
  - [Material Types](#material-types)
  - [PBR Material Guide](#pbr-material-guide)
  - [Transparency & Alpha](#transparency--alpha)
  - [Loading Textures](#loading-textures)
- [Cameras](#cameras)
- [Lighting](#lighting)
  - [Light Types](#light-types)
  - [Shadows](#shadows)
  - [Environment Lighting (IBL)](#environment-lighting-ibl)
- [Background & Skybox](#background--skybox)
- [Loading 3D Models](#loading-3d-models)
  - [glTF Loading](#gltf-loading)
  - [HTTP/Network Loading](#httpnetwork-loading)
- [Animation](#animation)
  - [Playing Animations](#playing-animations)
  - [Animation Control](#animation-control)
  - [Morph Targets](#morph-targets)
- [Post-Processing Effects](#post-processing-effects)
  - [Render Paths](#render-paths)
  - [Bloom](#bloom)
  - [Tone Mapping & Color Grading](#tone-mapping--color-grading)
  - [SSAO](#ssao)
  - [Anti-Aliasing](#anti-aliasing)
- [Input Handling](#input-handling)
- [Camera Controls](#camera-controls)
- [Custom Render Passes](#custom-render-passes)
- [Building for WebAssembly](#building-for-webassembly)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Introduction

**Myth Engine** is a high-performance 3D rendering engine built with Rust and [wgpu](https://wgpu.rs/). Inspired by the simplicity of [Three.js](https://threejs.org/) and powered by modern graphics APIs (Vulkan, Metal, DX12, WebGPU), Myth provides a developer-friendly API for building interactive 3D applications that run on desktop and web.

### Key Features

- **Physically Based Rendering (PBR)** with advanced extensions (clearcoat, sheen, iridescence, transmission, anisotropy, SSS)
- **Image-Based Lighting (IBL)** with automatic PMREM generation
- **Cascaded Shadow Maps (CSM)** for dynamic shadows
- **HDR post-processing pipeline**: bloom, tone mapping, SSAO, FXAA, color grading
- **Full glTF 2.0 support** including skeletal and morph target animations
- **Cross-platform**: native (Windows/macOS/Linux) and web (WASM + WebGPU)
- **Transient render graph architecture** for maximum GPU efficiency

---

## Getting Started

### Prerequisites

- **Rust** (edition 2024) — install via [rustup](https://rustup.rs/)
- A GPU supporting Vulkan, Metal, DX12, or WebGPU
- For WASM builds: `wasm-bindgen-cli` and `wasm32-unknown-unknown` target

### Installation

Add Myth to your `Cargo.toml`:

```toml
[dependencies]
myth = { git = "https://github.com/panxinmiao/myth", branch = "main" }
env_logger = "0.11"  # Recommended for debug logging
```

### Your First Application

Here's a complete "Hello World" — a spinning textured cube in under 40 lines:

```rust
use myth::prelude::*;

struct MyApp;

impl AppHandler for MyApp {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        // 1. Create a scene
        let scene = engine.scene_manager.create_active();

        // 2. Create a textured cube
        let texture = Texture::create_checkerboard(Some("checker"), 512, 512, 64);
        let tex_handle = engine.assets.textures.add(texture);
        let cube = scene.spawn_box(
            1.0, 1.0, 1.0,
            MeshPhongMaterial::new(Vec4::new(1.0, 0.76, 0.33, 1.0))
                .with_map(tex_handle),
        );

        // 3. Add a light
        scene.add_light(Light::new_directional(Vec3::ONE, 5.0));

        // 4. Setup camera
        let cam = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene.node(&cam).set_position(0.0, 0.0, 5.0).look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);

        // 5. Animate: rotate the cube each frame
        scene.on_update(move |scene, _input, _dt| {
            if let Some(node) = scene.get_node_mut(cube) {
                node.transform.rotation *= Quat::from_rotation_y(0.02) * Quat::from_rotation_x(0.01);
            }
        });

        MyApp
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new().with_title("Myth Engine Demo").run::<MyApp>()
}
```

Run it:

```bash
cargo run --release
```

---

## Core Concepts

### Architecture Overview

Myth Engine follows a layered architecture:

```
App (winit event loop)
  └─ Engine (central coordinator)
       ├─ SceneManager (multiple scenes)
       │    └─ Scene (nodes, components, settings)
       ├─ Renderer (GPU rendering pipeline)
       │    ├─ Render Graph (frame composition)
       │    ├─ Pipeline Cache (shader compilation)
       │    └─ Resource Manager (GPU resources)
       ├─ AssetServer (resource registry)
       └─ Input (keyboard, mouse)
```

**Key design principles**:
- **Engine** is the coordinator — it holds no window logic and is driven by the `AppHandler` trait
- **Scene** uses a hybrid ECS: a slot-map for nodes + sparse component maps for optional data
- **Renderer** rebuilds a transient render graph each frame (Extract → Prepare → Queue → Render)
- **AssetServer** is thread-safe (`Clone`, all `Arc` internally)

### Application Lifecycle

Your application implements the `AppHandler` trait with four lifecycle methods:

```
App::run::<MyApp>()
    │
    ├── GPU initialization
    │
    ├── MyApp::init()          ← Create scene, load assets
    │
    └── Frame Loop ─────────────┐
         ├── MyApp::on_event()  │  (optional: raw event handling)
         ├── Engine::update()   │  (internal: animations, transforms)
         ├── MyApp::update()    │  ← Your per-frame logic
         ├── Renderer::begin_frame()
         ├── MyApp::compose_frame() ← Add custom render passes
         └── Present to screen ─┘
```

### Handle System

All entity references use **generational handles** — 8-byte, `Copy`, type-safe wrappers over `slotmap` keys:

```rust
NodeHandle       // Scene nodes
GeometryHandle   // Geometry resources
MaterialHandle   // Material resources
TextureHandle    // Texture resources
SamplerHandle    // Sampler resources
ActionHandle     // Animation actions
SkeletonKey      // Skeleton resources
```

Handles are safe to store and copy freely. If the referenced entity is removed, the generational check prevents dangling access — lookups simply return `None`.

```rust
// Handles are Copy - store them in your app struct
struct MyApp {
    cube: NodeHandle,
    material: MaterialHandle,
}

// Always check validity before use
if let Some(node) = scene.get_node_mut(self.cube) {
    node.transform.rotation *= Quat::from_rotation_y(dt);
}
```

---

## Building Scenes

### Creating a Scene

Every application needs at least one scene:

```rust
fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
    // Create a scene and set it as active (most common)
    let scene = engine.scene_manager.create_active();

    // Or create without activating
    let handle = engine.scene_manager.create_scene();
    engine.scene_manager.set_active(handle);
}
```

### Adding Objects

The **spawn API** is the fastest way to add objects:

```rust
// Spawn primitives with any material type
let cube = scene.spawn_box(1.0, 1.0, 1.0, material);
let sphere = scene.spawn_sphere(1.0, material);
let plane = scene.spawn_plane(10.0, 10.0, material);

// General-purpose spawn (custom geometry + material)
let handle = scene.spawn(my_geometry, my_material);
```

For more control, use the manual approach:

```rust
// 1. Register resources
let geo_h = engine.assets.geometries.add(geometry);
let mat_h = engine.assets.materials.add(material);

// 2. Create mesh
let mesh = Mesh::new(geo_h, mat_h);

// 3. Add to scene
let node = scene.add_mesh(mesh);
```

### Node Hierarchy

Build parent-child relationships:

```rust
let parent = scene.create_node_with_name("Parent");
let child = scene.create_node_with_name("Child");

// Attach child to parent
scene.attach(child, parent);

// Or use the builder
scene.build_node("Child")
    .with_parent(parent)
    .with_position(Vec3::new(1.0, 0.0, 0.0))
    .build();
```

Child nodes inherit their parent's world transform (position, rotation, scale are combined hierarchically).

### Transforms

Every node has a `Transform` with position, rotation, and scale:

```rust
// Direct access
if let Some(node) = scene.get_node_mut(handle) {
    node.transform.position = Vec3::new(1.0, 2.0, 3.0);
    node.transform.rotation = Quat::from_rotation_y(std::f32::consts::PI / 4.0);
    node.transform.scale = Vec3::splat(2.0);
}

// Chainable wrapper (recommended for setup)
scene.node(&handle)
    .set_position(1.0, 2.0, 3.0)
    .set_rotation_euler(0.0, 1.57, 0.0)
    .set_scale(2.0)
    .look_at(Vec3::ZERO);
```

Myth uses **lazy dirty tracking**: the local matrix is only recomputed when TRS values actually change, and world matrices propagate down the hierarchy automatically each frame.

### Scene Logic

Add per-frame behavior directly to a scene using closures or the `SceneLogic` trait:

```rust
// Closure-based (quick & easy)
scene.on_update(move |scene, input, dt| {
    if input.get_key(Key::Space) {
        if let Some(node) = scene.get_node_mut(cube_handle) {
            node.transform.position.y += 5.0 * dt;
        }
    }
});

// Trait-based (for complex behaviors)
struct RotationBehavior {
    target: NodeHandle,
    speed: f32,
}

impl SceneLogic for RotationBehavior {
    fn update(&mut self, scene: &mut Scene, _input: &Input, dt: f32) {
        if let Some(node) = scene.get_node_mut(self.target) {
            node.transform.rotation *= Quat::from_rotation_y(self.speed * dt);
        }
    }
}

scene.add_logic(RotationBehavior { target: cube, speed: 1.0 });
```

---

## Materials & Textures

### Material Types

Myth provides three built-in material types:

| Type | Lighting | Use Case |
|------|----------|----------|
| `MeshBasicMaterial` | None (unlit) | UI elements, debug visualization, flat colors |
| `MeshPhongMaterial` | Blinn-Phong | Simple shaded objects, legacy workflows |
| `MeshPhysicalMaterial` | Full PBR | Realistic rendering, production quality |

All materials support a **builder pattern** for concise creation:

```rust
// Basic (unlit)
let mat = MeshBasicMaterial::new(Vec4::new(1.0, 0.0, 0.0, 1.0));

// Phong
let mat = MeshPhongMaterial::new(Vec4::new(0.8, 0.8, 0.8, 1.0))
    .with_shininess(32.0)
    .with_map(diffuse_tex);

// PBR Physical
let mat = MeshPhysicalMaterial::new(Vec4::new(1.0, 1.0, 1.0, 1.0))
    .with_roughness(0.3)
    .with_metalness(1.0)
    .with_map(albedo_tex)
    .with_normal_map(normal_tex);
```

Materials can be passed directly to `spawn_*` methods — they're automatically registered in the asset server.

### PBR Material Guide

`MeshPhysicalMaterial` is the most capable material, supporting the full glTF 2.0 PBR specification plus extensions:

#### Basic PBR Setup

```rust
let mat = MeshPhysicalMaterial::new(Vec4::new(0.8, 0.2, 0.2, 1.0))
    .with_roughness(0.4)    // 0 = mirror, 1 = matte
    .with_metalness(0.0);   // 0 = dielectric, 1 = metal
```

#### Common Material Recipes

```rust
// Gold metal
MeshPhysicalMaterial::new(Vec4::new(1.0, 0.84, 0.0, 1.0))
    .with_roughness(0.3)
    .with_metalness(1.0)

// Smooth plastic
MeshPhysicalMaterial::new(Vec4::new(0.2, 0.5, 1.0, 1.0))
    .with_roughness(0.15)
    .with_metalness(0.0)

// Rough concrete
MeshPhysicalMaterial::new(Vec4::new(0.6, 0.6, 0.55, 1.0))
    .with_roughness(0.95)
    .with_metalness(0.0)

// Glass
MeshPhysicalMaterial::new(Vec4::new(1.0, 1.0, 1.0, 1.0))
    .with_roughness(0.0)
    .with_metalness(0.0)
    .with_transmission(1.0, 0.01, 0.5, Vec3::ONE)

// Car paint (clearcoat)
MeshPhysicalMaterial::new(Vec4::new(0.8, 0.0, 0.0, 1.0))
    .with_roughness(0.6)
    .with_clearcoat(1.0, 0.03)

// Velvet cloth (sheen)
MeshPhysicalMaterial::new(Vec4::new(0.3, 0.0, 0.5, 1.0))
    .with_roughness(0.8)
    .with_sheen(Vec3::new(0.5, 0.3, 0.7), 0.5)

// Soap bubble (iridescence)
MeshPhysicalMaterial::new(Vec4::new(1.0, 1.0, 1.0, 0.3))
    .with_iridescence(1.0, 1.3, 100.0, 400.0)
    .with_alpha_mode(AlphaMode::Blend)

// Brushed metal (anisotropy)
MeshPhysicalMaterial::new(Vec4::new(0.9, 0.9, 0.9, 1.0))
    .with_roughness(0.3)
    .with_metalness(1.0)
    .with_anisotropy(0.8, 0.0)
```

### Transparency & Alpha

```rust
// Alpha blending (glass, particles)
material.with_alpha_mode(AlphaMode::Blend)
    .with_opacity(0.5)
    .with_depth_write(false)  // Important for correct blending

// Alpha testing (foliage, fences)
material.with_alpha_mode(AlphaMode::Mask(0.5, false))

// Double-sided rendering
material.with_side(Side::Double)
```

### Loading Textures

```rust
// From file (synchronous, native only)
let tex = engine.assets.load_texture(
    "path/to/diffuse.png",
    ColorSpace::Srgb,   // Use Srgb for color textures
    true,                // Generate mipmaps
)?;

// Linear color space for data textures
let normal = engine.assets.load_texture("normal.png", ColorSpace::Linear, true)?;

// HDR environment maps
let hdr = engine.assets.load_hdr_texture("environment.hdr")?;

// Cubemap (6 faces)
let cube = engine.assets.load_cube_texture(
    ["px.jpg", "nx.jpg", "py.jpg", "ny.jpg", "pz.jpg", "nz.jpg"],
    ColorSpace::Srgb, true,
)?;

// Procedural textures
let checker = Texture::create_checkerboard(Some("checker"), 512, 512, 64);
let solid = Texture::create_solid_color(Some("red"), [255, 0, 0, 255]);
let tex_h = engine.assets.textures.add(checker);
```

**Color space rule**: Use `ColorSpace::Srgb` for visual textures (albedo, emissive). Use `ColorSpace::Linear` for data textures (normal maps, roughness, metalness, AO).

---

## Cameras

### Creating a Camera

```rust
// Perspective camera (most common)
let camera = Camera::new_perspective(
    45.0,           // Field of view in degrees
    16.0 / 9.0,    // Aspect ratio
    0.1,            // Near clip plane
);
// Far plane is infinite by default (reverse-Z for maximum precision)

// Add to scene and activate
let cam_node = scene.add_camera(camera);
scene.node(&cam_node).set_position(0.0, 5.0, 10.0).look_at(Vec3::ZERO);
scene.active_camera = Some(cam_node);
```

### Camera Tips

- Myth uses **infinite reverse-Z** projection for optimal depth precision
- The engine auto-updates the camera aspect ratio on window resize
- Use `Camera::fit_to_scene()` to automatically frame a model

---

## Lighting

### Light Types

```rust
// Directional (sun) — color + intensity
let sun = Light::new_directional(Vec3::ONE, 3.0);
let node = scene.add_light(sun);
scene.node(&node).set_position(5.0, 10.0, 5.0).look_at(Vec3::ZERO);

// Point (bulb) — color + intensity + range
let bulb = Light::new_point(Vec3::new(1.0, 0.9, 0.8), 100.0, 10.0);
let node = scene.add_light(bulb);
scene.node(&node).set_position(2.0, 3.0, 0.0);

// Spot (flashlight) — color + intensity + range + inner/outer cone angles
let spot = Light::new_spot(Vec3::ONE, 100.0, 15.0, 0.3, 0.6);
let node = scene.add_light(spot);
scene.node(&node).set_position(0.0, 5.0, 0.0).look_at(Vec3::ZERO);
```

### Shadows

Enable shadows on any light:

```rust
let mut light = Light::new_directional(Vec3::ONE, 5.0);
light.cast_shadows = true;

// Fine-tune shadow quality
if let Some(shadow) = light.shadow.as_mut() {
    shadow.map_size = 2048;             // Resolution (power of 2)
    shadow.cascade_count = 4;           // CSM cascades for directional lights
    shadow.max_shadow_distance = 100.0; // Max shadow render distance
    shadow.normal_bias = 0.02;          // Reduce shadow acne
}

let light_node = scene.add_light(light);
```

Control per-object shadow behavior:

```rust
// Ground plane: receives shadows but doesn't cast them
scene.node(&ground).set_cast_shadows(false).set_receive_shadows(true);

// Character: casts and receives shadows
scene.node(&character).set_shadows(true, true);
```

### Environment Lighting (IBL)

Image-Based Lighting for realistic ambient illumination:

```rust
// Load HDR environment map
let hdr = engine.assets.load_hdr_texture("studio_garden.hdr")?;

// Apply to scene
scene.environment.set_env_map(Some(hdr));
scene.environment.set_intensity(1.0);  // IBL brightness

// Optional ambient fill light
scene.environment.set_ambient_light(Vec3::splat(0.01));
```

The engine automatically generates a **PMREM** (Prefiltered Mipmap Radiance Environment Map) from your source HDR for both diffuse and specular IBL.

---

## Background & Skybox

Five background modes are available:

```rust
// 1. Solid color (cheapest — hardware clear)
scene.background.set_mode(BackgroundMode::color(0.1, 0.1, 0.15));

// 2. Vertical gradient
scene.background.set_mode(BackgroundMode::gradient(
    Vec4::new(0.05, 0.05, 0.25, 1.0),  // top
    Vec4::new(0.7, 0.45, 0.2, 1.0),    // bottom
));

// 3. Equirectangular HDR panorama (most common for outdoor scenes)
scene.background.set_mode(BackgroundMode::equirectangular(hdr_handle, 1.0));

// 4. Cubemap
scene.background.set_mode(BackgroundMode::cubemap(cube_tex, 1.0));

// 5. Planar image
scene.background.set_mode(BackgroundMode::planar(image_tex, 1.0));
```

**Tip**: For photorealistic scenes, use the same HDR texture for both environment lighting and the background:

```rust
let hdr = engine.assets.load_hdr_texture("environment.hdr")?;
scene.environment.set_env_map(Some(hdr));
scene.background.set_mode(BackgroundMode::equirectangular(hdr, 1.0));
```

---

## Loading 3D Models

### glTF Loading

Myth has comprehensive glTF 2.0 support, including PBR materials, skeletal animations, morph targets, and scene hierarchy:

```rust
use myth::assets::GltfLoader;

// Load from file
let prefab = GltfLoader::load(
    std::path::Path::new("models/character.glb"),
    engine.assets.clone(),
)?;

// Instantiate into scene
let root = scene.instantiate(&prefab);

// Position the model
scene.node(&root).set_position(0.0, 0.0, 0.0).set_scale(1.0);

// Play animations if available
scene.play_if_any_animation(root);
```

### HTTP/Network Loading

Load models from URLs (useful for web applications):

```rust
let url = "https://example.com/models/scene.glb";
let prefab = GltfLoader::load_sync(url, engine.assets.clone())?;
scene.instantiate(&prefab);
```

For WASM, use async loading:

```rust
let prefab = GltfLoader::load_async(url, engine.assets.clone()).await?;
```

---

## Animation

### Playing Animations

glTF models with animations automatically get an `AnimationMixer`:

```rust
// Play the first available animation
scene.play_if_any_animation(root_node);

// Play a specific animation by name
scene.play_animation(root_node, "Walk");

// Or access the mixer directly
if let Some(mixer) = scene.animation_mixers.get_mut(root_node) {
    // List all available animations
    for name in mixer.list_animations() {
        println!("Animation: {}", name);
    }
    mixer.play("Run");
}
```

### Animation Control

Use `ActionControl` for fine-grained playback control:

```rust
if let Some(mixer) = scene.animation_mixers.get_mut(node) {
    // Configure an animation
    mixer.action("Walk").unwrap()
        .play()
        .set_loop_mode(LoopMode::Repeat)  // Loop forever
        .set_time_scale(1.5)               // 1.5x speed
        .set_weight(1.0);                  // Full influence

    // Pause/resume
    mixer.action("Walk").unwrap().pause();
    mixer.action("Walk").unwrap().resume();

    // Stop all animations
    mixer.stop_all();
}
```

**Loop Modes**:
- `LoopMode::Once` — Play once and stop
- `LoopMode::Repeat` — Loop indefinitely
- `LoopMode::PingPong` — Alternate forward/backward

### Morph Targets

Morph targets (blend shapes) are automatically loaded from glTF. They're driven by animations or can be set manually:

```rust
// Set weights directly
scene.set_morph_weights(node, vec![0.5, 0.3, 0.0]);

// Check morph target info
if let Some(mesh) = scene.meshes.get(node) {
    if let Some(geo) = engine.assets.geometries.get(mesh.geometry) {
        if geo.has_morph_targets() {
            println!("Morph target count: {}", geo.morph_target_count);
        }
    }
}
```

---

## Post-Processing Effects

### Render Paths

Myth offers two render paths:

| Path | Description |
|------|-------------|
| **`HighFidelity`** (default) | HDR rendering + full post-processing chain |
| **`BasicForward`** | Simple LDR forward rendering with hardware MSAA |

```rust
// Use HighFidelity for best quality (default)
App::new()
    .with_settings(RendererSettings {
        path: RenderPath::HighFidelity,
        ..Default::default()
    })
    .run::<MyApp>()?;

// Switch at runtime
engine.renderer.set_render_path(RenderPath::BasicForward { msaa_samples: 4 });
```

Post-processing is only available in `HighFidelity` mode.

### Bloom

Physically-based bloom for bright emissive surfaces and highlights:

```rust
scene.bloom.set_enabled(true);
scene.bloom.set_strength(0.04);      // Intensity (0–1)
scene.bloom.set_radius(0.005);       // Blur spread
scene.bloom.set_karis_average(true); // Suppress firefly artifacts
```

### Tone Mapping & Color Grading

Convert HDR values to displayable LDR with optional cinematic effects:

```rust
// Tone mapping operator
scene.tone_mapping.set_mode(ToneMappingMode::ACESFilmic);
scene.tone_mapping.set_exposure(1.2);

// Color adjustments
scene.tone_mapping.set_contrast(1.1);
scene.tone_mapping.set_saturation(1.05);

// Cinematic effects
scene.tone_mapping.set_chromatic_aberration(0.002);
scene.tone_mapping.set_film_grain(0.02);
scene.tone_mapping.set_vignette_intensity(0.3);

// LUT-based color grading
scene.tone_mapping.set_lut_texture(Some(lut_handle));
```

**Available tone mapping modes**: `Linear`, `Neutral` (default), `Reinhard`, `Cineon`, `ACESFilmic`, `AgX`

### SSAO

Screen Space Ambient Occlusion adds subtle contact shadows:

```rust
scene.ssao.set_enabled(true);
scene.ssao.set_radius(0.5);      // Affect radius in world units
scene.ssao.set_intensity(1.0);   // Effect strength
scene.ssao.set_sample_count(32); // Quality (1–64)
```

### Anti-Aliasing

Two AA methods depending on render path:

```rust
// HighFidelity: FXAA (post-process)
scene.fxaa.set_enabled(true);
scene.fxaa.set_quality(FxaaQuality::High);

// BasicForward: Hardware MSAA
App::new()
    .with_settings(RendererSettings {
        path: RenderPath::BasicForward { msaa_samples: 4 },
        ..Default::default()
    })
    .run::<MyApp>()?;
```

---

## Input Handling

Access input state through `engine.input`:

```rust
fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
    let input = &engine.input;

    // Keyboard
    if input.get_key(Key::W) {
        // W is currently held down
    }
    if input.get_key_down(Key::Space) {
        // Space was just pressed this frame
    }

    // Mouse
    if input.get_mouse_button(MouseButton::Left) {
        let delta = input.mouse_delta();
        // Process mouse drag
    }

    // Scroll wheel
    let scroll = input.scroll_delta();
}
```

**Key distinction**: `get_key()` returns `true` for every frame the key is held. `get_key_down()` returns `true` only on the frame the key is first pressed. `get_key_up()` returns `true` only on the frame of release.

---

## Camera Controls

`OrbitControls` provides a ready-to-use camera controller:

```rust
struct MyApp {
    controls: OrbitControls,
}

impl AppHandler for MyApp {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();
        // ... setup scene ...

        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0/9.0, 0.1));
        scene.node(&cam).set_position(0.0, 5.0, 10.0).look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);

        MyApp {
            controls: OrbitControls::new(
                Vec3::new(0.0, 5.0, 10.0), // Camera position
                Vec3::ZERO,                  // Target point
            ),
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        let scene = engine.scene_manager.active_scene_mut().unwrap();
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls.update(transform, &engine.input, camera.fov, frame.dt);
        }
    }
}
```

**Controls**: Left-drag to orbit, right-drag to pan, scroll to zoom.

Auto-fit the camera to frame a loaded model:

```rust
let root = scene.instantiate(&prefab);
self.controls.fit(scene, root); // Adjusts distance based on bounding box
```

---

## Custom Render Passes

For advanced rendering, implement the `RenderNode` trait:

```rust
use myth::render::{RenderNode, RenderStage, FrameComposer};
use myth::render::core::{PrepareContext, ExecuteContext};

struct UiOverlay {
    // Your GPU resources, bind groups, pipelines...
}

impl RenderNode for UiOverlay {
    fn name(&self) -> &str { "UiOverlay" }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // Phase 1: Allocate GPU resources, compile shaders, etc.
        // This is the MUTABLE phase — you can access device, queue,
        // create buffers, bind groups, and pipelines here.
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        // Phase 2: Record GPU commands
        // This is READ-ONLY — no allocations.
        // Use ctx.surface_view for the final output texture.
    }
}

// Register in compose_frame
impl AppHandler for MyApp {
    fn compose_frame(&mut self, composer: FrameComposer<'_>) {
        composer
            .add_node(RenderStage::UI, &mut self.ui_overlay)
            .render();
    }
}
```

**Render stages** execute in order: PreProcess → ShadowMap → Opaque → Skybox → BeforeTransparent → Transparent → PostProcess → **UI**. Choose the appropriate stage for your custom pass.

---

## Building for WebAssembly

Myth supports WASM with WebGPU for running in modern browsers.

### Build Steps

```bash
# Install the WASM target (one-time)
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli

# Build (use the provided scripts)
scripts/build_wasm.bat gltf_viewer        # Windows
./scripts/build_wasm.sh gltf_viewer       # Unix

# Serve locally
python -m http.server 8080 --directory examples/gltf_viewer/web
# Then open http://localhost:8080 in Chrome/Edge 113+
```

### WASM-Specific Code

```rust
// Set the HTML canvas element ID
App::new()
    .with_canvas_id("my-canvas")
    .run::<MyApp>()?;

// Use async loading (required on WASM — no blocking I/O)
let tex = assets.load_texture_async("textures/diffuse.png", ColorSpace::Srgb, true).await?;
let hdr = assets.load_hdr_texture_async("env.hdr").await?;
```

Platform-specific code is gated with `#[cfg(target_arch = "wasm32")]`. The engine handles platform differences internally — most code works unchanged across native and web.

---

## Best Practices

### Performance

1. **Use `--release` builds** for GPU workloads — debug builds are significantly slower
2. **Prefer `HighFidelity` path** for complex scenes with many post-processing effects; switch to `BasicForward` for simpler scenes or mobile
3. **Minimize per-frame allocations** in `update()` and `SceneLogic::update()`
4. **Disable unneeded features**: `scene.ssao.set_enabled(false)` if not needed
5. **Enable VSync** (`vsync: true`) to save power when not benchmarking

### Resource Management

1. **Handles are cheap** — store `NodeHandle`, `MaterialHandle`, etc. freely (they're 8-byte `Copy` types)
2. **AssetServer is thread-safe** — `engine.assets.clone()` is just an `Arc` clone
3. **Use UUID dedup** for shared resources: `assets.textures.add_with_uuid(uuid, texture)`
4. **Color space matters**: Use `ColorSpace::Srgb` for color textures, `ColorSpace::Linear` for data textures (normals, roughness, AO)

### Architecture

1. **Keep `init()` focused**: Load assets, create the scene, set up cameras and lights
2. **Use `on_update()` or `SceneLogic`** for per-frame scene behavior instead of putting everything in `AppHandler::update()`
3. **Use the chainable `node()` API** for setup: `scene.node(&h).set_position(1,2,3).look_at(Vec3::ZERO)`
4. **Use `query_main_camera_bundle()`** to get the active camera's transform for orbit controls

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Black screen | Check that `scene.active_camera` is set and the camera can see the objects |
| No lighting | Add at least one light: `scene.add_light(Light::new_directional(Vec3::ONE, 3.0))` |
| Texture appears dark | Ensure correct `ColorSpace` — use `Srgb` for color textures |
| Normal map looks wrong | Use `ColorSpace::Linear` for normal maps |
| Objects invisible | Check `node.visible = true` and that the object is within the camera frustum |
| WASM build fails | Ensure `wasm32-unknown-unknown` target is installed; use async loading APIs |
| No shadows | Set `light.cast_shadows = true` and verify shadow map configuration |
| Post-processing not working | Ensure using `RenderPath::HighFidelity` (default) |
| Low FPS | Use `--release` builds; reduce `msaa_samples` or `ssao.sample_count` |
| Model too large/small | Use `OrbitControls::fit()` or manually adjust camera position/FOV |

### Logging

Enable verbose logging to diagnose issues:

```rust
// In main()
env_logger::init();

// Run with debug logging
// RUST_LOG=myth=debug cargo run --example my_app
```

### Getting Help

- **Examples**: Check the `examples/` directory for working code covering all features
- **API Reference**: See `docs/API.md` for the complete API surface
- **Online Demo**: [Live glTF Viewer](https://panxinmiao.github.io/myth/gltf_viewer/) — test models in your browser
