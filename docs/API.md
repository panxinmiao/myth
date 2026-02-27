# Myth Engine API Reference

A high-performance 3D rendering engine built with Rust and wgpu, inspired by Three.js.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Module Overview](#module-overview)
- [Application Lifecycle](#application-lifecycle)
- [Engine](#engine)
- [Scene Graph](#scene-graph)
  - [Scene](#scene)
  - [Node & Transform](#node--transform)
  - [SceneNode (Chainable Wrapper)](#scenenode-chainable-wrapper)
  - [NodeBuilder](#nodebuilder)
  - [SceneLogic](#scenelogic)
  - [SceneManager](#scenemanager)
- [Camera](#camera)
- [Lights & Shadows](#lights--shadows)
- [Environment & Background](#environment--background)
- [Resources](#resources)
  - [Geometry](#geometry)
  - [Materials](#materials)
  - [Mesh](#mesh)
  - [Texture & Image](#texture--image)
- [Asset System](#asset-system)
- [Animation](#animation)
- [Rendering](#rendering)
  - [RendererSettings & RenderPath](#renderersettings--renderpath)
  - [Post-Processing](#post-processing)
  - [Custom Render Passes](#custom-render-passes)
  - [Render Graph API](#render-graph-api)
- [Input](#input)
- [Utilities](#utilities)
- [Error Handling](#error-handling)
- [Feature Flags](#feature-flags)
- [Platform Support](#platform-support)
- [Examples](#examples)

---

## Quick Start

```rust
use myth::prelude::*;

struct MyApp;

impl AppHandler for MyApp {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        // Create an active scene
        let scene = engine.scene_manager.create_active();

        // Spawn a cube with PBR material (convenience API)
        let cube = scene.spawn_box(
            1.0, 1.0, 1.0,
            MeshPhysicalMaterial::new(Vec4::new(1.0, 0.5, 0.2, 1.0))
                .with_roughness(0.5)
                .with_metalness(0.0),
        );

        // Add a directional light
        scene.add_light(Light::new_directional(Vec3::ONE, 3.0));

        // Setup camera
        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene.node(&cam).set_position(0.0, 2.0, 5.0).look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);

        // Animate the cube each frame
        scene.on_update(move |scene, _input, dt| {
            if let Some(node) = scene.get_node_mut(cube) {
                node.transform.rotation *= Quat::from_rotation_y(0.5 * dt);
            }
        });

        MyApp
    }

    fn update(&mut self, _engine: &mut Engine, _window: &dyn Window, _frame: &FrameState) {}
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_title("My 3D App")
        .run::<MyApp>()
}
```

---

## Module Overview

### Import Patterns

```rust
// Recommended: Use prelude for common types
use myth::prelude::*;

// Alternative: Import specific modules
use myth::scene::{Scene, Camera, Light};
use myth::resources::{Geometry, Material, Mesh};
use myth::math::{Vec3, Vec4, Quat, Mat4, Affine3A};
use myth::animation::{AnimationClip, AnimationMixer, LoopMode};
```

### Module Hierarchy

| Module | Description |
|--------|-------------|
| `myth::prelude` | Common imports — one-line access to all everyday types |
| `myth::app` | Application lifecycle (`App`, `AppHandler`, `Window`) |
| `myth::engine` | Central engine coordinator (`Engine`, `FrameState`) |
| `myth::scene` | Scene graph: nodes, cameras, lights, transforms, environment, background |
| `myth::resources` | CPU-side resource definitions: geometry, material, texture, mesh |
| `myth::assets` | Async-capable asset loading & management (`AssetServer`, `GltfLoader`) |
| `myth::animation` | Skeletal & morph target animation (`AnimationMixer`, `AnimationAction`) |
| `myth::math` | Math types re-exported from `glam` (`Vec2`–`Vec4`, `Mat4`, `Quat`, `Affine3A`) |
| `myth::render` | High-level rendering API aliases + `core` for low-level GPU access |
| `myth::errors` | Hierarchical error types (`Error`, `AssetError`, `RenderError`, `PlatformError`) |
| `myth::utils` | Utilities: `OrbitControls`, `FpsCounter`, string interning |

### Top-Level Re-exports

The most commonly used types are available directly from `myth::`:

```rust
use myth::{Engine, App, AppHandler, Scene, Mesh, Camera, Light, OrbitControls};
use myth::{Geometry, Material, MeshPhysicalMaterial, MeshPhongMaterial, Texture};
use myth::{NodeHandle, GeometryHandle, MaterialHandle, TextureHandle};
use myth::{create_box, create_sphere, create_plane, SphereOptions, PlaneOptions};
use myth::{FrameComposer, RenderStage, RendererSettings, RenderPath};
use myth::{AnimationClip, AnimationMixer, AnimationAction, LoopMode};
use myth::{Error, Result};
```

---

## Application Lifecycle

### App (Builder)

The entry point for creating and running a Myth application.

```rust
App::new()
    .with_title("My App")                      // Window title
    .with_settings(RendererSettings {           // Renderer configuration
        path: RenderPath::HighFidelity,
        vsync: false,
        ..Default::default()
    })
    .run::<MyApp>()?;                           // Start the event loop
```

| Method | Description |
|--------|-------------|
| `App::new()` | Create a new application builder |
| `.with_title(impl Into<String>)` | Set window title |
| `.with_settings(RendererSettings)` | Configure renderer |
| `.with_canvas_id(impl Into<String>)` | *WASM only:* Set HTML canvas element ID |
| `.run::<H: AppHandler>()` | Start the event loop and run the application |

### AppHandler (Trait)

User-implemented trait that drives the application.

```rust
pub trait AppHandler: Sized + 'static {
    /// Called once after GPU initialization. Create your scene here.
    fn init(engine: &mut Engine, window: &dyn Window) -> Self;

    /// Called every frame. Update game logic here.
    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        // Default: no-op
    }

    /// Called to configure the render pipeline (add custom render passes).
    fn compose_frame(&mut self, composer: FrameComposer<'_>) {
        composer.render(); // Default: render with built-in passes only
    }

    /// Handle raw winit events (advanced: e.g., egui integration).
    /// Return true to consume the event.
    fn on_event(&mut self, engine: &mut Engine, window: &dyn Window, event: &Event) -> bool {
        false
    }
}
```

### Window (Trait)

Platform-agnostic window abstraction.

```rust
pub trait Window {
    fn set_title(&self, title: &str);
    fn inner_size(&self) -> (u32, u32);
    fn scale_factor(&self) -> f64;
    fn request_redraw(&self);
    fn set_cursor_visible(&self, visible: bool);
}
```

### FrameState

Read-only frame timing information passed to `update()`.

```rust
pub struct FrameState {
    pub time: f32,        // Accumulated time (seconds)
    pub dt: f32,          // Delta time since last frame
    pub frame_count: u64, // Total frames rendered
}
```

---

## Engine

Central coordinator holding all subsystems. No window logic — driven by `AppHandler`.

```rust
pub struct Engine {
    pub renderer: Renderer,          // GPU rendering subsystem
    pub scene_manager: SceneManager, // Multi-scene management
    pub assets: AssetServer,         // Global asset registry (Arc internally, cheap to clone)
    pub input: Input,                // Unified input state
    pub time: f32,                   // Accumulated time
    pub frame_count: u64,            // Frame counter
}
```

### Key Methods

```rust
// Create with default settings
let engine = Engine::default();

// Create with custom settings
let engine = Engine::new(RendererSettings { vsync: false, ..Default::default() });

// These are called automatically by App:
engine.init(window, width, height).await?;  // Initialize GPU
engine.update(dt);                          // Per-frame update
engine.resize(width, height, scale_factor); // Handle window resize
engine.maybe_prune();                       // Periodic GPU resource cleanup
```

---

## Scene Graph

### Scene

The primary container for a 3D scene. Uses a **hybrid ECS** architecture: `SlotMap<NodeHandle, Node>` for the core hierarchy + `SparseSecondaryMap` for sparse components.

#### Node Operations

```rust
// Create nodes
let node = scene.create_node();
let named = scene.create_node_with_name("MyNode");

// Hierarchy
scene.attach(child, parent);
scene.remove_node(handle);

// Access nodes
let node_ref = scene.get_node(handle);         // Option<&Node>
let node_mut = scene.get_node_mut(handle);     // Option<&mut Node>
scene.node(&handle).set_position(1.0, 2.0, 3.0); // Chainable wrapper
```

#### Component Operations (ECS-style)

```rust
// Mesh
let mesh_node = scene.add_mesh(mesh);           // Creates node + attaches mesh
scene.set_mesh(existing_node, mesh);            // Attach mesh to existing node

// Camera
let cam_node = scene.add_camera(camera);
scene.set_camera(existing_node, camera);
scene.active_camera = Some(cam_node);           // Set active camera

// Light
let light_node = scene.add_light(light);
scene.set_light(existing_node, light);

// Skeleton
scene.bind_skeleton(node, skin_binding);

// Morph Weights
scene.set_morph_weights(node, vec![0.5, 0.3, 0.0]);
```

#### Convenience Spawn API

One-liner methods that register resources in `AssetServer` and create mesh nodes automatically:

```rust
// Spawn with any geometry + material (auto-registers to AssetServer)
let handle = scene.spawn(geometry, material);

// Built-in primitive spawners
let cube = scene.spawn_box(1.0, 1.0, 1.0, material);
let sphere = scene.spawn_sphere(1.0, material);
let plane = scene.spawn_plane(10.0, 10.0, material);
```

The `material` parameter accepts any type implementing `ResolveMaterial`: `Material`, `MeshPhysicalMaterial`, `MeshPhongMaterial`, `MeshBasicMaterial`, or `MaterialHandle`.

#### Query API

```rust
// Camera bundle (returns mutable transform + read-only camera)
if let Some((transform, camera)) = scene.query_main_camera_bundle() { ... }
if let Some((transform, camera)) = scene.query_camera_bundle(cam_node) { ... }

// Light bundles
for (node_handle, transform, light) in scene.query_light_bundle() { ... }

// Mesh iteration via component map
for (node_handle, mesh) in scene.meshes.iter() { ... }

// Find by name
if let Some(handle) = scene.find_node_by_name("LeftArm") { ... }
```

#### Scene-Level Settings

```rust
// Post-processing (HighFidelity render path only)
scene.bloom.set_enabled(true);
scene.bloom.set_strength(0.04);
scene.tone_mapping.set_exposure(1.5);
scene.fxaa.set_enabled(true);
scene.ssao.set_enabled(true);

// Background
scene.background.set_mode(BackgroundMode::color(0.1, 0.1, 0.15));

// Environment
scene.environment.set_env_map(Some(hdr_handle));
scene.environment.set_intensity(1.0);
scene.environment.set_ambient_light(Vec3::splat(0.01));
```

#### Prefab Instantiation

```rust
let prefab = GltfLoader::load(path, assets.clone())?;
let root = scene.instantiate(&prefab);
```

Performs a 7-step process: create nodes → build hierarchy → rebuild skeletons → bind skins → mount root → bind animations → GC orphans.

#### Per-Frame Update Cycle (internal)

Called automatically by the engine each frame:

1. Scene logic scripts (`SceneLogic::update`)
2. Animation update
3. Transform tree computation
4. Skeleton joint matrix update
5. Morph weight synchronization
6. Shader macro synchronization
7. GPU buffer upload

---

### Node & Transform

Nodes are intentionally minimal for cache efficiency (~70 lines of code):

```rust
pub struct Node {
    pub parent: Option<NodeHandle>,
    pub children: Vec<NodeHandle>,
    pub transform: Transform,
    pub visible: bool,
}
```

All other data (meshes, cameras, lights) lives in component maps on `Scene`.

#### Transform

```rust
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    // Cached matrices (auto-computed)
    pub local_matrix: Affine3A,
    pub world_matrix: Affine3A,
}

// Direct manipulation
node.transform.position = Vec3::new(1.0, 2.0, 3.0);
node.transform.rotation = Quat::from_rotation_y(PI / 4.0);
node.transform.scale = Vec3::splat(2.0);

// Helpers
node.transform.set_rotation_euler(x, y, z);
node.transform.look_at(target, Vec3::Y);
node.transform.mark_dirty(); // Force matrix recomputation next frame
```

Transform uses **dirty tracking**: `local_matrix` is only recomputed when position/rotation/scale actually change.

---

### SceneNode (Chainable Wrapper)

Temporary mutable borrow wrapper. All methods return `Self` for chaining and silently no-op if the handle is invalid.

```rust
scene.node(&handle)
    .set_position(1.0, 2.0, 3.0)
    .set_scale(2.0)
    .set_rotation_euler(0.0, PI / 2.0, 0.0)
    .look_at(Vec3::ZERO)
    .set_visible(true)
    .set_cast_shadows(true)
    .set_receive_shadows(true);
```

| Method | Description |
|--------|-------------|
| `set_position(x, y, z)` | Set world position |
| `set_position_vec(Vec3)` | Set position from vector |
| `set_scale(s)` | Uniform scale |
| `set_scale_xyz(x, y, z)` | Non-uniform scale |
| `set_rotation(Quat)` | Set rotation quaternion |
| `set_rotation_euler(x, y, z)` | Set rotation from Euler angles (radians) |
| `rotate_x(angle)` / `rotate_y(angle)` | Incremental rotation |
| `look_at(target)` | Orient node toward target point |
| `set_visible(bool)` | Toggle visibility |
| `set_cast_shadows(bool)` | Enable/disable shadow casting |
| `set_receive_shadows(bool)` | Enable/disable shadow receiving |
| `set_shadows(cast, receive)` | Set both shadow flags |

---

### NodeBuilder

Builder-pattern API for node creation:

```rust
let handle = scene.build_node("MyNode")
    .with_position(Vec3::new(1.0, 2.0, 3.0))
    .with_scale(Vec3::splat(2.0))
    .with_parent(parent_handle)
    .with_mesh(mesh)
    .build();
```

---

### SceneLogic

Custom per-frame behavior scripts:

```rust
// Trait-based
pub trait SceneLogic: Send + Sync + 'static {
    fn update(&mut self, scene: &mut Scene, input: &Input, dt: f32);
}
scene.add_logic(my_logic);

// Closure-based shorthand
scene.on_update(|scene, input, dt| {
    // Per-frame logic
});
```

---

### SceneManager

Manages multiple scenes:

```rust
let scene = engine.scene_manager.create_active();     // Create + set active
let handle = engine.scene_manager.create_scene();      // Create without activating
engine.scene_manager.set_active(handle);               // Activate existing scene
engine.scene_manager.remove_scene(handle);

let scene = engine.scene_manager.active_scene_mut();   // Option<&mut Scene>
let scene = engine.scene_manager.get_scene_mut(handle); // Option<&mut Scene>
```

---

## Camera

```rust
// Perspective camera (infinite reverse-Z for maximum depth precision)
let camera = Camera::new_perspective(
    fov_degrees,   // Field of view in degrees (e.g., 45.0)
    aspect_ratio,  // Width / Height (e.g., 16.0 / 9.0)
    near_plane,    // Near clipping distance (e.g., 0.1)
);
// Note: Far plane is infinite by default (reverse-Z)

// Key fields (publicly accessible)
camera.fov;             // Field of view in degrees
camera.aspect;          // Aspect ratio
camera.near;            // Near plane distance
camera.far;             // Far plane (f32::INFINITY for perspective)
camera.projection_type; // ProjectionType::Perspective | Orthographic
```

### Setting the Active Camera

```rust
let cam_node = scene.add_camera(camera);
scene.node(&cam_node).set_position(0.0, 5.0, 10.0).look_at(Vec3::ZERO);
scene.active_camera = Some(cam_node);
```

### Frustum Culling

The camera automatically maintains a `Frustum` (6 clip planes) updated each frame, used for efficient view frustum culling. Supports both reverse-Z (main camera) and standard-Z (shadow maps).

### Camera Methods

| Method | Description |
|--------|-------------|
| `Camera::new_perspective(fov, aspect, near)` | Create perspective camera (reverse-Z, infinite far) |
| `fit_to_scene(scene, node_handle)` | Auto-fit camera to a node's bounding box |
| `extract_render_camera()` | Generate lightweight POD `RenderCamera` for the renderer |
| `update_projection_matrix()` | Manually recompute projection matrix |

---

## Lights & Shadows

### Light Types

```rust
// Directional light (sun-like, infinite range)
let sun = Light::new_directional(
    Vec3::ONE,  // color (white)
    3.0,        // intensity
);

// Point light (omni-directional)
let bulb = Light::new_point(
    Vec3::new(1.0, 0.9, 0.8), // warm white
    100.0,                      // intensity (candela)
    10.0,                       // range
);

// Spot light (cone-shaped)
let spot = Light::new_spot(
    Vec3::ONE,  // color
    100.0,      // intensity
    10.0,       // range
    0.5,        // inner cone angle (radians)
    0.7,        // outer cone angle (radians)
);
```

### Light Fields

```rust
pub struct Light {
    pub uuid: Uuid,
    pub id: u64,
    pub color: Vec3,
    pub intensity: f32,
    pub kind: LightKind,          // Directional | Point | Spot
    pub cast_shadows: bool,
    pub shadow: Option<ShadowConfig>,
}
```

### Shadow Configuration

```rust
let mut light = Light::new_directional(Vec3::ONE, 5.0);
light.cast_shadows = true;

// Configure shadow parameters
if let Some(shadow) = light.shadow.as_mut() {
    shadow.map_size = 2048;             // Shadow map resolution (default: 2048)
    shadow.bias = 0.0;                  // Depth bias (default: 0.0)
    shadow.normal_bias = 0.02;          // Normal bias (default: 0.02)
    shadow.cascade_count = 4;           // CSM cascades, 1-4 (default: 4, directional only)
    shadow.cascade_split_lambda = 0.5;  // Cascade split distribution (default: 0.5)
    shadow.max_shadow_distance = 100.0; // Max shadow render distance (default: 100.0)
}

let light_node = scene.add_light(light);
scene.node(&light_node).set_position(0.0, 12.0, 6.0).look_at(Vec3::ZERO);
```

### Per-Node Shadow Settings

```rust
scene.node(&ground_node)
    .set_cast_shadows(false)
    .set_receive_shadows(true);
```

---

## Environment & Background

### Environment (IBL)

Image-Based Lighting with cubemap or equirectangular HDR maps:

```rust
// Load HDR environment
let hdr = engine.assets.load_hdr_texture("environment.hdr")?;
scene.environment.set_env_map(Some(hdr));
scene.environment.set_intensity(1.5);

// Ambient light (added to environmental lighting)
scene.environment.set_ambient_light(Vec3::splat(0.01));

// Rotation (radians)
scene.environment.rotation = 0.5;
```

### Background Settings

```rust
// Solid color (hardware clear, zero draw calls)
scene.background.set_mode(BackgroundMode::color(0.1, 0.1, 0.15));

// Vertical gradient (fullscreen triangle)
scene.background.set_mode(BackgroundMode::gradient(
    Vec4::new(0.05, 0.05, 0.25, 1.0),  // top
    Vec4::new(0.7, 0.45, 0.2, 1.0),    // bottom
));

// Equirectangular HDR panorama
scene.background.set_mode(BackgroundMode::equirectangular(hdr_handle, 1.0));

// Cubemap
scene.background.set_mode(BackgroundMode::cubemap(cube_handle, 1.0));

// Planar (screen-space) texture
scene.background.set_mode(BackgroundMode::planar(tex_handle, 1.0));
```

### BackgroundMapping

| Mode | Description |
|------|-------------|
| `Cube` | Standard cubemap sampling |
| `Equirectangular` | Equirectangular projection (HDR panoramas) |
| `Planar` | Flat screen-space mapping |

---

## Resources

### Geometry

Vertex data containers with built-in primitive factories.

```rust
use myth::{create_box, create_sphere, create_plane, SphereOptions, PlaneOptions};

// Built-in primitives
let box_geo = create_box(1.0, 1.0, 1.0);

let sphere_geo = create_sphere(&SphereOptions {
    radius: 1.0,
    width_segments: 64,   // default: 32 (min: 3)
    height_segments: 32,  // default: 16 (min: 2)
});

let plane_geo = create_plane(&PlaneOptions {
    width: 10.0,          // default: 1.0
    height: 10.0,         // default: 1.0
    width_segments: 1,    // default: 1
    height_segments: 1,   // default: 1
});

// Custom geometry
let mut geometry = Geometry::new();
geometry.set_attribute("position",
    Attribute::new_planar(&positions, VertexFormat::Float32x3));
geometry.set_attribute("normal",
    Attribute::new_planar(&normals, VertexFormat::Float32x3));
geometry.set_attribute("uv",
    Attribute::new_planar(&uvs, VertexFormat::Float32x2));
geometry.set_indices_u16(&indices);
```

#### Attribute Types

| Constructor | Description |
|-------------|-------------|
| `Attribute::new_planar(data, format)` | Non-interleaved (one attribute per buffer) |
| `Attribute::new_interleaved(data, format, offset, stride)` | Multiple attributes sharing a buffer |
| `Attribute::new_instanced(data, format)` | Instance-rate attribute |

#### Geometry Features

- **Bounding volumes**: Auto-computed `BoundingBox` and `BoundingSphere`
- **Morph targets**: Up to 128 morph targets with automatic influence sorting
- **CPU read-back**: `attribute.read_vec3()`, `attribute.read_vec4()`, `attribute.read()`
- **Copy-on-Write updates**: `attribute.update_data()` uses `Arc::make_mut`

---

### Materials

Surface appearance definitions with **static dispatch + dynamic escape** pattern.

#### Material Enum

```rust
// Factory methods
let basic = Material::new_basic(Vec4::new(1.0, 0.0, 0.0, 1.0));
let phong = Material::new_phong(Vec4::new(0.8, 0.8, 0.8, 1.0));
let physical = Material::new_physical(Vec4::new(1.0, 1.0, 1.0, 1.0));
let custom = Material::new_custom(my_custom_material);

// Downcasting
if let Some(pbr) = material.as_physical_mut() { pbr.set_roughness(0.5); }
if let Some(phong) = material.as_phong_mut() { phong.set_shininess(32.0); }
if let Some(basic) = material.as_basic_mut() { basic.set_map(Some(tex)); }
```

#### MeshPhysicalMaterial (Full PBR)

The most feature-rich material. Builder-style API with `#[must_use]` chaining:

```rust
let material = MeshPhysicalMaterial::new(Vec4::new(1.0, 1.0, 1.0, 1.0))
    .with_roughness(0.3)
    .with_metalness(1.0)
    .with_map(albedo_tex)
    .with_normal_map(normal_tex)
    .with_emissive(Vec3::new(1.0, 0.5, 0.0), 2.0)
    .with_side(Side::Double)
    .with_alpha_mode(AlphaMode::Blend);
```

**Base Properties**:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `color` | `Vec4` | White | Base color (RGBA) |
| `roughness` | `f32` | `1.0` | Surface roughness (0 = mirror, 1 = rough) |
| `metalness` | `f32` | `0.0` | Metallic factor |
| `opacity` | `f32` | `1.0` | Overall opacity |
| `emissive` | `Vec3` | Black | Emissive color |
| `emissive_intensity` | `f32` | `1.0` | Emissive strength |
| `normal_scale` | `Vec2` | (1,1) | Normal map intensity |
| `ao_map_intensity` | `f32` | `1.0` | Ambient occlusion strength |
| `ior` | `f32` | `1.5` | Index of refraction |

**Advanced PBR Extensions** (enabled via `PhysicalFeatures` bitflags):

| Feature | Builder Method | Description |
|---------|---------------|-------------|
| **Clearcoat** | `.with_clearcoat(factor, roughness)` | Car paint, varnished wood |
| **Sheen** | `.with_sheen(color, roughness)` | Cloth, velvet |
| **Iridescence** | `.with_iridescence(intensity, ior, thickness_min, thickness_max)` | Soap bubbles, oil films |
| **Anisotropy** | `.with_anisotropy(strength, rotation)` | Brushed metals |
| **Transmission** | `.with_transmission(factor, thickness, att_distance, att_color)` | Glass, water, translucent |
| **Dispersion** | `.with_dispersion(factor)` | Prismatic color separation |
| **SSS** | `.with_sss_id(feature_id)` | Sub-surface scattering (skin, wax) |
| **SSR** | `.with_ssr_id(feature_id)` | Screen-space reflections |

**Texture Slots** (18 total):

| Texture | Builder | Description |
|---------|---------|-------------|
| `map` | `.with_map(h)` | Base color / albedo |
| `normal_map` | `.with_normal_map(h)` | Normal map |
| `roughness_map` | `.with_roughness_map(h)` | Roughness |
| `metalness_map` | `.with_metalness_map(h)` | Metalness |
| `ao_map` | `.with_ao_map(h)` | Ambient occlusion |
| `emissive_map` | `.with_emissive_map(h)` | Emissive |
| `specular_map` | — | Specular color |
| `specular_intensity_map` | — | Specular intensity |
| `clearcoat_map` | — | Clearcoat intensity |
| `clearcoat_roughness_map` | — | Clearcoat roughness |
| `clearcoat_normal_map` | — | Clearcoat normal |
| `sheen_color_map` | — | Sheen color |
| `sheen_roughness_map` | — | Sheen roughness |
| `iridescence_map` | — | Iridescence intensity |
| `iridescence_thickness_map` | — | Iridescence thickness |
| `anisotropy_map` | — | Anisotropy direction |
| `transmission_map` | — | Transmission factor |
| `thickness_map` | — | Thickness for transmission |

**PhysicalFeatures Bitflags**:

```rust
PhysicalFeatures::IBL          // Image-based lighting (default ON)
PhysicalFeatures::SPECULAR     // Specular highlights (default ON)
PhysicalFeatures::IOR          // Index of refraction (default ON)
PhysicalFeatures::CLEARCOAT    // Clear coat layer
PhysicalFeatures::SHEEN        // Sheen layer
PhysicalFeatures::IRIDESCENCE  // Iridescent thin-film
PhysicalFeatures::ANISOTROPY   // Anisotropic highlights
PhysicalFeatures::TRANSMISSION // Dielectric transmission
PhysicalFeatures::DISPERSION   // Chromatic dispersion
PhysicalFeatures::SSS          // Sub-surface scattering
PhysicalFeatures::SSR          // Screen-space reflections
PhysicalFeatures::STANDARD_PBR // = IBL | SPECULAR | IOR (default)
```

**Runtime Feature Control**:

```rust
material.enable_feature(PhysicalFeatures::CLEARCOAT);
material.disable_feature(PhysicalFeatures::SHEEN);
```

#### MeshPhongMaterial (Blinn-Phong)

Classic lighting model, lighter than PBR:

```rust
let mat = MeshPhongMaterial::new(Vec4::new(1.0, 0.76, 0.33, 1.0))
    .with_shininess(32.0)
    .with_specular(Vec3::splat(0.5))
    .with_map(diffuse_tex)
    .with_normal_map(normal_tex)
    .with_emissive(Vec3::new(0.1, 0.1, 0.05), 3.0)
    .with_side(Side::Double);
```

| Property | Type | Default |
|----------|------|---------|
| `color` | `Vec4` | White |
| `specular` | `Vec3` | (0.05, 0.05, 0.05) |
| `shininess` | `f32` | `30.0` |
| `opacity` | `f32` | `1.0` |
| `emissive` / `emissive_intensity` | `Vec3` / `f32` | Black / 1.0 |
| `normal_scale` | `Vec2` | (1, 1) |

Texture slots: `map`, `normal_map`, `specular_map`, `emissive_map`

#### MeshBasicMaterial (Unlit)

No lighting calculations — useful for UI, debug, or flat-shaded objects:

```rust
let mat = MeshBasicMaterial::new(Vec4::new(1.0, 0.0, 0.0, 1.0))
    .with_map(tex_handle)
    .with_side(Side::Double);
```

#### Material Settings (Common)

```rust
// Alpha modes
AlphaMode::Opaque      // Fully opaque (default)
AlphaMode::Mask(0.5, true)  // Alpha test (threshold, double_sided)
AlphaMode::Blend       // Alpha blending

// Face culling
Side::Front            // Back-face culling (default)
Side::Back             // Front-face culling
Side::Double           // No culling (render both faces)

// Depth buffer
material.set_depth_write(false);  // e.g., for transparent overlays
```

---

### Mesh

Combines a geometry and material into a renderable unit:

```rust
let mesh = Mesh::new(geometry_handle, material_handle);

// Shadow settings
mesh.cast_shadows = true;
mesh.receive_shadows = true;

// Render ordering (higher = rendered later)
mesh.render_order = 10;

// Add to scene
let node = scene.add_mesh(mesh);
```

#### Morph Target Support

Up to **128 morph targets** per mesh:

```rust
// Set morph target weights (set via scene API)
scene.set_morph_weights(node, vec![0.5, 0.3, 0.0]);

// Weights are automatically sorted by priority, truncated, and packed
// into Vec4/UVec4 uniforms for GPU consumption
```

---

### Texture & Image

#### Texture

```rust
// Factory methods
Texture::new_2d(name, width, height, data, format);
Texture::new_cube(name, size, data, format);
Texture::create_solid_color(Some("red"), [255, 0, 0, 255]);
Texture::create_checkerboard(Some("checker"), 512, 512, 64);

// Configuration
texture.sampler = TextureSampler { ... };  // Sampling parameters
texture.generate_mipmaps = true;            // Auto mip generation
```

#### TextureSampler (Defaults)

| Parameter | Default |
|-----------|---------|
| `address_mode_u/v/w` | `Repeat` |
| `mag_filter` | `Linear` |
| `min_filter` | `Linear` |
| `mipmap_filter` | `Linear` |
| `anisotropy_clamp` | `1` |

#### TextureTransform

Per-texture-slot UV transformations:

```rust
TextureTransform {
    offset: Vec2::new(0.0, 0.0), // UV offset
    rotation: 0.0,                // UV rotation (radians)
    scale: Vec2::splat(1.0),      // UV scale
}
```

#### Image

Thread-safe (`Arc<ImageInner>`) with atomic version tracking:

```rust
image.update_data(new_bytes);    // Content update (bumps version)
image.resize(1024, 1024);        // Structural change (bumps generation_id)
image.set_format(new_format);    // Format change
```

---

## Asset System

### AssetServer

Thread-safe (`Clone`, all `Arc` internally) central asset registry:

```rust
// Access via engine
let assets = &engine.assets;

// Register resources and get handles
let geo_h = assets.geometries.add(geometry);
let mat_h = assets.materials.add(material);
let tex_h = assets.textures.add(texture);

// Retrieve by handle (returns Arc<T>)
let geo: Option<Arc<Geometry>> = assets.geometries.get(geo_h);

// UUID-based deduplication
let h = assets.geometries.add_with_uuid(uuid, geometry);
let h = assets.geometries.get_handle_by_uuid(uuid);

// Batch access via read lock
let lock = assets.geometries.read_lock();
```

### Texture Loading

```rust
// Synchronous (Native only, uses tokio block_on internally)
let tex = assets.load_texture("path/to/image.png", ColorSpace::Srgb, true)?;
let cube = assets.load_cube_texture(
    ["px.jpg", "nx.jpg", "py.jpg", "ny.jpg", "pz.jpg", "nz.jpg"],
    ColorSpace::Srgb, true,
)?;
let hdr = assets.load_hdr_texture("environment.hdr")?;

// Asynchronous (cross-platform, including WASM)
let tex = assets.load_texture_async("path", ColorSpace::Srgb, true).await?;
let cube = assets.load_cube_texture_async(sources, ColorSpace::Srgb, true).await?;
let hdr = assets.load_hdr_texture_async("env.hdr").await?;
```

`ColorSpace`: `Srgb` (color textures) or `Linear` (normal maps, data textures).

### glTF Loading

```rust
use myth::assets::GltfLoader;

// Synchronous (native, supports file path or HTTP URL)
let prefab = GltfLoader::load(path, engine.assets.clone())?;
let prefab = GltfLoader::load_sync(url, engine.assets.clone())?;

// Instantiate into scene (returns root node handle)
let root = scene.instantiate(&prefab);

// Play animations (if model has any)
scene.play_animation(root, "Walk");
scene.play_if_any_animation(root);  // Play the first available animation
```

### Prefab

Pure data structure (no handles, fully thread-safe) for asset interchange:

```rust
pub struct Prefab {
    pub nodes: Vec<PrefabNode>,
    pub root_indices: Vec<usize>,
    pub skeletons: Vec<PrefabSkeleton>,
    pub animations: Vec<AnimationClip>,
}
pub type SharedPrefab = Arc<Prefab>;
```

---

## Animation

### AnimationMixer

Manages multiple animation actions on a node, supporting blending:

```rust
// Access mixer (automatically created for glTF models with animations)
if let Some(mixer) = scene.animation_mixers.get_mut(node_handle) {
    // List available animations
    let names: Vec<String> = mixer.list_animations();

    // Play by name
    mixer.play("Walk");

    // Stop
    mixer.stop("Walk");
    mixer.stop_all();
}
```

### ActionControl (Chainable)

Fine-grained control over individual animation actions:

```rust
mixer.action("Walk").unwrap()
    .play()
    .set_loop_mode(LoopMode::Repeat)
    .set_time_scale(1.5)
    .set_weight(0.8)
    .set_time(0.0);

// Other methods
.pause()
.resume()
.stop()          // Terminal (does not return Self)
.fade_in(0.3)
```

### LoopMode

```rust
LoopMode::Once       // Play once and stop
LoopMode::Repeat     // Loop forever
LoopMode::PingPong   // Alternate forward/backward
```

### Scene-Level Animation Helpers

```rust
// Play a specific animation on a node
scene.play_animation(root, "Run");

// Play the first available animation (useful for quick testing)
scene.play_if_any_animation(root);
```

### AnimationClip

Data container for animation keyframes:

```rust
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<Track>,
}
```

**TrackData variants**: `Vector3` (position/scale), `Quaternion` (rotation), `Scalar`, `MorphWeights`

**InterpolationMode**: `Linear`, `Step`, `CubicSpline`

**TargetPath**: `Translation`, `Rotation`, `Scale`, `Weights`

---

## Rendering

### RendererSettings & RenderPath

```rust
let settings = RendererSettings {
    path: RenderPath::HighFidelity,                         // Default
    vsync: true,                                             // Default
    clear_color: wgpu::Color::BLACK,                        // Default
    depth_format: wgpu::TextureFormat::Depth24PlusStencil8, // Default
    power_preference: wgpu::PowerPreference::HighPerformance,
    backends: None,       // Auto-detect
    required_features: wgpu::Features::empty(),
    required_limits: wgpu::Limits::default(),
};
```

#### RenderPath

| Path | Features | Use Case |
|------|----------|----------|
| `HighFidelity` | HDR RT, Bloom, ToneMapping, FXAA, SSAO, SSSSS | Full-quality rendering (default) |
| `BasicForward { msaa_samples }` | Hardware MSAA, no post-processing | Lightweight / mobile / simple scenes |

```rust
// Switch at runtime
engine.renderer.set_render_path(RenderPath::HighFidelity);
engine.renderer.set_render_path(RenderPath::BasicForward { msaa_samples: 4 });

// Query capabilities
path.supports_post_processing();  // true for HighFidelity
path.requires_z_prepass();        // true for HighFidelity
path.msaa_samples();              // 1 for HighFidelity, configurable for BasicForward
```

#### Built-in Render Passes (15 total)

| Category | Passes |
|----------|--------|
| Data Preparation | `SceneCullPass`, `ShadowPass` |
| Pre-Pass | `DepthNormalPrepass` |
| LDR Path | `SimpleForwardPass` |
| HDR Path | `OpaquePass`, `TransparentPass`, `TransmissionCopyPass` |
| Skybox | `SkyboxPass` |
| Compute | `BRDFLutComputePass`, `IBLComputePass` |
| Post-Processing | `SssssPass`, `BloomPass`, `ToneMapPass`, `FxaaPass`, `SsaoPass` |

### Post-Processing

All post-processing effects are per-scene settings, only available in `HighFidelity` render path.

#### Bloom

```rust
scene.bloom.set_enabled(true);           // Default: false
scene.bloom.set_strength(0.04);          // Blend intensity (0.0–1.0), default: 0.04
scene.bloom.set_radius(0.005);           // Upsample filter radius, default: 0.005
scene.bloom.set_max_mip_levels(6);       // Max mip levels (1–16), default: 6
scene.bloom.set_karis_average(true);     // Firefly suppression, default: true
```

#### Tone Mapping

```rust
scene.tone_mapping.set_mode(ToneMappingMode::Neutral);  // Default
scene.tone_mapping.set_exposure(1.0);        // default: 1.0
scene.tone_mapping.set_contrast(1.0);        // default: 1.0
scene.tone_mapping.set_saturation(1.0);      // default: 1.0

// Cinematic effects
scene.tone_mapping.set_chromatic_aberration(0.002); // default: 0.0
scene.tone_mapping.set_film_grain(0.03);            // default: 0.0
scene.tone_mapping.set_vignette_intensity(0.3);     // default: 0.0
scene.tone_mapping.set_vignette_smoothness(0.5);    // default: 0.5

// 3D LUT color grading
scene.tone_mapping.set_lut_texture(Some(lut_handle));
scene.tone_mapping.set_lut_contribution(1.0);        // default: 1.0
```

**ToneMappingMode**: `Linear`, `Neutral` (default), `Reinhard`, `Cineon`, `ACESFilmic`, `AgX`

#### FXAA

```rust
scene.fxaa.set_enabled(true);               // Default: true
scene.fxaa.set_quality(FxaaQuality::Medium); // Low(4) / Medium(8) / High(12) iterations
```

#### SSAO

```rust
scene.ssao.set_enabled(true);       // Default: false
scene.ssao.set_radius(0.5);         // Sampling radius (default: 0.5)
scene.ssao.set_bias(0.025);         // Depth bias (default: 0.025)
scene.ssao.set_intensity(1.0);      // AO intensity (default: 1.0)
scene.ssao.set_sample_count(32);    // Hemisphere samples 1–64 (default: 32)
```

#### Screen-Space Effects

```rust
scene.screen_space.enable_sss = true;   // Sub-surface scattering
scene.screen_space.enable_ssr = true;   // Screen-space reflections (reserved)
```

### Custom Render Passes

Implement the `RenderNode` trait to add custom GPU work:

```rust
use myth::render::{RenderNode, RenderStage, FrameComposer};
use myth::render::core::{PrepareContext, ExecuteContext};

struct MyPass { /* ... */ }

impl RenderNode for MyPass {
    fn name(&self) -> &str { "MyPass" }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // Mutable phase: allocate GPU resources, compile pipelines, create bind groups
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        // Read-only phase: record GPU commands (no allocations allowed)
    }
}
```

Register in `compose_frame`:

```rust
impl AppHandler for MyApp {
    fn compose_frame(&mut self, composer: FrameComposer<'_>) {
        composer
            .add_node(RenderStage::UI, &mut self.ui_pass)
            .add_node(RenderStage::PostProcess, &mut self.custom_effect)
            .render();
    }
}
```

### Render Graph API

#### RenderStage (Execution Order)

| Stage | Value | Purpose |
|-------|-------|---------|
| `PreProcess` | 0 | BRDF LUT, IBL prefiltering |
| `ShadowMap` | 1 | Shadow map generation |
| `Opaque` | 2 | Opaque geometry |
| `Skybox` | 3 | Sky rendering |
| `BeforeTransparent` | 4 | Pre-transparent (e.g., transmission copy) |
| `Transparent` | 5 | Alpha-blended geometry |
| `PostProcess` | 6 | Bloom, tone mapping, FXAA |
| `UI` | 7 | UI overlays (egui, debug) |

#### FrameComposer

```rust
composer
    .add_node(RenderStage::UI, &mut ui_pass)       // Single node
    .add_nodes(RenderStage::PostProcess, nodes)      // Batch add
    .render();                                        // Execute pipeline
```

`render()` consumes the composer and executes: acquire surface → sort nodes → **Prepare** → **Execute** → present → recycle transient textures.

#### PrepareContext / ExecuteContext

| Field | PrepareContext | ExecuteContext |
|-------|---------------|----------------|
| `wgpu_ctx` | ✅ | ✅ |
| `resource_manager` | ✅ (mutable) | ✅ (read-only) |
| `pipeline_cache` | ✅ (mutable) | ❌ |
| `assets` | ✅ | ❌ |
| `scene` | ✅ (mutable) | ❌ |
| `camera` | ✅ | ❌ |
| `blackboard` | ✅ (mutable) | ✅ (read-only) |
| `frame_resources` | ✅ | ✅ |
| `transient_pool` | ✅ (mutable) | ✅ (read-only) |
| `render_lists` | ✅ (mutable) | ✅ (read-only) |
| `surface_view` | ❌ | ✅ |

#### Low-Level GPU Access

```rust
let device = engine.renderer.device().unwrap();
let queue = engine.renderer.queue().unwrap();
let format = engine.renderer.surface_format().unwrap();
let ctx = engine.renderer.wgpu_ctx().unwrap();
```

---

## Input

```rust
let input = &engine.input;

// Keyboard
input.get_key(Key::Space)          // Currently held
input.get_key_down(Key::Escape)    // Just pressed this frame
input.get_key_up(Key::Enter)       // Just released this frame

// Mouse
input.get_mouse_button(MouseButton::Left)
input.get_mouse_button_down(MouseButton::Right)
input.get_mouse_button_up(MouseButton::Middle)
input.mouse_position()             // Vec2: current position
input.mouse_delta()                // Vec2: movement since last frame
input.scroll_delta()               // Vec2: scroll wheel delta

// Screen info
input.screen_size()                // Vec2: window dimensions
```

### Key Enum (Partial List)

`A`–`Z`, `Key0`–`Key9`, `F1`–`F12`, `Space`, `Enter`, `Escape`, `Tab`, `Backspace`, `ArrowUp`/`Down`/`Left`/`Right`, `ShiftLeft`/`Right`, `ControlLeft`/`Right`, `AltLeft`/`Right`

### MouseButton

`Left`, `Right`, `Middle`, `Back`, `Forward`, `Other(u16)`

---

## Utilities

### OrbitControls

Interactive camera orbit controller (left-drag rotate, right-drag pan, scroll zoom):

```rust
use myth::OrbitControls;

let mut controls = OrbitControls::new(
    Vec3::new(0.0, 5.0, 10.0), // initial camera position
    Vec3::ZERO,                  // orbit target
);

// Configuration
controls.enable_damping = true;       // default: true
controls.damping_factor = 0.05;       // default: 0.05
controls.zoom_damping_factor = 0.1;   // default: 0.1
controls.enable_zoom = true;          // default: true
controls.zoom_speed = 1.0;            // default: 1.0
controls.enable_rotate = true;        // default: true
controls.rotate_speed = 0.1;          // default: 0.1
controls.enable_pan = true;           // default: true
controls.pan_speed = 1.0;             // default: 1.0
controls.min_distance = 0.5;          // default: 0.0
controls.max_distance = 100.0;        // default: f32::INFINITY
controls.min_polar_angle = 0.0;       // default: 0.0
controls.max_polar_angle = PI;        // default: PI

// Per-frame update
if let Some((transform, camera)) = scene.query_main_camera_bundle() {
    controls.update(transform, &engine.input, camera.fov, frame.dt);
}

// Target / position adjustment
controls.set_target(Vec3::new(0.0, 1.0, 0.0));
controls.set_position(Vec3::new(0.0, 5.0, 10.0));

// Auto-fit to node bounding box
controls.fit(scene, node_handle);
```

### FpsCounter

```rust
use myth::utils::FpsCounter;

let mut fps = FpsCounter::new();

// Returns Some(fps) once per second
if let Some(current_fps) = fps.update() {
    window.set_title(&format!("My App | FPS: {:.0}", current_fps));
}
```

### String Interning

High-performance string interning via `lasso`:

```rust
use myth::utils::interner::Symbol;
// Symbol provides O(1) equality comparison
```

---

## Error Handling

Hierarchical `thiserror` enums with seamless `?` operator chains:

```rust
use myth::errors::{Error, Result};

pub enum Error {
    Platform(PlatformError),  // Window/platform issues
    Asset(AssetError),        // Asset I/O  
    Render(RenderError),      // GPU rendering
    General(String),          // Miscellaneous
}

// Convenience alias
pub type Result<T> = std::result::Result<T, Error>;
```

| Error Type | Variants |
|------------|----------|
| `PlatformError` | `WindowHandle`, `EventLoop`, `AdapterNotFound`, `SurfaceConfigFailed`, `Wasm`, `FeatureNotEnabled` |
| `AssetError` | `NotFound`, `Io`, `Network`, `UrlParse`, `HttpResponse`, `Format`, `InvalidData`, `Base64`, `TaskJoin` |
| `RenderError` | `RequestDeviceFailed`, `ShaderCompile`, `Graph` |

Automatic `From` conversions: `image::ImageError`, `gltf::Error`, `io::Error`, `reqwest::Error` → `Error`

---

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `winit` | ✅ | Window management via winit |
| `gltf` | ✅ | glTF 2.0 model loading |
| `http` | ✅ | HTTP/network asset loading |

```toml
[dependencies]
myth = { git = "https://github.com/panxinmiao/myth", branch = "main" }

# Minimal build
myth = { git = "...", default-features = false, features = ["winit"] }
```

---

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| Windows | Vulkan / DX12 | ✅ Full support |
| macOS | Metal | ✅ Full support |
| Linux | Vulkan | ✅ Full support |
| Web (WASM) | WebGPU | ✅ Full support (Chrome/Edge 113+) |

### WASM-Specific

```rust
App::new()
    .with_canvas_id("my-canvas")
    .run::<MyApp>()?;

// Async asset loading (required on WASM)
let tex = assets.load_texture_async("path", ColorSpace::Srgb, true).await?;
```

---

## Examples

| Example | Features Demonstrated |
|---------|----------------------|
| `hello_triangle.rs` | Custom geometry, basic material, minimal setup |
| `rotating_cube.rs` | Rotation animation, frame timing |
| `box.rs` | Phong material + checkerboard texture |
| `box_pbr.rs` | PBR material, cubemap IBL, `spawn_box()` |
| `box_phong.rs` | Phong material, texture loading |
| `earth.rs` | Multi-texture Phong, transparency, orbit controls |
| `hdr_env.rs` | HDR environment maps, IBL lighting |
| `skybox.rs` | 5 background modes, render path switching |
| `helmet_gltf.rs` | glTF model loading, PBR viewing |
| `shadows.rs` | Cascaded shadow maps, per-node shadow settings |
| `shadow_basic.rs` | Basic shadow setup |
| `shadow_spot.rs` | Spot light shadows |
| `skinning.rs` | Skeletal animation from glTF |
| `morph_target.rs` | Morph target / blend shape animation |
| `bloom.rs` | HDR bloom, tone mapping, interactive controls |
| `sponza.rs` | Large scene, SSAO, HTTP model loading |
| `gltf_viewer/` | Full-featured model viewer with egui inspector |
| `showcase/` | Production web showcase |

```bash
# Native (release recommended for GPU workloads)
cargo run --example earth --release
cargo run --example gltf_viewer --release

# WASM build
scripts/build_wasm.bat gltf_viewer        # Windows
./scripts/build_wasm.sh gltf_viewer       # Unix
python -m http.server 8080 --directory examples/gltf_viewer/web
```
