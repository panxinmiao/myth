# Myth Engine API Reference

A high-performance 3D rendering engine built with Rust and wgpu.

## Quick Start

```rust
use myth::prelude::*;

struct MyApp;

impl AppHandler for MyApp {
    fn init(engine: &mut Engine, _window: &Arc<Window>) -> Self {
        // Create an active scene
        let scene = engine.scene_manager.create_active();
        
        // Add a mesh
        let geometry = Geometry::new_box(1.0, 1.0, 1.0);
        let material = Material::new_basic(Vec4::new(1.0, 0.5, 0.2, 1.0));
        let mesh = Mesh::new(
            engine.assets.geometries.add(geometry),
            engine.assets.materials.add(material),
        );
        scene.add_mesh(mesh);
        
        // Setup camera
        let camera = Camera::new_perspective(60.0, 16.0/9.0, 0.1);
        let cam_node = scene.add_camera(camera);
        scene.active_camera = Some(cam_node);
        
        MyApp
    }
    
    fn update(&mut self, _: &mut Engine, _: &Arc<Window>, _: &FrameState) {
        // Update logic here
    }
}

fn main() -> myth::errors::Result<()> {
    App::new().with_title("My 3D App").run::<MyApp>()
}
```

## Module Overview

### Import Patterns

```rust
// Recommended: Use prelude for common types
use myth::prelude::*;

// Alternative: Import specific modules
use myth::scene::{Scene, Camera, Light};
use myth::resources::{Geometry, Material, Mesh};
use myth::math::{Vec3, Quat, Mat4};
```

### Module Hierarchy

| Module | Description |
|--------|-------------|
| `myth::prelude` | Common imports for everyday use |
| `myth::app` | Application lifecycle and windowing |
| `myth::engine` | Core engine instance |
| `myth::scene` | Scene graph (nodes, cameras, lights) |
| `myth::resources` | Resource definitions (geometry, material, texture) |
| `myth::assets` | Asset loading and management |
| `myth::animation` | Skeletal and morph target animations |
| `myth::math` | Math types (re-exported from glam) |
| `myth::render` | Rendering configuration and advanced APIs |

---

## Core Types

### Engine

The central engine instance that orchestrates all subsystems.

```rust
pub struct Engine {
    pub renderer: Renderer,
    pub scene_manager: SceneManager,
    pub assets: AssetServer,
    pub input: Input,
}
```

**Key Methods:**

```rust
// Create with default settings
let engine = Engine::default();

// Create with custom settings
let engine = Engine::new(RendererSettings {
    vsync: false,
    ..Default::default()
});

// Initialize GPU (called automatically by App)
engine.init(window, width, height).await?;

// Per-frame update
engine.update(dt);

// Handle window resize
engine.resize(width, height, scale_factor);
```

### Scene

Container for all scene objects using a component-based architecture.

```rust
let mut scene = Scene::new();

// Create nodes
let node = scene.create_node_with_name("MyNode");

// Add components
scene.set_mesh(node, mesh);
scene.set_camera(node, camera);
scene.set_light(node, light);

// Hierarchy management
let child = scene.create_node_with_name("Child");
scene.attach(child, node);  // Make child a child of node

// Query nodes
if let Some(node) = scene.get_node_mut(handle) {
    node.transform.position = Vec3::new(1.0, 2.0, 3.0);
}

// Convenience methods
let mesh_node = scene.add_mesh(mesh);
let camera_node = scene.add_camera(camera);
let light_node = scene.add_light(light);
```

### Node & Transform

Scene nodes with hierarchical transforms.

```rust
// Transform properties
node.transform.position = Vec3::new(1.0, 2.0, 3.0);
node.transform.rotation = Quat::from_rotation_y(PI / 4.0);
node.transform.scale = Vec3::splat(2.0);

// Euler angles helper
node.transform.set_rotation_euler(0.0, PI / 2.0, 0.0);

// Look-at helper
node.transform.look_at(target_position, Vec3::Y);

// Visibility
node.visible = false;
```

### NodeHandle

Lightweight handle for referencing nodes safely.

```rust
// Handles are Copy and can be freely shared
let handle: NodeHandle = scene.create_node();
let handle_copy = handle;

// Check if handle is still valid
if scene.get_node(handle).is_some() {
    // Node exists
}

// Remove node (invalidates handle)
scene.remove_node(handle);
```

---

## Resources

### Geometry

Vertex data containers.

```rust
// Built-in primitives
let box_geo = Geometry::new_box(width, height, depth);
let sphere_geo = Geometry::new_sphere(radius, segments_w, segments_h);
let plane_geo = Geometry::new_plane(width, height, segments_w, segments_h);
let cylinder_geo = Geometry::new_cylinder(radius_top, radius_bottom, height, segments);

// Custom geometry
let mut geometry = Geometry::new();
geometry.set_attribute("position", Attribute::new_planar(&positions, VertexFormat::Float32x3));
geometry.set_attribute("normal", Attribute::new_planar(&normals, VertexFormat::Float32x3));
geometry.set_attribute("uv", Attribute::new_planar(&uvs, VertexFormat::Float32x2));
geometry.set_indices_u16(&indices);
```

### Materials

Surface appearance definitions.

```rust
// Basic (unlit)
let basic = Material::new_basic(Vec4::new(1.0, 0.0, 0.0, 1.0));

// Standard PBR (metallic-roughness workflow)
let standard = MeshStandardMaterial::new()
    .with_base_color(Vec4::new(0.8, 0.8, 0.8, 1.0))
    .with_metallic(0.0)
    .with_roughness(0.5);

// Phong (classic shading)
let phong = MeshPhongMaterial::new()
    .with_diffuse(Vec4::new(0.8, 0.2, 0.2, 1.0))
    .with_shininess(32.0);

// Physical (advanced PBR)
let physical = MeshPhysicalMaterial::new()
    .with_base_color(Vec4::new(1.0, 1.0, 1.0, 1.0))
    .with_clearcoat(1.0)
    .with_clearcoat_roughness(0.1);

// Material settings
material.settings.side = Side::Double;  // Render both faces
material.settings.alpha_mode = AlphaMode::Blend;  // Enable transparency
```

### Mesh

Combines geometry and material.

```rust
// Create mesh from handles
let mesh = Mesh::new(geometry_handle, material_handle);

// Add to scene
let node = scene.add_mesh(mesh);

// Access mesh component
if let Some(mesh) = scene.get_mesh_mut(node) {
    mesh.visible = true;
    mesh.render_order = 10;  // Higher = rendered later
}
```

### Texture

Image data with sampling configuration.

```rust
// Load from file (via AssetServer)
let tex_handle = assets.load_texture_from_file(
    "diffuse.png",
    ColorSpace::Srgb,  // or ColorSpace::Linear
    true,  // generate mipmaps
)?;

// Apply to material
let mut material = MeshStandardMaterial::new();
material.set_base_color_map(tex_handle);
material.set_normal_map(normal_handle);
material.set_metallic_roughness_map(mr_handle);
```

---

## Camera & Lights

### Camera

```rust
// Perspective camera (most common)
let camera = Camera::new_perspective(
    fov_degrees,  // Field of view (e.g., 60.0)
    aspect_ratio, // Width / Height
    near_plane,   // Near clipping plane
);

// Orthographic camera
let camera = Camera::new_orthographic(size, aspect, near, far);

// Set as active
scene.active_camera = Some(camera_node);
```

### Light Types

```rust
// Directional light (sun-like)
let sun = Light::new_directional(
    Vec3::ONE,  // color (white)
    1.0,        // intensity
);

// Point light (bulb-like)
let bulb = Light::new_point(
    Vec3::new(1.0, 0.9, 0.8),  // warm white
    100.0,  // intensity (candela)
    10.0,   // range
);

// Spot light (flashlight-like)
let spot = Light::new_spot(
    Vec3::ONE,  // color
    100.0,      // intensity
    10.0,       // range
    0.5,        // inner cone angle (radians)
    0.7,        // outer cone angle (radians)
);
```

---

## Asset Loading

### AssetServer

Central registry for all assets.

```rust
// Access via engine
let assets = &mut engine.assets;

// Add resources and get handles
let geo_handle = assets.geometries.add(geometry);
let mat_handle = assets.materials.add(material);
let tex_handle = assets.textures.add(texture);

// Get resources by handle
let geometry = assets.geometries.get(geo_handle);
let material = assets.materials.get_mut(mat_handle);
```

### glTF Loading

```rust
use myth::assets::GltfLoader;

// Load glTF/GLB file
let prefab = GltfLoader::load(&assets, "model.gltf").await?;

// Instantiate into scene
let root_node = scene.instantiate(&prefab);

// Access animations (if any)
if let Some(mixer) = scene.animation_mixers.get_mut(root_node) {
    mixer.play("Walk");
}
```

### Environment Maps

```rust
// Load HDR environment
let hdr_handle = assets.load_hdr_texture("environment.hdr")?;

// Set as scene environment
scene.environment.set_env_map(Some((hdr_handle.into(), &texture)));
scene.environment.set_intensity(1.0);
```

---

## Animation

### Animation System

```rust
// Get mixer from node (automatically created for glTF models with animations)
if let Some(mixer) = scene.animation_mixers.get_mut(node) {
    // Play by name
    mixer.play("Walk");
    
    // Control playback
    mixer.action("Walk")?.set_loop(LoopMode::Repeat);
    mixer.action("Walk")?.set_time_scale(1.5);
    
    // Crossfade between animations
    mixer.crossfade("Walk", "Run", 0.3);
}
```

### Creating Custom Animations

```rust
use myth::animation::{AnimationClip, Track, TrackData};

let clip = AnimationClip::new("bounce")
    .with_track(Track {
        target_path: "position".into(),
        data: TrackData::Vector3 {
            times: vec![0.0, 0.5, 1.0],
            values: vec![Vec3::ZERO, Vec3::Y, Vec3::ZERO],
        },
        interpolation: InterpolationMode::Linear,
    });
```

---

## Input Handling

```rust
// In update callback
fn update(&mut self, engine: &mut Engine, _: &Arc<Window>, _: &FrameState) {
    let input = &engine.input;
    
    // Keyboard
    if input.get_key(Key::Space) {
        // Space is held
    }
    if input.key_just_pressed(Key::Escape) {
        // Escape was just pressed this frame
    }
    
    // Mouse
    if input.get_mouse_button(MouseButton::Left) {
        let delta = input.mouse_delta();
        let position = input.mouse_position();
    }
    
    // Scroll
    let scroll = input.scroll_delta();
    
    // Screen size
    let size = input.screen_size();
}
```

---

## Render Configuration

### RendererSettings

```rust
use myth::render::{RendererSettings, RenderPath};

let settings = RendererSettings {
    power_preference: wgpu::PowerPreference::HighPerformance,
    vsync: true,
    clear_color: wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
    depth_format: wgpu::TextureFormat::Depth32Float,
    ..Default::default()
};

App::new()
    .with_settings(settings)
    .run::<MyApp>()?;
```

### Custom Render Passes

```rust
use myth::render::{FrameComposer, RenderStage, RenderNode};

impl AppHandler for MyApp {
    fn compose_frame<'a>(&'a self, composer: FrameComposer<'a>) {
        composer
            .add_node(RenderStage::UI, &self.ui_pass)
            .add_node(RenderStage::PostProcess, &self.bloom_pass)
            .render();
    }
}
```

---

## Utilities

### OrbitControls

Camera orbit controller for interactive viewing.

```rust
use myth::OrbitControls;

// Create controller
let orbit = OrbitControls::new(
    Vec3::new(0.0, 5.0, 10.0),  // camera position
    Vec3::ZERO,                   // target
);

// Customize
orbit.enable_damping = true;
orbit.rotate_speed = 0.5;
orbit.zoom_speed = 1.0;
orbit.min_distance = 2.0;
orbit.max_distance = 50.0;

// Update each frame
orbit.update(&mut transform, &input, camera.fov, dt);
```

### Geometry Primitives

Built-in functions to create common geometry shapes.

```rust
use myth::{create_box, create_sphere, SphereOptions, create_plane, PlaneOptions};

// Simple box
let box_geo = create_box(1.0, 1.0, 1.0);

// Sphere with default options
let sphere_geo = create_sphere(1.0, &SphereOptions::default());

// Sphere with custom segments
let smooth_sphere = create_sphere(1.0, &SphereOptions {
    width_segments: 64,
    height_segments: 32,
});

// Plane
let floor = create_plane(&PlaneOptions {
    width: 10.0,
    height: 10.0,
    width_segments: 1,
    height_segments: 1,
});
```

---

## Advanced API

For users who need deeper control over the rendering system.

### Custom Materials

Implement `MaterialTrait` and `RenderableMaterialTrait` for custom materials:

```rust
use myth::{MaterialTrait, RenderableMaterialTrait, ShaderDefines, TextureSlot};

pub struct MyCustomMaterial {
    pub color: Vec4,
    // ...
}

impl MaterialTrait for MyCustomMaterial {
    fn material_type(&self) -> &'static str { "custom" }
    fn alpha_mode(&self) -> AlphaMode { AlphaMode::Opaque }
    fn side(&self) -> Side { Side::Front }
    // ...
}

impl RenderableMaterialTrait for MyCustomMaterial {
    fn shader_defines(&self) -> ShaderDefines {
        let mut defines = ShaderDefines::new();
        defines.set("CUSTOM_FEATURE", "1");
        defines
    }
    
    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        // Define GPU bindings
    }
}
```

### Low-Level Render Access

Access wgpu context and resource management:

```rust
use myth::render::core::{WgpuContext, ResourceManager, BindingResource, ResourceBuilder};

// In RenderNode implementation
fn render(&self, ctx: &mut RenderContext) {
    let device = &ctx.wgpu.device;
    let queue = &ctx.wgpu.queue;
    
    // Custom rendering...
}
```

### Module Path Reference

For users who prefer explicit module paths:

```rust
// Application
use myth::app::winit::{App, AppHandler};

// Scene graph
use myth::scene::{Scene, Camera, Light};

// Resources
use myth::resources::{Geometry, Material, Mesh};
use myth::resources::primitives::{create_sphere, SphereOptions};

// Renderer internals
use myth::renderer::core::{BindingResource, ResourceBuilder};
use myth::renderer::graph::{RenderNode, RenderStage};

// Assets
use myth::assets::{GltfLoader, AssetServer};
```

---

## Best Practices

### Handle Management

```rust
// Handles are lightweight and Copy - store them freely
struct MyApp {
    cube_node: NodeHandle,
    material: MaterialHandle,
}

// Always check validity before use
if let Some(node) = scene.get_node_mut(self.cube_node) {
    node.transform.rotation *= Quat::from_rotation_y(dt);
}
```

### Resource Organization

```rust
// Group related resources
let textures = TextureBundle {
    albedo: assets.load_texture("albedo.png", ColorSpace::Srgb, true)?,
    normal: assets.load_texture("normal.png", ColorSpace::Linear, true)?,
    roughness: assets.load_texture("roughness.png", ColorSpace::Linear, true)?,
};
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `winit` | ✅ | Window management via winit |
| `gltf` | ✅ | glTF 2.0 model loading |
| `http` | ✅ | HTTP asset loading (native) |

```toml
[dependencies]
myth = { version = "0.1", default-features = false, features = ["winit", "gltf"] }
```

---

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| Windows | Vulkan/DX12 | ✅ Full support |
| macOS | Metal | ✅ Full support |
| Linux | Vulkan | ✅ Full support |
| Web (WASM) | WebGPU | ✅ Full support |

---

## Examples

Check the `examples/` directory for complete working examples:

- `rotating_cube.rs` - Basic animation
- `box_pbr.rs` - PBR materials
- `helmet_gltf.rs` - glTF loading
- `hdr_env.rs` - HDR environment maps
- `skinning.rs` - Skeletal animation
- `morph_target.rs` - Morph target animation
- `gltf_viewer/` - Full-featured model viewer

Run examples with:

```bash
cargo run --example gltf_viewer --release
```
