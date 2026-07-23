# Scene & Node System

## 1. Mental Model

Understanding Myth comes down to grasping its four cooperating layers. From top to bottom, they form the engine's entire runtime flow:

1. **App**: Handles window creation, context management, and the event loop.
2. **Engine**: The core carrier, holding the Renderer, Scene Manager, Asset Server, and Input State.
3. **Scene**: Holds Nodes, Cameras, Lights, environment configuration, and animation mixers.
4. **Render Graph**: At the lowest level, it dynamically assembles and compiles the rendering pipeline every frame based on the topology of the currently active scene.

## 2. Nodes & Hierarchy

### Spawn Helpers

For prototyping or writing examples, the helper methods from `SceneExt` are the most concise way to build content. They automatically register geometry and materials in the resource manager:

```rust
let cube = scene.spawn_box(1.0, 1.0, 1.0, material, &engine.assets);
let sphere = scene.spawn_sphere(0.5, material, &engine.assets);
let ground = scene.spawn_plane(20.0, 20.0, material, &engine.assets);
```

### Transforms & Parent-Child Hierarchy

Myth manages the scene graph through entity handles. You can easily bind parent-child relationships and manipulate transforms via a fluent interface:

```rust
let parent = scene.spawn_box(1.0, 1.0, 1.0, material, &engine.assets);
let child = scene.spawn_sphere(0.35, material, &engine.assets);

// Establish parent-child relationship
scene.attach(child, parent);

// Chainable transform setup
scene.node(&parent).set_position(0.0, 1.0, 0.0);
scene.node(&child).set_position(1.0, 0.0, 0.0);
```

::: info Two Ways to Access a Node

* `scene.node(&handle)`: Provides a chainable, convenient API (e.g. `set_position`, `look_at`), ideal for quick initialization.
* `scene.get_node_mut(handle)`: Returns a mutable reference to the node, ideal for complex logic and low-level component access inside `on_update` loops.
:::

## 3. Camera & Environment Control

### Camera

The renderer skips scenes that have no active camera. A typical setup:

```rust
let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
scene.node(&camera).set_position(0.0, 2.0, 5.0).look_at(Vec3::ZERO);

// Be sure to mark it as the active camera
scene.active_camera = Some(camera);
```

### Lights

A Myth scene typically blends one or more analytic lights with Image-Based Lighting (IBL).

Directional lights always take the global-lighting path; point and spot lights are managed by the RDG and routed into the high-performance **Clustered Lighting** track:

```rust
// Global directional light, casting the main shadow
scene.add_light(Light::new_directional(Vec3::ONE, 5.0));

// Local point light, participating in clustered light culling
scene.add_light(Light::new_point(Vec3::new(1.0, 0.8, 0.6), 100.0, 0.0));
```

## 4. Manual Resource Lifetime

If you need extremely tight control over asset reuse, you can bypass the spawn helpers and interact directly with the engine's resource pools:

```rust
// 1. Manually add geometry and material resources
let geo = engine.assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
let mat = engine.assets.materials.add(PhysicalMaterial::default());

// 2. Instantiate a mesh node from the resource handles
let node = scene.add_mesh(Mesh::new(geo, mat));
```

## Next Steps

- Load models and play animations → [Assets, glTF & Animation](/guide/assets-animation)
- Apply realistic materials → [PBR Materials](/advanced/pbr-materials)
