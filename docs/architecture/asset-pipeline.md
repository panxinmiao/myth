# Async Asset Pipeline

For a modern 3D engine, blocking the main thread to load hundred-megabyte model assets is unacceptable. Myth builds a robust **asynchronous Asset Server** that keeps the render loop perfectly smooth.

## 1. Acquiring & Managing Assets

The engine's resources are managed uniformly by `engine.assets`, which owns the lifetime of textures, models, materials, and meshes.

### Loading Basic Resources
Loading most local or networked resources is very straightforward:

```rust
// Load a texture asynchronously
let albedo = engine.assets.load_texture("assets/uv_grid.png", ColorSpace::Srgb, true);

// Load an HDR environment map asynchronously
let env = engine.assets.load_texture("assets/studio.hdr.jpg", ColorSpace::Srgb, false);
scene.environment.set_env_map(Some(env));
```

### glTF Scene Prefabs

For complex glTF or GLB files, the engine parses the node tree, binds skins, and automatically installs animation mixers:

```rust
let model_prefab = engine.assets.load_gltf("assets/model.glb");
// Instantiate once the prefab is ready (see timing below)
let root_node = scene.instantiate(prefab.as_ref());
```

## 2. Critical Rules for Async Timing

Myth's loading pipeline is **fully asynchronous**. This means that when you call `scene.instantiate()`, the scene node tree is created immediately, but the underlying GPU resources (vertex buffers, texture data) may still be queued for background upload.

::: warning ⚠️ Important Pitfall: Beware of "First Few Frames" Timing
Because model loading is asynchronous, **it is normal for a model to not be fully loaded into the scene during the first few frames after startup.**

If your low-level logic strictly depends on geometry data existing immediately, you must explicitly handle this async timing. 

This timing mismatch often doesn't panic outright; instead it manifests as extremely subtle rendering bugs (sudden flickering, shadow tearing). Best practice: in your `on_update` loop, always confirm resource readiness via `Option` or asset state checks before running logic that depends on the asset being fully loaded.
:::

## 3. Practical Guide

For the full workflow of loading glTF, instantiating, playing animations, and handling readiness checks, see the guide chapter [Assets, glTF & Animation](/guide/assets-animation).
