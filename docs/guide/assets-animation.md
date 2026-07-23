# Assets, glTF & Animation

Myth ships with full **glTF 2.0** loading and an **asynchronous asset system**. This page focuses on real-world workflows: loading models, playing skeletal/morph animations, and handling async timing correctly.

## 1. Loading glTF / GLB Models

glTF resources are loaded via `engine.assets.load_gltf()`. It parses the node tree, meshes, PBR materials, skinned skeletons, animation tracks, and morph targets, returning a **Prefab handle**:

```rust
// Returns a PrefabHandle — loaded asynchronously in the background
let model_prefab = engine.assets.load_gltf("assets/Michelle.glb");
```

::: tip Prefab vs Instance
`load_gltf` returns a **Prefab**, essentially a reusable template. You can instantiate any number of copies of the same Prefab in a scene, and they share the underlying geometry and material resources.
:::

## 2. Instantiation & Timing Checks

Because loading is **fully asynchronous**, the Prefab is not ready immediately after calling `load_gltf`. The correct approach is to poll inside the `update` loop and `instantiate` only once the resource is ready:

```rust
fn update(&mut self, engine: &mut Engine, _window: &dyn Window, _frame: &FrameState) {
    let assets = engine.assets.clone();
    let Some(scene) = engine.scene_manager.active_scene_mut() else { return };

    if !self.model_loaded {
        if let Some(prefab) = assets.prefabs.get(self.model_prefab) {
            // Ready — instantiate into the scene
            let root = scene.instantiate(prefab.as_ref());
            self.model_loaded = true;
            // …start animation here (see below)
        } else if let Some(err) = assets.prefabs.get_error(self.model_prefab) {
            eprintln!("Failed to load model: {err}");
            self.model_loaded = true;
        }
    }
}
```

::: warning ⚠️ Beware of Async Timing Pitfalls
During the first few frames after startup, models may not have finished uploading to the GPU. Be sure to perform readiness checks to avoid subtle bugs like flickering or shadow tearing. See [Async Asset Pipeline](/architecture/asset-pipeline) for details.
:::

## 3. Playing Animations

After instantiation, the engine automatically installs an **Animation Mixer** on nodes that contain animations. Access and control playback via `scene.animation_mixers`:

```rust
let root = scene.instantiate(prefab.as_ref());

if let Some(mixer) = scene.animation_mixers.get_mut(root) {
    // List all animation clips in the model
    for name in mixer.list_animations() {
        println!(" - {name}");
    }

    // Play by name
    mixer.play("SambaDance");
}
```

The mixer supports both skeletal (skinning) and morph-target animation. The engine advances the timeline and updates bone matrices / morph weights every frame automatically — no manual driving required.

## 4. Textures & Environment Maps

Besides models, common asset-loading patterns include:

```rust
// Standard 2D texture (color space, whether to generate mipmaps)
let albedo = engine.assets.load_texture(
    "assets/uv_grid.png",
    ColorSpace::Srgb,
    true,
);

// HDR environment map for Image-Based Lighting (IBL)
let env = engine.assets.load_texture(
    "assets/studio.hdr.jpg",
    ColorSpace::Srgb,
    false,
);
scene.environment.set_env_map(Some(env));
```

Once an environment map is set, the engine automatically prefilters it (PMREM) and uses it as the source of diffuse irradiance and specular reflection. Combined with PBR materials, this yields realistic image-based lighting.

## 5. Procedural Textures

For prototyping, the engine provides convenient procedural texture generators that require no external assets:

```rust
// Checkerboard texture: dimension 512, cell size 64
let checker = engine.assets.checkerboard(512, 64);
```

## Next Steps

- Understand the underlying async pipeline → [Async Asset Pipeline](/architecture/asset-pipeline)
- Customize material appearance → [PBR Materials](/advanced/pbr-materials)
