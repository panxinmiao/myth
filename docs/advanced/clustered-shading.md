# GPU-Driven Clustered Lighting

In complex 3D scenes, as the number of dynamic lights grows, traditional forward rendering degrades sharply in performance, while deferred rendering struggles with transparency and MSAA.

Myth natively adopts a **Clustered Forward Lighting** architecture, combined with a powerful GPU-driven pipeline, to effortlessly handle massive numbers of dynamic lights.

## 1. Principle & Advantages

Clustered lighting divides the camera frustum into many sub-frusta (clusters) in 3D space.
Before rendering each frame, the engine dispatches a compute shader to perform precise **light culling**, recording the indices of lights affecting each cluster in GPU memory.

- **Extreme Performance:** In the final shading stage (fragment shader), each pixel only iterates over the lights in its own cluster, dropping lighting complexity from $O(M \times N)$ to $O(M \times K)$ (where $K \ll N$).
- **Artifact-Free Shading:** The engine uses extremely rigorous bounding-box / frustum intersection tests. This precise math not only improves culling efficiency but fundamentally eliminates the "blocky shadows" or lighting discontinuities common in cluster-based approaches.

## 2. Hybrid Light Injection System

A Myth scene typically blends CPU-driven analytic lights with lights generated entirely on the GPU:

```rust
// 1. Standard CPU-driven light
scene.add_light(Light::new_point(Vec3::new(1.0, 0.8, 0.6), 100.0, 0.0));
```

For extreme effects (e.g. a swarm of point lights from a GPU particle explosion), you can inject a GPU local-light track directly at the low level through the Render Graph's `FrameComposer`:

```rust
// 2. Inject GPU-driven particle lights
composer.inject_gpu_local_lights(move |ctx| {
    Some(ctx.graph.add_pass("GpuSwarmLights", |builder| {
        // Create and bind the GPU-generated light buffer
        // ...
    }))
});
```

The engine handles routing automatically, intelligently feeding pure-CPU lights, pure-GPU lights, or a merged track into the downstream Clustered Lighting computation, achieving seamless zero-overhead fallback and integration.

## Next Steps

- Add sky and shadows to your scene → [Procedural Sky & Atmosphere](/advanced/procedural-sky)
- Inject custom GPU passes → [Custom Shaders & Post FX](/advanced/custom-shader)
