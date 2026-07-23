# Custom Shaders & Post FX

Myth provides an extremely powerful extension mechanism for advanced users. Thanks to the SSA-based Render Graph, all custom logic can be embedded into the engine's main loop with zero side effects.

## 1. Custom Mesh Materials (`#[myth_material]`)

To write a custom mesh material that stays within the standard geometry pipeline, the high-level `#[myth_material]` API is the first choice. Via a macro it automatically generates an aligned uniform struct, version tracking, and WGSL bindings.

Below is a minimal Hologram material. We inject the logic body via `shader_src` (Body mode), and the engine automatically fills in the lighting structs and vertex preamble for you:

```rust
use myth::prelude::*;
use myth::resources::myth_material;

const HOLO_SHADER: &str = r#"
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = u_model.world_matrix * vec4<f32>(in.position.xyz, 1.0);
    out.position = u_render_state.view_projection * world_pos;
    out.world_position = world_pos.xyz / world_pos.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let pulse = 0.5 + 0.5 * sin(u_render_state.time * u_material.speed);
    let color = u_material.base_color.rgb * (0.4 + pulse * 0.6);
    return pack_fragment_output(vec4<f32>(color, u_material.opacity));
}
"#;

#[myth_material(shader = "examples/holo", shader_src = HOLO_SHADER)]
pub struct HoloMaterial {
    #[uniform(default = "Vec4::new(0.1, 0.8, 1.2, 1.0)")]
    pub base_color: Vec4,
    #[uniform(default = "1.0")]
    pub opacity: f32,
    #[uniform(default = "2.0")]
    pub speed: f32,
}
```

::: tip Macro-Generated Dynamic Defines
If you add a `#[texture]` field to the struct, the macro automatically generates the corresponding binding declaration in WGSL, the `u_material.xxx_transform`, and precompiler defines like `HAS_MAP`.
:::

## 2. The Template Pass System

For work that "doesn't naturally belong to an object's material" (full-screen post-processing, screen-space effects, or GPU data generation), use the **Template Pass** system.

The system splits into two stages for maximum decoupling and reuse:

**Stage 1: Builder stage (declare the static layout)**

```rust
let post_pass = RenderPassBuilder::fullscreen("Custom Post FX")
    .inline_shader_template(SHADER_NAME, SHADER_WGSL)
    .bind_texture_2d(0, 0, wgpu::ShaderStages::FRAGMENT, true) // declare binding slot
    .color_target(...)
    .build(&mut engine.renderer);
```

**Stage 2: Graph stage (inject dynamic graph resources)**
Combined with the `FrameComposer` hook system (e.g. `HookStage::BeforePostProcess`), insert it into the Render Graph each frame and bind the concrete transient graph texture:

```rust
let node = post_pass.build_node(
    builder,
    "Custom Post FX Pass",
    out,
    RenderTargetOps::DontCare,
    Some("CustomPostFX BindGroup"),
    |bindings| {
        bindings.bind_texture(0, 0, scene_color); // bind the scene texture actually produced this frame
    },
);
```

This design lets the compiler "see through" your custom post-processing node and intelligently reuse physical memory across upstream and downstream pipelines.

## Next Steps

- Understand the frame composition order for hooks → [Render Paths & Frame Composer](/architecture/rendering-pipeline)
- Understand the underlying resource scheduling → [Render Graph](/architecture/render-graph)
