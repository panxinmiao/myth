# 自定义 Shader 与后处理

Myth 提供了一套极为强大的面向高级用户的自定义扩展机制。得益于基于 SSA 的 Render Graph，所有自定义逻辑都可以零副作用地嵌入引擎的主循环。

## 1. 自定义网格材质 (`#[myth_material]`)

想要编写一个保留在标准几何管线内的自定义网格材质，首选高层 API `#[myth_material]`。它通过宏自动生成对齐的 Uniform 结构、版本追踪以及 WGSL 绑定。

以下是一个极简的全息 (Hologram) 材质实现，我们使用 `shader_src` 注入逻辑主体（Body 模式），引擎会自动为你补全光照结构体和顶点前导块：

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

::: tip 宏生成的动态 Define
如果你在结构体中添加了 `#[texture]` 字段，宏会自动在 WGSL 中生成对应的绑定声明、`u_material.xxx_transform` 以及 `HAS_MAP` 等预编译宏。
:::

## 2. 模板通道 (Template Pass) 系统

对于那些“不自然属于物体材质”的工作（如全屏后处理、屏幕空间特效或 GPU 数据生成），请使用 **Template Pass** 系统。

该系统分为两个阶段以实现最大化的解耦与复用：

**阶段 1：Builder 阶段 (声明静态布局)**

```rust
let post_pass = RenderPassBuilder::fullscreen("Custom Post FX")
    .inline_shader_template(SHADER_NAME, SHADER_WGSL)
    .bind_texture_2d(0, 0, wgpu::ShaderStages::FRAGMENT, true) // 声明绑定槽
    .color_target(...)
    .build(&mut engine.renderer);

```

**阶段 2：Graph 阶段 (注入动态图资源)**
结合 `FrameComposer` 的钩子系统（如 `HookStage::BeforePostProcess`），在每帧将其插入 Render Graph，并绑定具体的瞬态图纹理：

```rust
let node = post_pass.build_node(
    builder,
    "Custom Post FX Pass",
    out,
    RenderTargetOps::DontCare,
    Some("CustomPostFX BindGroup"),
    |bindings| {
        bindings.bind_texture(0, 0, scene_color); // 绑定当帧实际产生的场景纹理
    },
);

```

这种设计让编译器能够“看透”你的自定义后处理节点，并智能地在上下游管线之间复用物理内存。

## 下一步

- 理解钩子插入的帧合成顺序 → [渲染路径与帧合成](/zh/architecture/rendering-pipeline)
- 理解底层资源调度 → [Render Graph 渲染图](/zh/architecture/render-graph)