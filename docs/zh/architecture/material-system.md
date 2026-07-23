# 材质系统

Myth 的材质系统围绕一个目标设计：**让自定义材质既易于编写，又不在渲染热路径（Hot Path）上引入额外开销。** 本章介绍当前的材质架构、`#[myth_material]` 宏的能力，以及如何实现你自己的材质。

## 1. 架构概览

引擎中的每一种材质都是一个**强类型、内存紧凑的 Rust 结构体**。材质属性按 `std140` 布局排列，可以直接作为 Uniform 上传到 GPU；纹理则通过独立的绑定槽（Texture Slot）声明。

这带来三个直接的好处：

- **缓存友好**：每帧为成千上万个材质实例构建 Bind Group 时，访问的是连续的紧凑内存，没有字符串哈希或散列查找。
- **编译期校验**：字段类型、对齐和 Shader 契约在编译期确定，错误在构建时即被捕获。
- **自动同步**：材质属性或贴图发生变化时，引擎会自动让对应的管线缓存失效并重建，无需手动管理。

```rust
// 一个最小的自定义材质定义
#[myth_material(shader = "examples/holo", shader_src = HOLO_SHADER)]
pub struct HoloMaterial {
    #[uniform(default = "Vec4::new(0.1, 0.8, 1.2, 1.0)")]
    pub base_color: Vec4,

    #[uniform(default = "1.0")]
    pub opacity: f32,

    #[texture]
    pub normal_map: TextureSlot,
}
```

## 2. `#[myth_material]` 宏

`#[myth_material]` 是连接 CPU 端数据与 GPU 端 Shader 契约的核心。它在编译期为每一个材质生成所需的全部样板代码：

| 生成内容 | 说明 |
| --- | --- |
| **GPU Uniform 结构** | 按 `std140` 自动处理字段对齐与填充，可直接上传，零运行时开销。 |
| **WGSL 映射** | `#[uniform]` 字段自动映射为 Shader 中 `u_material` 的成员；`#[texture]` 字段自动生成纹理 / 采样器绑定，并定义 `HAS_NORMAL_MAP` 等条件编译宏。 |
| **版本追踪** | 属性或贴图变更时自动标记脏状态，触发对应管线缓存的失效与重建。 |
| **默认值** | `default = "..."` 表达式在材质构造时求值，省去手写 `Default` 实现。 |

### 字段属性

- `#[uniform]`：标记一个会被打包进 Uniform Buffer 的数值字段（`f32`、`Vec3`、`Vec4`、`Mat4` 等）。可选 `default` 表达式提供初始值。
- `#[texture]`：标记一个 `TextureSlot` 字段。绑定存在时，引擎会在 Shader 中自动定义对应的 `HAS_*_MAP` 宏，便于在 WGSL 里做分支。

宏头部的 `shader` 用于标识材质类别（影响管线缓存键），`shader_src` 指向材质的 WGSL 源（可内联常量或外部文件）。

## 3. 着色器模板系统

为避免开发者编写大量与光照、几何相关的样板 WGSL，材质着色器支持两种模式：

- **`MaterialBody` 模式（默认）**：你只需编写核心着色逻辑（`vs_main` / `fs_main` 的内部计算）。编译器会自动为你注入：
  - 场景光照结构体（`scene_lighting_structs`）
  - 集群光照定义（`clustered_lighting_structs`）
  - 依据几何布局自动生成的顶点输入结构（`VertexInput`）
- **`Template` 模式**：当你需要完全掌控整个 Shader（包括入口函数与绑定布局）时使用，引擎仅做最少的必要注入。

由于注入是按管线进行的，**同一份材质代码可以无缝复用于前向渲染、深度预处理（Depth Prepass）与阴影投射（Shadow Pass）等多套管线**，无需为每条管线重复编写。

## 4. 实现一个自定义材质

完整流程分三步：定义结构体、编写 WGSL、在场景中使用。

```rust
// 1. 定义材质（CPU 侧）
#[myth_material(shader = "examples/holo", shader_src = HOLO_SHADER)]
pub struct HoloMaterial {
    #[uniform(default = "Vec4::new(0.1, 0.8, 1.2, 1.0)")]
    pub base_color: Vec4,
    #[uniform(default = "1.0")]
    pub opacity: f32,
}

// 2. 编写着色逻辑（MaterialBody 模式，仅核心部分）
const HOLO_SHADER: &str = r#"
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // u_material 由宏根据字段自动生成
    let glow = u_material.base_color.rgb * (0.6 + 0.4 * sin(globals.time * 3.0));
    return vec4<f32>(glow, u_material.opacity);
}
"#;

// 3. 在场景中使用
let mat = HoloMaterial::default();
let mesh = scene.spawn_box(1.0, 1.0, 1.0, mat, &engine.assets);
```

引擎会自动为该材质创建并缓存管线，处理 Uniform 上传、绑定与版本追踪。运行时修改 `mat.opacity` 等字段，引擎会在下一帧自动同步到 GPU。

## 下一步

- 使用内置 PBR 材质 → [PBR 物理材质](/zh/advanced/pbr-materials)
- 编写更复杂的自定义材质与后处理 → [自定义 Shader 与后处理](/zh/advanced/custom-shader)