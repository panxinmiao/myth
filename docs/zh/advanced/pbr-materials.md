# PBR 物理材质

Myth 内置了一套稳健的**基于物理的渲染 (PBR)** 材质管线。`PhysicalMaterial` 不仅支持标准的金属-粗糙度工作流，还实现了 glTF 扩展中的多项高级表面特性：清漆 (Clearcoat)、彩虹色 (Iridescence)、透射 (Transmission)、织物光泽 (Sheen) 与各向异性 (Anisotropy)。

## 1. 基础工作流

`PhysicalMaterial` 采用链式 (builder) API。从基础色出发，逐步叠加物理属性：

```rust
use myth::prelude::*;

let material = engine.assets.materials.add(
    PhysicalMaterial::new(Vec4::new(0.8, 0.1, 0.1, 1.0)) // 基础色 (RGBA)
        .with_metallic(1.0)      // 金属度 [0, 1]
        .with_roughness(0.25)    // 粗糙度 [0, 1]
);
```

::: tip 金属度 / 粗糙度的直觉
- **金属度 (Metallic)：** 0 表示电介质（塑料、木头、皮肤），1 表示纯金属。
- **粗糙度 (Roughness)：** 0 表示镜面般光滑，1 表示完全漫反射。
:::

## 2. 自发光与纹理贴图

```rust
let material = PhysicalMaterial::new(Vec4::new(0.1, 0.1, 0.1, 1.0))
    .with_roughness(0.18)
    // 自发光：颜色 + 强度，会参与 Bloom
    .with_emissive(Vec3::new(0.25, 0.86, 1.0), 2.2)
    // 反照率贴图
    .with_map(albedo_tex);
```

自发光材质配合 [Bloom](/zh/advanced/post-processing) 可以得到霓虹灯、能量核心等发光效果。

## 3. 高级表面特性

下面这些特性对应 glTF 2.0 的官方扩展，可与基础属性自由组合：

### 清漆 Clearcoat

在基础材质之上叠加一层透明涂层，模拟车漆、上光木材等：

```rust
PhysicalMaterial::new(base_color)
    .with_roughness(0.22)
    .with_clearcoat(1.0, 0.04); // (清漆强度, 清漆粗糙度)
```

### 织物光泽 Sheen

模拟天鹅绒、丝绸等织物边缘的柔和高光：

```rust
PhysicalMaterial::new(base_color)
    .with_roughness(0.92)
    .with_sheen(Vec3::new(0.85, 0.42, 0.96), 0.38); // (光泽颜色, 光泽粗糙度)
```

### 彩虹色 Iridescence

模拟肥皂泡、甲虫外壳、油膜等薄膜干涉色彩：

```rust
PhysicalMaterial::new(base_color)
    .with_roughness(0.08)
    .with_iridescence(1.0, 1.3, 120.0, 900.0);
    //               (强度, 薄膜IOR, 最小厚度nm, 最大厚度nm)
```

### 各向异性 Anisotropy

模拟拉丝金属、毛发等方向性高光：

```rust
PhysicalMaterial::new(base_color)
    .with_roughness(0.18)
    .with_anisotropy(0.95, std::f32::consts::PI * 0.35); // (强度, 旋转角)
```

### 透射与色散 Transmission

实现玻璃、宝石、液体等折射透明材质：

```rust
PhysicalMaterial::new(base_color)
    .with_ior(1.52)          // 折射率
    .with_roughness(0.03)
    .with_transmission(1.0, 0.55, 4.0, Vec3::new(0.96, 0.98, 1.0));
    //                (透射强度, 厚度, 色散, 吸收/衰减色)
```

::: warning 透射依赖高保真管线
透射材质需要读取场景颜色的折射背景缓冲，依赖 `HighFidelity` 渲染路径与透射拷贝/Mipmap 阶段。请参考 [渲染路径与帧合成](/zh/architecture/rendering-pipeline)。
:::

## 4. 其他内置材质

除了 `PhysicalMaterial`，引擎还提供更轻量的材质类型：

| 材质 | 用途 |
| :--- | :--- |
| `PhysicalMaterial` | 完整 PBR，生产级渲染首选 |
| `PhongMaterial` | 经典 Phong 着色，开销更低，适合原型 / 低端设备 |
| `UnlitMaterial` | 无光照，直接输出颜色 / 贴图，适合 UI、调试、天空盒等 |

```rust
// 不受光照影响，直接显示纹理
let mat = UnlitMaterial::new(Vec4::ONE).with_map(tex_handle);
```

## 下一步

- 配合环境光照看效果 → [程序化天空与大气](/zh/advanced/procedural-sky)
- 想编写完全自定义的材质？ → [自定义 Shader 与后处理](/zh/advanced/custom-shader)
- 理解底层强类型材质设计 → [高性能材质系统](/zh/architecture/material-system)
