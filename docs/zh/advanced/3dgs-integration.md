# 3D Gaussian Splatting (3DGS) 融合渲染

**Myth 引擎极具前瞻性地将 3D Gaussian Splatting 作为一等公民集成到了渲染管线中。** 这不仅仅是一个独立的查看器，而是真正实现了神经辐射场与传统多边形 PBR 管线的**深度混合渲染**。

## 1. 开启 3DGS 特性

由于 3DGS 需要高度定制的 GPU 基数排序 (Radix Sort) 和间接绘制指令，该功能默认隐藏在 Feature Flag 之后以保持引擎极致的轻量化。

在 `Cargo.toml` 中启用相关特性：

```toml
myth = { git = "...", branch = "main", features = ["3dgs", "gaussian-npz", "gaussian-spz"] }

```

## 2. 基础加载与渲染

你可以像加载普通网格体一样轻松加载点云数据，并将其放置在场景图 (Scene Graph) 中：

```rust
// 1. 加载压缩的 NPZ 或 SPZ 格式高斯点云
let cloud = engine.assets.load_gaussian_npz("assets/3dgs/point_cloud.npz".into());
let spz_cloud = engine.assets.load_gaussian_spz("assets/3dgs/scene.spz".into());

// 2. 注入场景节点
let node = scene.add_gaussian_cloud("gaussian_cloud", cloud);

// 3. 自由进行空间变换
scene.node(&node).set_rotation_euler(
    std::f32::consts::FRAC_PI_2, 
    0.0, 
    0.0
);

```

::: warning 必须使用高保真管线
3DGS 强依赖于高精度的深度缓冲与后处理合成逻辑，因此在初始化 `App` 时，必须显式指定使用 `RenderPath::HighFidelity` 渲染路径。
:::

## 3. 物理正确的混合管线 (Hybrid Pipeline)

Myth 对 3DGS 的集成解决了目前业界的几个痛点：

1. **深度测试 (Depth Testing)：** 高斯点云在进行光栅化时，会读取 PBR 不透明几何体（Opaque Pass）生成的深度缓冲，从而实现传统 3D 模型与高斯溅射场景的完美空间遮挡。
2. **色彩空间合成：** 引擎在执行投影与协方差计算时，严格校准了颜色转换逻辑。确保点云的输出结果能以正确的线性颜色空间与 PBR 场景融合，最终统一交由 Bloom 和 Tone Mapping 节点处理，杜绝了突兀的色差问题。

## 下一步

- 理解 3DGS 在帧中的插入位置 → [渲染路径与帧合成](/zh/architecture/rendering-pipeline)
- 在线体验 3DGS Demo → [Gallery](https://panxinmiao.github.io/myth)
