# 后处理与屏幕空间特效

Myth 的 `HighFidelity` 渲染路径内置了一整套现代后处理与屏幕空间特效。所有特效都作为 Render Graph 的节点参与编译，**未启用的特效会被自动剔除，不产生任何开销**。

大部分后处理特效都挂在 `scene` 上，可在初始化或运行时随时配置。

::: warning 前提
本页所有特效均依赖 `RenderPath::HighFidelity`。请确保在创建 `App` 时已选择高保真渲染路径，详见 [渲染路径与帧合成](/zh/architecture/rendering-pipeline)。
:::

## 1. Bloom（泛光）

Bloom 提取画面中的高亮区域并向四周扩散，营造发光感。配合自发光 PBR 材质效果最佳：

```rust
scene.bloom.set_enabled(true);
scene.bloom.set_strength(0.04);        // 强度
scene.bloom.set_radius(0.005);         // 扩散半径
scene.bloom.set_karis_average(true);   // Karis 平均，抑制萤火虫噪点
```

Bloom 采用经典的多级降采样 / 升采样金字塔结构（见 [Render Graph](/zh/architecture/render-graph) 中的拓扑图），在质量与性能之间取得平衡。

## 2. SSAO（屏幕空间环境光遮蔽）

SSAO 为几何缝隙、接触区域增加柔和的暗部，显著提升画面的体积感与真实感：

```rust
scene.ssao.enabled = true;
```

SSAO 复用 Pre Pass 产生的深度与法线缓冲，并在使用前进行模糊处理以消除噪点。

## 3. SSR（屏幕空间反射）

SSR 在屏幕空间内追踪光线，为湿滑地面、抛光金属等表面提供实时反射：

```rust
scene.ssr.set_enabled(true);
scene.ssr.set_quality(SsrQuality::Ultra); // Low / Medium / High / Ultra
scene.ssr.set_thickness(0.01);            // 厚度阈值，影响相交判定
```

::: tip 质量与性能
SSR 提供多档质量预设。`Ultra` 追求最高视觉品质，`Low` 则在移动端等受限平台上保持流畅。可在运行时动态切换。
:::

## 4. SSGI（屏幕空间全局光照）

SSGI 在屏幕空间内近似计算间接光照（光线弹射），让色彩在物体间相互渗透 (color bleeding)，大幅增强真实感：

```rust
scene.ssgi.set_enabled(true);
scene.ssgi.set_quality(SsgiQuality::Ultra);
```

## 5. SSSS（屏幕空间次表面散射）

SSSS 模拟光线在皮肤、蜡、大理石等半透明材质内部的散射，对人物皮肤渲染尤为关键。它依赖 Pre Pass 的法线与 Feature ID，在不透明着色后进行水平 / 垂直两遍模糊。

```rust
scene.ssss.set_enabled(true);
```

## 6. 抗锯齿：TAA / FXAA / MSAA

引擎提供多种抗锯齿方案：

| 方案 | 特点 |
| :--- | :--- |
| **TAA**（时域抗锯齿） | 利用历史帧 + 速度缓冲，质量最高，同时抑制高频闪烁；配合 CAS 锐化恢复清晰度 |
| **FXAA**（快速近似） | 单帧后处理，开销极低，作为最终边缘平滑 |
| **MSAA**（多重采样） | 硬件级几何边缘抗锯齿 |

TAA 是高保真路径的推荐方案，它在解析阶段读写历史颜色 / 深度缓冲，并通过速度缓冲做重投影。

## 7. HDR、色调映射与色彩分级

整条管线运行在 **HDR 线性色彩空间**中。在最终输出前，Tone Mapping 阶段将高动态范围压缩到显示器可呈现的范围，并应用色彩分级。这保证了 Bloom、自发光、透射等高动态范围效果能够物理正确地融合。

## 性能哲学

得益于 SSA Render Graph，这些特效**按需付费**：

- 关闭某个特效 → 编译器自动剔除其节点及唯一为其服务的前置节点（死节点剔除）。
- 多个特效共享的中间纹理 → 编译器自动进行内存别名复用，零显存浪费。

因此你可以放心地按场景需求自由开关特效，无需担心残留开销。详见 [Render Graph 渲染图](/zh/architecture/render-graph)。

## 下一步

- 想插入完全自定义的后处理？ → [自定义 Shader 与后处理](/zh/advanced/custom-shader)
- 配置发光材质 → [PBR 物理材质](/zh/advanced/pbr-materials)
