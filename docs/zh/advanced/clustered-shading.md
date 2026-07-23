# GPU-Driven 与聚类光照

在复杂的 3D 场景中，随着动态光源数量的增加，传统的前向渲染（Forward Rendering）会导致性能急剧下降，而延迟渲染（Deferred Rendering）又在处理透明物体和多重采样抗锯齿（MSAA）时面临诸多限制。

Myth 引擎原生采用 **聚类前向光照 (Clustered Forward Lighting)** 架构，结合强大的 GPU-Driven 管线，轻松应对海量动态光源。

## 1. 原理与优势

聚类光照将相机的视锥体在 3D 空间中划分为多个小视锥（即 Cluster 聚类）。
在渲染每一帧之前，引擎会调度 Compute Shader 执行精准的**灯光剔除 (Light Culling)**，将影响每个 Cluster 的光源索引记录在 GPU 的显存中。

- **极致性能：** 在最终的着色阶段 (Fragment Shader)，每个像素只需遍历其所属 Cluster 内部的光源，从而将光照计算的复杂度从 $O(M \times N)$ 骤降至 $O(M \times K)$（$K$ 远小于 $N$）。
- **无瑕疵着色：** 引擎底层采用了极其严谨的包围盒与视锥体相交测试逻辑。这种精确的数学计算不仅提高了剔除效率，更从根本上消除了聚类分块时容易出现的“块状阴影 (Blocky Shadows)”或光照割裂等视觉瑕疵。

## 2. 混合光源注入体系

Myth 场景通常将 CPU 驱动的分析光源与完全由 GPU 生成的光源混合使用：

```rust
// 1. 常规的 CPU 驱动光源
scene.add_light(Light::new_point(Vec3::new(1.0, 0.8, 0.6), 100.0, 0.0));

```

对于极端的特效需求（例如 GPU 粒子爆炸产生的大量点光源），你可以通过 Render Graph 的 `FrameComposer` 直接在底层注入 GPU 局部光源轨道：

```rust
// 2. 注入 GPU 驱动的粒子光源
composer.inject_gpu_local_lights(move |ctx| {
    Some(ctx.graph.add_pass("GpuSwarmLights", |builder| {
        // 创建并绑定 GPU 生成的光源 Buffer
        // ...
    }))
});

```

引擎内部会自动处理路由逻辑，智能地将纯 CPU 光源、纯 GPU 光源或两者的合并轨道送入后续的 Clustered Lighting 计算中，实现零开销的无缝回退与整合。

## 下一步

- 为场景添加天空与阴影 → [程序化天空与大气](/zh/advanced/procedural-sky)
- 注入自定义 GPU Pass → [自定义 Shader 与后处理](/zh/advanced/custom-shader)