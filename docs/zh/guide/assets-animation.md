# 资产、glTF 与动画

Myth 内置了完整的 **glTF 2.0** 加载能力与一套**异步资产系统**。本页聚焦于实际工作流：如何加载模型、播放骨骼/变形动画，以及如何正确处理异步时序。

## 1. 加载 glTF / GLB 模型

glTF 资源通过 `engine.assets.load_gltf()` 加载。它会解析节点树、网格、PBR 材质、蒙皮骨骼、动画轨道与 Morph Target，并返回一个 **Prefab 句柄**：

```rust
// 返回 PrefabHandle —— 后台异步加载
let model_prefab = engine.assets.load_gltf("assets/Michelle.glb");
```

::: tip Prefab vs 实例
`load_gltf` 返回的是一个**预制件 (Prefab)**，相当于一个可复用的模板。你可以用同一个 Prefab 在场景中实例化任意多个副本，它们共享底层几何与材质资源。
:::

## 2. 实例化与时序检查

由于加载是**完全异步**的，调用 `load_gltf` 后 Prefab 不会立刻就绪。正确的做法是在 `update` 循环中轮询，待资源加载完成后再 `instantiate`：

```rust
fn update(&mut self, engine: &mut Engine, _window: &dyn Window, _frame: &FrameState) {
    let assets = engine.assets.clone();
    let Some(scene) = engine.scene_manager.active_scene_mut() else { return };

    if !self.model_loaded {
        if let Some(prefab) = assets.prefabs.get(self.model_prefab) {
            // 资源就绪，实例化进场景
            let root = scene.instantiate(prefab.as_ref());
            self.model_loaded = true;
            // …在此处启动动画（见下文）
        } else if let Some(err) = assets.prefabs.get_error(self.model_prefab) {
            eprintln!("模型加载失败: {err}");
            self.model_loaded = true;
        }
    }
}
```

::: warning ⚠️ 警惕异步时序陷阱
在程序启动的前几帧，模型可能尚未完成 GPU 上传。应注意就绪检查保护，否则会出现闪烁、阴影撕裂等隐蔽 Bug。详见 [异步资源与加载管线](/zh/architecture/asset-pipeline)。
:::

## 3. 播放动画

实例化后，引擎会为含动画的节点自动安装一个 **Animation Mixer（动画混合器）**。通过 `scene.animation_mixers` 获取并控制播放：

```rust
let root = scene.instantiate(prefab.as_ref());

if let Some(mixer) = scene.animation_mixers.get_mut(root) {
    // 列出模型内的所有动画片段
    for name in mixer.list_animations() {
        println!(" - {name}");
    }

    // 按名称播放
    mixer.play("SambaDance");
}
```

混合器支持骨骼蒙皮（Skinning）与 Morph Target（变形目标）动画。引擎会在每帧自动推进时间轴并更新骨骼矩阵 / 形变权重，无需手动驱动。

## 4. 纹理与环境贴图

除了模型，常见的资产加载方式还包括：

```rust
// 普通 2D 纹理（指定色彩空间，是否生成 mipmap）
let albedo = engine.assets.load_texture(
    "assets/uv_grid.png",
    ColorSpace::Srgb,
    true,
);

// HDR 环境贴图，用于 IBL 基于图像的光照
let env = engine.assets.load_texture(
    "assets/studio.hdr.jpg",
    ColorSpace::Srgb,
    false,
);
scene.environment.set_env_map(Some(env));
```

设置环境贴图后，引擎会自动完成预过滤（PMREM）并将其作为漫反射辐照度与镜面反射的光照来源，配合 PBR 材质即可得到真实的基于图像的光照效果。

## 5. 程序化纹理

对于原型设计，引擎内置了便捷的程序化纹理生成器，无需任何外部资源：

```rust
// 棋盘格纹理：尺寸 512，格子大小 64
let checker = engine.assets.checkerboard(512, 64);
```

## 下一步

- 想理解异步管线的底层机制？ → [异步资源与加载管线](/zh/architecture/asset-pipeline)
- 想自定义材质外观？ → [PBR 物理材质](/zh/advanced/pbr-materials)
