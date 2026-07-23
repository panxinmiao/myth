# 异步资源与加载管线

对于一个现代化的 3D 引擎而言，阻塞主线程来加载百兆级别的模型资源是不可接受的。Myth 引擎构建了一套稳健的**异步资产加载服务器 (Asset Server)**，确保了渲染循环的绝对流畅。

## 1. 资产的获取与管理

引擎的资源统一由 `engine.assets` 进行管理，它接管了纹理、模型、材质和网格体的生命周期。

### 基础资源的加载
大部分本地或网络资源的加载都非常直观：

```rust
// 异步加载纹理
let albedo = engine.assets.load_texture("assets/uv_grid.png", ColorSpace::Srgb, true);

// 异步加载 HDR 环境贴图
let env = engine.assets.load_hdr_texture("assets/studio.hdr");
scene.environment.set_env_map(Some(env));

```

### glTF 场景预制件

对于复杂的 glTF 或 GLB 文件，引擎会解析其节点树、绑定蒙皮，并自动安装动画混合器：

```rust
let prefab = GltfLoader::load("assets/model.glb", engine.assets.clone())?;
let root_node = scene.instantiate(&prefab);

```

## 2. 异步时序处理的关键准则

Myth 的加载管线是**完全异步**的。这意味着当你调用 `scene.instantiate()` 时，场景节点树虽然被立即创建了，但底层的 GPU 资源（如顶点 Buffer、贴图数据）可能仍在后台排队上传。

::: warning ⚠️ 重要陷阱：警惕“前几帧”的时序逻辑
由于模型加载使用异步方式，**在程序启动的前几帧中，模型尚未被完全加载到场景是常态。**

如果你的底层逻辑强依赖于实体立即存在的数据，必须显式地处理这种异步时序。

这种时序错位往往不会直接抛出 Panic，而是会表现出极其隐蔽的渲染 Bug（如突然的画面闪烁、阴影撕裂等）。最佳实践是：在 `on_update` 循环中，始终通过 `Option` 或资产状态检查来确认资源是否已就绪，再执行相应逻辑。
:::
## 3. 实践指引

关于加载 glTF、实例化、播放动画以及处理就绪检查的完整工作流，请参考指南章节 [资产、glTF 与动画](/zh/guide/assets-animation)。