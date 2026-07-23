# 场景与节点系统

## 1. 心智模型

理解 Myth 的核心在于掌握它的四个协作层级。它们自上而下构成了整个引擎的运行流：

1. **App**: 负责窗口创建、上下文管理和事件循环处理。
2. **Engine**: 引擎核心载体，持有 Renderer（渲染器）、Scene Manager（场景管理器）、Asset Server（资产服务器）以及 Input State（输入状态）。
3. **Scene**: 持有 Nodes（节点）、Cameras（相机）、Lights（光源）、环境配置及动画混合器。
4. **Render Graph**: 位于最底层，每一帧都会根据当前活跃 Scene 的拓扑结构，动态组装并编译当前的渲染管线。

## 2. 节点与层级构建

### 快捷生成助手 (Spawn Helpers)

在进行原型设计或编写示例时，使用 `SceneExt` 提供的 helper 方法是最简洁的构建途径。它们会自动在资源管理器中注册几何体与材质：

```rust
let cube = scene.spawn_box(1.0, 1.0, 1.0, material, &engine.assets);
let sphere = scene.spawn_sphere(0.5, material, &engine.assets);
let ground = scene.spawn_plane(20.0, 20.0, material, &engine.assets);

```

### 空间变换与父子层级

Myth 采用实体句柄（Handle）的方式管理场景图。你可以轻松地绑定父子关系，并使用流畅的接口操作空间变换：

```rust
let parent = scene.spawn_box(1.0, 1.0, 1.0, material, &engine.assets);
let child = scene.spawn_sphere(0.35, material, &engine.assets);

// 建立父子关系
scene.attach(child, parent);

// 链式调用设置 Transform
scene.node(&parent).set_position(0.0, 1.0, 0.0);
scene.node(&child).set_position(1.0, 0.0, 0.0);

```

::: info 获取节点的两种方式

* 使用 `scene.node(&handle)`：提供链式的快捷 API（如 `set_position`、`look_at`），适合快速初始化。
* 使用 `scene.get_node_mut(handle)`：返回节点的可变引用，适合在 `on_update` 循环中进行复杂的逻辑计算和底层组件访问。
:::

## 3. 摄像机与环境控制

### 摄像机 (Camera)

渲染器会跳过没有激活摄像机的场景。典型的设置如下：

```rust
let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
scene.node(&camera).set_position(0.0, 2.0, 5.0).look_at(Vec3::ZERO);

// 务必将其设置为活跃相机
scene.active_camera = Some(camera);

```

### 光源 (Lights)

Myth 场景通常将一个或多个分析光源（Analytic Lights）与基于图像的光照（IBL）混合使用。

方向光（Directional Light）始终走全局光照路径；而点光源（Point）和聚光灯（Spot）将被 RDG 托管，并送入高性能的**聚类光照 (Clustered Lighting)** 轨道：

```rust
// 全局方向光，投射主要阴影
scene.add_light(Light::new_directional(Vec3::ONE, 5.0));

// 局部点光源，参与聚类光照剔除
scene.add_light(Light::new_point(Vec3::new(1.0, 0.8, 0.6), 100.0, 0.0));

```

## 4. 手动资源生命周期

如果你需要对资产的复用进行极其严格的控制，可以绕过 Spawn Helpers，直接与引擎的资源池进行交互：

```rust
// 1. 手动添加几何体与材质资源
let geo = engine.assets.geometries.add(Geometry::new_box(1.0, 1.0, 1.0));
let mat = engine.assets.materials.add(PhysicalMaterial::default());

// 2. 利用资源句柄实例化 Mesh 节点
let node = scene.add_mesh(Mesh::new(geo, mat));

```