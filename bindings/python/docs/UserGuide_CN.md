# Myth Engine Python — 用户指南

> **Myth Engine** 是基于 wgpu 的高性能 3D 渲染引擎，提供 Three.js 风格的面向对象 API。  
> 本指南将带你从零开始，逐步掌握使用 Python 构建 3D 场景的全过程。

---

## 目录

- [安装](#安装)
  - [前置条件](#前置条件)
  - [一键安装](#一键安装)
  - [手动安装](#手动安装)
- [快速入门](#快速入门)
  - [你的第一个场景](#你的第一个场景)
  - [核心概念](#核心概念)
- [应用架构](#应用架构)
  - [App 模式 — 内置窗口](#app-模式--内置窗口)
  - [Renderer 模式 — 外部窗口](#renderer-模式--外部窗口)
- [场景构建](#场景构建)
  - [几何体](#几何体)
  - [材质系统](#材质系统)
  - [创建网格](#创建网格)
  - [场景层级](#场景层级)
- [相机](#相机)
  - [透视相机](#透视相机)
  - [正交相机](#正交相机)
  - [轨道控制器](#轨道控制器)
- [灯光与阴影](#灯光与阴影)
  - [灯光类型](#灯光类型)
  - [阴影](#阴影)
- [组件代理](#组件代理)
  - [相机组件](#相机组件)
  - [灯光组件](#灯光组件)
  - [网格组件](#网格组件)
- [纹理](#纹理)
  - [加载纹理](#加载纹理)
  - [应用纹理](#应用纹理)
- [环境与背景](#环境与背景)
  - [背景颜色](#背景颜色)
  - [环境贴图与 IBL](#环境贴图与-ibl)
- [后处理](#后处理)
  - [色调映射](#色调映射)
  - [Bloom 泛光](#bloom-泛光)
  - [FXAA 抗锯齿](#fxaa-抗锯齿)
  - [SSAO 环境光遮蔽](#ssao-环境光遮蔽)
- [动画](#动画)
  - [播放动画](#播放动画)
  - [高级动画控制](#高级动画控制)
- [输入处理](#输入处理)
  - [键盘输入](#键盘输入)
  - [鼠标输入](#鼠标输入)
- [加载 glTF 模型](#加载-gltf-模型)
- [自定义几何体](#自定义几何体)
- [外部窗口集成](#外部窗口集成)
  - [GLFW 集成](#glfw-集成)
  - [PySide6 (Qt) 集成](#pyside6-qt-集成)
  - [RenderCanvas 集成](#rendercanvas-集成)
- [渲染路径](#渲染路径)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 安装

### 前置条件

- **Rust** 工具链 — [rustup.rs](https://rustup.rs)
- **Python 3.9+**
- **myth-engine** 源码位于 `../myth-engine`（兄弟目录）

### 一键安装

```bash
# Windows
scripts\install.bat

# Linux / macOS
./scripts/install.sh
```

该脚本会自动创建虚拟环境、安装 [maturin](https://www.maturin.rs)，并以 release 模式构建库。

### 手动安装

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate

# 2. 安装 maturin
pip install maturin

# 3. 构建并安装（release 模式）
maturin develop --release
```

### 构建选项

```bash
# Release 构建（默认，推荐）
scripts/build.bat release

# Debug 构建（编译更快，运行更慢）
scripts/build.bat debug

# 构建可分发的 wheel 包
scripts/build.bat wheel

# 清除所有构建产物
scripts/build.bat clean
```

---

## 快速入门

### 你的第一个场景

创建一个包含旋转立方体的最小 3D 场景：

```python
import myth

# 创建应用
app = myth.App(title="Hello Myth", render_path="basic")

cube = None
cam = None

@app.init
def on_init(ctx):
    global cube, cam
    scene = ctx.create_scene()

    # 添加一个立方体
    cube = scene.add_mesh(
        myth.BoxGeometry(1, 1, 1),
        myth.PhysicalMaterial(color="#ff8033", roughness=0.4),
    )
    cube.position = [0, 0.5, 0]

    # 添加相机
    cam = scene.add_camera(myth.PerspectiveCamera(fov=60))
    cam.position = [3, 3, 5]
    cam.look_at([0, 0, 0])
    scene.active_camera = cam

    # 添加灯光
    sun = scene.add_light(myth.DirectionalLight(intensity=2.0, cast_shadows=True))
    sun.position = [5, 10, 5]
    sun.look_at([0, 0, 0])

@app.update
def on_update(ctx, frame):
    cube.rotate_y(frame.dt * 0.5)

app.run()
```

运行：

```bash
python your_script.py
```

### 核心概念

Myth Engine 的 Python API 围绕以下核心概念组织：

```
App / Renderer          → 应用入口，管理窗口和渲染循环
  └── Engine            → 引擎上下文（回调中使用）
        └── Scene       → 场景，持有所有 3D 对象
              ├── Object3D (Mesh)     → 几何体 + 材质
              ├── Object3D (Camera)   → 相机
              └── Object3D (Light)    → 灯光
```

**工作流程**：
1. 创建 `App`（或 `Renderer`）
2. 在 `@app.init` 回调中构建场景：添加网格、相机、灯光
3. 在 `@app.update` 回调中执行每帧逻辑：动画、输入响应
4. 调用 `app.run()` 启动主循环

---

## 应用架构

Myth 提供两种使用模式：

### App 模式 — 内置窗口

`App` 类自动创建窗口并管理事件循环，适合独立 3D 应用：

```python
import myth

app = myth.App(
    title="我的3D应用",
    render_path=myth.RenderPath.HIGH_FIDELITY,
    vsync=True,
)

@app.init
def on_init(ctx: myth.Engine):
    scene = ctx.create_scene()
    # ... 构建场景

@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    # ... 每帧更新

app.run()  # 阻塞，直到窗口关闭
```

### Renderer 模式 — 外部窗口

`Renderer` 类是 GUI 无关的底层渲染器。当需要将 Myth 嵌入现有窗口框架（GLFW、Qt、wxPython 等）时使用：

```python
import myth

renderer = myth.Renderer(render_path=myth.RenderPath.BASIC)

# 使用外部窗口的原生句柄初始化
renderer.init_with_handle(window_handle, width, height)

scene = renderer.create_scene()
# ... 构建场景

# 手动驱动渲染循环
while running:
    process_events()             # 外部窗口的事件处理
    renderer.frame(dt)           # 更新 + 渲染一帧

renderer.dispose()               # 释放 GPU 资源
```

> `Renderer` 也支持上下文管理器：`with myth.Renderer(...) as r:`

---

## 场景构建

### 几何体

Myth 提供三种内置几何体和一种自定义几何体：

```python
# 立方体
box = myth.BoxGeometry(width=2, height=1, depth=1)

# 球体
sphere = myth.SphereGeometry(radius=0.5, width_segments=32, height_segments=16)

# 平面
plane = myth.PlaneGeometry(width=10, height=10)

# 自定义几何体
custom = myth.Geometry()
custom.set_positions([0, 0, 0, 1, 0, 0, 0, 1, 0])
custom.set_indices([0, 1, 2])
```

### 材质系统

Myth 提供三种材质，从简单到高级：

#### UnlitMaterial — 无光照

最简单的材质，不受光照影响。适合 UI 元素、调试辅助线等。

```python
mat = myth.UnlitMaterial(color="#ff0000", opacity=0.8)
```

#### PhongMaterial — Blinn-Phong 光照

经典光照模型，支持漫反射和高光。

```python
mat = myth.PhongMaterial(
    color="#ffffff",
    specular="#aaaaaa",
    shininess=64.0,
)
```

#### PhysicalMaterial — PBR 物理材质

基于物理的渲染（PBR），使用金属-粗糙度工作流。**推荐用于大多数场景。**

```python
mat = myth.PhysicalMaterial(
    color="#ff8033",
    metalness=0.8,        # 0.0 = 非金属, 1.0 = 金属
    roughness=0.2,        # 0.0 = 光滑, 1.0 = 粗糙
    emissive="#000000",
    emissive_intensity=1.0,
)
```

**高级属性**：

```python
mat.clearcoat = 1.0               # 清漆涂层
mat.clearcoat_roughness = 0.1
mat.transmission = 0.9             # 透射（玻璃效果）
mat.ior = 1.5                      # 折射率
```

#### 颜色格式

所有材质的颜色参数支持多种格式：

```python
# 十六进制字符串
mat = myth.PhysicalMaterial(color="#ff8033")

# RGB 列表 (0.0–1.0)
mat = myth.PhysicalMaterial(color=[1.0, 0.5, 0.2])

# RGB 元组
mat = myth.PhysicalMaterial(color=(1.0, 0.5, 0.2))
```

#### 面剔除

通过 `side` 参数控制面剔除：

| 值 | 说明 |
|----|------|
| `"front"` | 仅渲染正面（默认） |
| `"back"` | 仅渲染背面 |
| `"double"` | 双面渲染 |

### 创建网格

将几何体和材质组合为网格，添加到场景中：

```python
cube = scene.add_mesh(
    myth.BoxGeometry(1, 1, 1),
    myth.PhysicalMaterial(color="#ff8033", roughness=0.4, metalness=0.3),
)
cube.position = [0, 1, 0]
cube.rotation_euler = [0, 45, 0]    # 旋转 45 度（角度制）
cube.scale = [1, 1, 1]
```

### 场景层级

使用 `scene.attach()` 构建父子层级关系：

```python
parent = scene.add_mesh(box_geo, mat)
child = scene.add_mesh(sphere_geo, mat)
scene.attach(child, parent)

# 移动 parent 时 child 会跟随
parent.position = [2, 0, 0]
```

通过名称查找节点：

```python
node = scene.find_node_by_name("MyNode")
if node:
    node.visible = False
```

---

## 相机

### 透视相机

模拟人眼/真实相机的透视投影，近大远小。

```python
cam_node = scene.add_camera(myth.PerspectiveCamera(
    fov=60,       # 垂直视场角（度）
    near=0.1,     # 近裁剪面
    far=1000.0,   # 远裁剪面
    aspect=0.0,   # 0 = 自动
))
cam_node.position = [0, 5, 10]
cam_node.look_at([0, 0, 0])
scene.active_camera = cam_node
```

### 正交相机

无透视效果，适合 2D 游戏、建筑图纸、UI 等场景。

```python
cam_node = scene.add_camera(myth.OrthographicCamera(
    size=10.0,    # 视图高度
    near=0.1,
    far=1000.0,
))
cam_node.position = [0, 10, 0]
cam_node.look_at([0, 0, 0])
scene.active_camera = cam_node
```

### 轨道控制器

`OrbitControls` 提供 Three.js 风格的鼠标交互：

| 操作 | 功能 |
|------|------|
| 鼠标左键拖拽 | 旋转 |
| 鼠标右键/Shift+左键拖拽 | 平移 |
| 滚轮 | 缩放 |

```python
orbit = myth.OrbitControls(
    position=[3, 3, 5],
    target=[0, 0, 0],
)

# 可选：调节参数
orbit.rotate_speed = 1.0
orbit.zoom_speed = 1.0
orbit.enable_damping = True
orbit.damping_factor = 0.05

@app.update
def on_update(ctx, frame):
    orbit.update(cam, frame.dt)    # 每帧必须调用
```

---

## 灯光与阴影

### 灯光类型

#### 平行光（DirectionalLight）

模拟太阳光，所有光线平行。通过节点的 `position` 和 `look_at()` 控制方向。

```python
sun = scene.add_light(myth.DirectionalLight(
    color=[1.0, 0.95, 0.9],
    intensity=3.0,
    cast_shadows=True,
))
sun.position = [5, 10, 5]
sun.look_at([0, 0, 0])
```

#### 点光源（PointLight）

从一个点向所有方向发射光线。

```python
bulb = scene.add_light(myth.PointLight(
    color=[1.0, 0.8, 0.6],
    intensity=5.0,
    range=20.0,
))
bulb.position = [0, 3, 0]
```

#### 聚光灯（SpotLight）

以锥形向特定方向发射光线。

```python
spot = scene.add_light(myth.SpotLight(
    intensity=10.0,
    range=30.0,
    inner_cone=0.2,      # 内锥角（弧度）
    outer_cone=0.5,      # 外锥角（弧度）
    cast_shadows=True,
))
spot.position = [0, 5, 0]
spot.look_at([0, 0, 0])
```

### 阴影

在灯光上启用 `cast_shadows=True`，并在网格上控制阴影行为：

```python
# 灯光投射阴影
sun = scene.add_light(myth.DirectionalLight(intensity=3.0, cast_shadows=True))

# 网格投射 + 接收阴影
cube.cast_shadows = True
cube.receive_shadows = True

# 地面只接收阴影
ground.cast_shadows = False
ground.receive_shadows = True
```

---

## 组件代理

每个 `Object3D` 节点可以携带一个或多个**组件**（相机、灯光、网格）。通过对应的**组件代理**属性，可以在运行时直接访问和修改这些组件的参数：

| 属性 | 返回类型 | 可用条件 |
|:---|:---|:---|
| `node.camera` | `PerspectiveCameraComponent` / `OrthographicCameraComponent` | 节点由 `scene.add_camera()` 创建 |
| `node.light` | `DirectionalLightComponent` / `PointLightComponent` / `SpotLightComponent` | 节点由 `scene.add_light()` 创建 |
| `node.mesh` | `MeshComponent` | 节点由 `scene.add_mesh()` 创建 |

如果节点不包含对应组件，属性返回 `None`。

### 相机组件

在运行时访问和修改相机参数：

```python
cam_node = scene.add_camera(myth.PerspectiveCamera(fov=60))
cam_node.position = [0, 5, 10]
scene.active_camera = cam_node

# 运行时修改：
cam = cam_node.camera  # PerspectiveCameraComponent
if cam:
    cam.fov = 45.0                                                # 修改视场角
    cam.near = 0.01                                               # 近裁剪面
    cam.antialiasing = myth.AntiAliasing.taa(feedback_weight=0.9)  # 启用 TAA
```

正交相机：

```python
cam_node = scene.add_camera(myth.OrthographicCamera(size=10.0))
cam = cam_node.camera  # OrthographicCameraComponent
if cam:
    cam.size = 20.0   # 调整视图大小
```

### 灯光组件

无需替换灯光即可检查和修改灯光参数：

```python
sun = scene.add_light(myth.DirectionalLight(intensity=3.0, cast_shadows=True))
sun.position = [5, 10, 5]

# 运行时修改：
light = sun.light  # DirectionalLightComponent
if light:
    light.intensity = 5.0
    light.color = [1.0, 0.8, 0.6]
```

不同灯光类型暴露不同属性：

```python
# PointLight → PointLightComponent
bulb = scene.add_light(myth.PointLight(intensity=5.0, range=20.0))
bulb.light.range = 30.0

# SpotLight → SpotLightComponent
spot = scene.add_light(myth.SpotLight(inner_cone=0.2, outer_cone=0.5))
spot.light.inner_cone = 0.1
spot.light.outer_cone = 0.4
```

### 网格组件

控制网格渲染属性：

```python
cube = scene.add_mesh(
    myth.BoxGeometry(1, 1, 1),
    myth.PhysicalMaterial(color="#ff8033"),
)

mesh = cube.mesh  # MeshComponent
if mesh:
    mesh.visible = False         # 隐藏网格但保留节点
    mesh.cast_shadows = True
    mesh.receive_shadows = True
    mesh.render_order = 10       # 控制绘制顺序
```

> **提示**：组件代理是轻量级句柄，直接读写引擎 ECS 存储，无额外拷贝开销。

---

## 纹理

### 加载纹理

```python
# 标准纹理（sRGB 色彩空间，自动生成 mipmap）
diffuse_tex = ctx.load_texture("textures/diffuse.png")

# 线性色彩空间（法线贴图、粗糙度贴图等）
normal_tex = ctx.load_texture("textures/normal.png", color_space="linear")

# 不生成 mipmap
tex = ctx.load_texture("textures/ui.png", generate_mipmaps=False)

# HDR 环境纹理
hdr_tex = ctx.load_hdr_texture("textures/env.hdr")
```

### 应用纹理

将纹理应用到材质的不同通道：

```python
mat = myth.PhysicalMaterial(color="#ffffff", roughness=0.5, metalness=0.0)

mat.set_map(diffuse_tex)                       # 基础颜色贴图
mat.set_normal_map(normal_tex, scale=1.0)      # 法线贴图
mat.set_roughness_map(roughness_tex)           # 粗糙度贴图
mat.set_metalness_map(metalness_tex)           # 金属度贴图
mat.set_emissive_map(emissive_tex)             # 自发光贴图
mat.set_ao_map(ao_tex)                         # 环境光遮蔽贴图
```

---

## 环境与背景

### 背景颜色

```python
scene.set_background_color(0.05, 0.05, 0.1)  # 深蓝色背景
```

### 环境贴图与 IBL

使用 HDR 环境贴图可以同时获得天空盒和基于图像的光照（IBL）：

```python
hdr = ctx.load_hdr_texture("environment.hdr")
scene.set_environment_map(hdr)
scene.set_environment_intensity(1.5)   # 调节 IBL 强度
```

也可设置简单的环境光：

```python
scene.set_ambient_light(0.2, 0.2, 0.3)  # 浅蓝色环境光
```

---

## 后处理

> 后处理效果需使用 `RenderPath.HIGH_FIDELITY`（或字符串 `'hdr'`）渲染路径。

```python
app = myth.App(render_path=myth.RenderPath.HIGH_FIDELITY)
```

### 色调映射

色调映射将 HDR 颜色映射到显示器可显示的 LDR 范围。

```python
scene.set_tone_mapping("aces", exposure=1.2)
```

可选模式：

| 模式 | 说明 |
|------|------|
| `'linear'` | 线性映射（无色调映射） |
| `'neutral'` | Khronos PBR neutral tone mapping |
| `'reinhard'` | Reinhard 映射 |
| `'cineon'` | Cineon 电影风格 |
| `'aces'` | ACES Filmic（推荐，电影级色调） |
| `'agx'` | AgX 映射 |

### Bloom 泛光

让高亮区域产生发光效果：

```python
scene.set_bloom(True, strength=0.04, radius=0.3)

# 或分步调用
scene.set_bloom_enabled(True)
scene.set_bloom_strength(0.04)
scene.set_bloom_radius(0.3)
```

### SSAO 环境光遮蔽

模拟物体接缝处和角落的自然遮蔽效果：

```python
scene.set_ssao_enabled(True)
scene.set_ssao_radius(0.5)
scene.set_ssao_intensity(1.0)
scene.set_ssao_bias(0.025)
```

### 推荐后处理配置

```python
# 高质量照片级渲染
scene.set_tone_mapping("aces", exposure=1.0)
scene.set_bloom(True, strength=0.02)
scene.set_ssao_enabled(True)
```

---

## 动画

### 播放动画

加载 glTF 模型后，直接播放动画：

```python
model = ctx.load_gltf("character.glb")

# 播放任意可用动画（快捷方式）
scene.play_any_animation(model)

# 播放指定名称的动画
scene.play_animation(model, "Walk")

# 列出所有可用动画
anims = scene.list_animations(model)
print(f"可用动画: {anims}")
```

### 高级动画控制

使用 `AnimationMixer` 进行精细控制：

```python
mixer = scene.get_animation_mixer(model)

if mixer:
    anims = mixer.list_animations()
    mixer.play("Walk")

    # 切换动画
    mixer.stop("Walk")
    mixer.play("Run")

    # 停止所有
    mixer.stop_all()
```

---

## 输入处理

### 键盘输入

```python
@app.update
def on_update(ctx, frame):
    inp = ctx.input

    # 持续按住检测
    if inp.key("w"):
        print("W 键按住")

    # 首次按下检测（单帧触发）
    if inp.key_down("Space"):
        print("空格键按下")

    # 释放检测
    if inp.key_up("Escape"):
        print("Escape 释放")
```

### 鼠标输入

```python
@app.update
def on_update(ctx, frame):
    inp = ctx.input

    # 鼠标位置
    x, y = inp.mouse_position()

    # 鼠标移动增量
    dx, dy = inp.mouse_delta()

    # 滚轮增量
    sx, sy = inp.scroll_delta()

    # 按钮状态
    if inp.mouse_button("Left"):
        print(f"左键拖拽: dx={dx}, dy={dy}")

    if inp.mouse_button_down("Right"):
        print("右键点击")
```

---

## 加载 glTF 模型

Myth 支持加载 glTF 2.0（`.gltf` 和 `.glb`）模型：

```python
import sys
import myth

app = myth.App(title="glTF 查看器", render_path=myth.RenderPath.HIGH_FIDELITY)

orbit = myth.OrbitControls()
model = None
cam = None

@app.init
def on_init(ctx):
    global model, cam
    scene = ctx.create_scene()

    # 加载模型
    model = ctx.load_gltf("path/to/model.glb")

    # 相机
    cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.01))
    cam.position = [0, 1.5, 3]
    cam.look_at([0, 0, 0])
    scene.active_camera = cam

    # 灯光
    sun = scene.add_light(myth.DirectionalLight(intensity=3.0, cast_shadows=True))
    sun.position = [3, 5, 3]

    # 环境 & 后处理
    scene.set_background_color(0.15, 0.15, 0.2)
    scene.set_tone_mapping("aces", exposure=1.0)
    scene.set_bloom(True, strength=0.02)

    # 播放动画
    scene.play_any_animation(model)

    anims = scene.list_animations(model)
    if anims:
        print(f"动画: {anims}")

@app.update
def on_update(ctx, frame):
    orbit.update(cam, frame.dt)

app.run()
```

---

## 自定义几何体

当内置几何体不满足需求时，可以从原始顶点数据构建自定义几何体：

```python
geo = myth.Geometry()

# 三角形的三个顶点
geo.set_positions([
    -0.5, -0.5, 0.0,   # 顶点 0
     0.5, -0.5, 0.0,   # 顶点 1
     0.0,  0.5, 0.0,   # 顶点 2
])

# 法线（都朝 +Z 方向）
geo.set_normals([
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
])

# UV 坐标
geo.set_uvs([
    0.0, 0.0,
    1.0, 0.0,
    0.5, 1.0,
])

# 索引
geo.set_indices([0, 1, 2])

# 创建网格
mesh = scene.add_mesh(geo, myth.PhysicalMaterial(color="#00ff00"))
```

---

## 外部窗口集成

### GLFW 集成

使用 `Renderer` 将 Myth 嵌入 GLFW 窗口：

```python
import glfw
import myth

# GLFW 初始化
glfw.init()
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # 不需要 OpenGL
window = glfw.create_window(1280, 720, "Myth + GLFW", None, None)

# 创建渲染器
renderer = myth.Renderer(render_path=myth.RenderPath.BASIC, vsync=False)
hwnd = glfw.get_win32_window(window)  # Windows 平台
renderer.init_with_handle(hwnd, 1280, 720)

# 构建场景
scene = renderer.create_scene()
cube = scene.add_mesh(
    myth.BoxGeometry(1, 1, 1),
    myth.PhysicalMaterial(color="#ff8033"),
)
cam = scene.add_camera(myth.PerspectiveCamera(fov=60))
cam.position = [3, 3, 5]
cam.look_at([0, 0, 0])
scene.active_camera = cam

sun = scene.add_light(myth.DirectionalLight(intensity=2.0, cast_shadows=True))
sun.position = [5, 10, 5]

# 转发输入事件
def on_resize(win, w, h):
    if w > 0 and h > 0:
        renderer.resize(w, h)

def on_cursor(win, x, y):
    renderer.inject_mouse_move(x, y)

def on_mouse_btn(win, button, action, mods):
    # GLFW: 0=left, 1=right, 2=middle → Myth: 0=left, 1=middle, 2=right
    btn_map = {0: 0, 1: 2, 2: 1}
    btn = btn_map.get(button, button)
    if action == glfw.PRESS:
        renderer.inject_mouse_down(btn)
    elif action == glfw.RELEASE:
        renderer.inject_mouse_up(btn)

def on_scroll(win, dx, dy):
    renderer.inject_scroll(dx, dy)

glfw.set_framebuffer_size_callback(window, on_resize)
glfw.set_cursor_pos_callback(window, on_cursor)
glfw.set_mouse_button_callback(window, on_mouse_btn)
glfw.set_scroll_callback(window, on_scroll)

# 渲染循环
last_time = glfw.get_time()
while not glfw.window_should_close(window):
    glfw.poll_events()
    now = glfw.get_time()
    dt = now - last_time
    last_time = now

    cube.rotate_y(dt * 0.5)
    renderer.frame(dt)

renderer.dispose()
glfw.terminate()
```

### PySide6 (Qt) 集成

将 Myth 嵌入 Qt 应用：

```python
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication, QWidget
import myth

class MythWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)

        self.renderer = myth.Renderer(render_path="hdr", vsync=True)
        self._initialized = False

        # 60fps 定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._render)
        self._timer.start(16)

    def _ensure_init(self):
        if self._initialized:
            return
        hwnd = int(self.winId())
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        self.renderer.init_with_handle(hwnd, w, h)
        self._setup_scene()
        self._initialized = True

    def _setup_scene(self):
        scene = self.renderer.create_scene()
        # ... 添加网格、相机、灯光

    def _render(self):
        self._ensure_init()
        if self._initialized:
            self.renderer.frame()

    # 转发输入事件
    def resizeEvent(self, event):
        if self._initialized:
            sz = event.size()
            self.renderer.resize(sz.width(), sz.height())

    def mouseMoveEvent(self, event):
        pos = event.position()
        self.renderer.inject_mouse_move(pos.x(), pos.y())

    def mousePressEvent(self, event):
        btn_map = {Qt.MouseButton.LeftButton: 0,
                   Qt.MouseButton.MiddleButton: 1,
                   Qt.MouseButton.RightButton: 2}
        btn = btn_map.get(event.button())
        if btn is not None:
            self.renderer.inject_mouse_down(btn)

    def mouseReleaseEvent(self, event):
        btn_map = {Qt.MouseButton.LeftButton: 0,
                   Qt.MouseButton.MiddleButton: 1,
                   Qt.MouseButton.RightButton: 2}
        btn = btn_map.get(event.button())
        if btn is not None:
            self.renderer.inject_mouse_up(btn)

    def wheelEvent(self, event):
        delta = event.angleDelta()
        self.renderer.inject_scroll(delta.x() / 120.0, delta.y() / 120.0)

    def closeEvent(self, event):
        self._timer.stop()
        self.renderer.dispose()
        super().closeEvent(event)

    def paintEngine(self):
        return None  # wgpu 直接渲染
```

### RenderCanvas 集成

使用 rendercanvas 作为窗口后端：

```python
import myth
from rendercanvas.glfw import GlfwRenderCanvas

canvas = GlfwRenderCanvas(title="Myth + RenderCanvas", size=(1280, 720))

renderer = myth.Renderer(render_path=myth.RenderPath.BASIC)
# ... 初始化和场景设置

# 转发 rendercanvas 事件到 Myth
@canvas.add_event_handler("pointer_move")
def on_move(event):
    pr = event.get("pixel_ratio", 1.0)
    renderer.inject_mouse_move(event["x"] * pr, event["y"] * pr)

@canvas.add_event_handler("pointer_down")
def on_down(event):
    renderer.inject_mouse_down(0)

# 渲染循环
while not canvas.get_closed():
    canvas._process_events()
    renderer.frame(dt)

renderer.dispose()
```

---

## 渲染路径

Myth 提供两种渲染路径：

### Basic（基础）

```python
app = myth.App(render_path=myth.RenderPath.BASIC)
```

- 前向渲染 + MSAA
- LDR 输出
- 适合简单场景、快速原型

### High Fidelity（高保真）

```python
app = myth.App(render_path=myth.RenderPath.HIGH_FIDELITY)
```

- HDR 渲染管线
- 支持后处理：Bloom、SSAO、色调映射、FXAA
- 适合照片级渲染、产品可视化

> 也可使用旧版字符串 `'basic'`、`'hdr'`、`'high_fidelity'`。

---

## 最佳实践

### 1. 场景初始化与更新分离

将场景构建逻辑放在 `@app.init`，每帧逻辑放在 `@app.update`：

```python
@app.init
def on_init(ctx):
    # 创建场景、添加对象 — 只执行一次

@app.update
def on_update(ctx, frame):
    # 动画、输入处理 — 每帧执行
```

### 2. 使用全局变量持有节点引用

`@app.init` 和 `@app.update` 是独立回调，需要通过全局变量共享 `Object3D` 引用：

```python
cube = None
cam = None

@app.init
def on_init(ctx):
    global cube, cam
    scene = ctx.create_scene()
    cube = scene.add_mesh(...)
    cam = scene.add_camera(...)

@app.update
def on_update(ctx, frame):
    cube.rotate_y(frame.dt)        # 使用全局引用
    orbit.update(cam, frame.dt)
```

### 3. 使用 delta_time 实现帧率无关动画

始终使用 `frame.dt`（而非固定值）来驱动动画，确保在不同帧率下表现一致：

```python
@app.update
def on_update(ctx, frame):
    cube.rotate_y(frame.dt * speed)  # ✓ 帧率无关
    # cube.rotate_y(0.01)           # ✗ 帧率相关
```

### 4. 合理选择渲染路径

- 简单场景 / 快速原型 → `RenderPath.BASIC`
- 高质量渲染 / 需要后处理 → `RenderPath.HIGH_FIDELITY`

### 5. FPS 计数器

```python
fps_time = 0.0
fps_frames = 0

@app.update
def on_update(ctx, frame):
    global fps_time, fps_frames
    fps_time += frame.dt
    fps_frames += 1
    if fps_time >= 0.5:
        fps = fps_frames / fps_time
        ctx.set_title(f"My App | FPS: {fps:.1f}")
        fps_time = 0.0
        fps_frames = 0
```

---

## 常见问题

### Q: 窗口出现但一片黑色

检查是否设置了活跃相机：

```python
scene.active_camera = cam_node
```

同时确保场景中有灯光（`UnlitMaterial` 除外）。

### Q: 后处理效果没有生效

确保使用高保真渲染路径：

```python
app = myth.App(render_path=myth.RenderPath.HIGH_FIDELITY)
# 或
app = myth.App(render_path="hdr")
```

### Q: 嵌入 Qt 窗口时画面不更新

确保：
1. 设置了 `WA_PaintOnScreen` 和 `WA_NativeWindow`
2. `paintEngine()` 返回 `None`
3. 使用 `QTimer` 定期调用 `renderer.frame()`

### Q: 鼠标交互在 Renderer 模式下不工作

`Renderer` 模式需要手动注入输入事件：

```python
renderer.inject_mouse_move(x, y)
renderer.inject_mouse_down(button)
renderer.inject_mouse_up(button)
renderer.inject_scroll(dx, dy)
renderer.inject_key_down(key)
renderer.inject_key_up(key)
```

### Q: 如何让平面水平放置？

`PlaneGeometry` 默认在 XY 平面上，旋转 -90° 使其水平：

```python
ground = scene.add_mesh(
    myth.PlaneGeometry(width=20, height=20),
    myth.PhysicalMaterial(color="#666666"),
)
ground.rotation_euler = [-90, 0, 0]
```
