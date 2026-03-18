# Myth Engine Python — API 参考手册

> **Myth Engine** 是基于 wgpu 的高性能 3D 渲染引擎。本文档为其 Python 绑定的完整 API 参考。

---

## 目录

- [类型别名](#类型别名)
- [枚举](#枚举)
  - [RenderPath](#renderpath)
- [应用层](#应用层)
  - [App](#app)
  - [Renderer](#renderer)
- [引擎上下文](#引擎上下文)
  - [Engine](#engine)
  - [FrameState](#framestate)
- [场景](#场景)
  - [Scene](#scene)
- [场景节点](#场景节点)
  - [Object3D](#object3d)
- [几何体](#几何体)
  - [BoxGeometry](#boxgeometry)
  - [SphereGeometry](#spheregeometry)
  - [PlaneGeometry](#planegeometry)
  - [Geometry](#geometry)
- [材质](#材质)
  - [UnlitMaterial](#unlitmaterial)
  - [PhongMaterial](#phongmaterial)
  - [PhysicalMaterial](#physicalmaterial)
- [相机](#相机)
  - [PerspectiveCamera](#perspectivecamera)
  - [OrthographicCamera](#orthographiccamera)
- [灯光](#灯光)
  - [DirectionalLight](#directionallight)
  - [PointLight](#pointlight)
  - [SpotLight](#spotlight)
- [组件代理](#组件代理)
  - [PerspectiveCameraComponent](#perspectivecameracomponent)
  - [OrthographicCameraComponent](#orthographiccameracomponent)
  - [DirectionalLightComponent](#directionallightcomponent)
  - [PointLightComponent](#pointlightcomponent)
  - [SpotLightComponent](#spotlightcomponent)
  - [MeshComponent](#meshcomponent)
- [纹理](#纹理)
  - [TextureHandle](#texturehandle)
- [控制器](#控制器)
  - [OrbitControls](#orbitcontrols)
- [输入](#输入)
  - [Input](#input)
- [动画](#动画)
  - [AnimationMixer](#animationmixer)

---

## 类型别名

| 别名 | 定义 | 说明 |
|------|------|------|
| `Color` | `str \| list[float] \| tuple[float, float, float]` | 颜色值：十六进制字符串 `'#RRGGBB'`、`[r, g, b]` 列表或 `(r, g, b)` 元组 |
| `Vec3` | `list[float] \| tuple[float, float, float]` | 三维向量：`[x, y, z]` |

---

## 枚举

### RenderPath

渲染管线路径。

| 值 | 说明 |
|----|------|
| `RenderPath.BASIC` | 前向 LDR + MSAA |
| `RenderPath.HIGH_FIDELITY` | HDR + 后处理（Bloom、SSAO、色调映射等） |

也可传递旧版字符串 `'basic'`、`'hdr'`、`'high_fidelity'`。

---

## 应用层

### App

主应用类。管理窗口创建、事件循环和渲染管线。

```python
app = myth.App(
    title: str = "Myth Engine",
    render_path: str | RenderPath = RenderPath.BASIC,
    vsync: bool = True,
    clear_color: list[float] = [0.0, 0.0, 0.0, 1.0],
)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `title` | `str` | 窗口标题 |
| `render_path` | `str \| RenderPath` | 渲染路径 |
| `vsync` | `bool` | 是否启用垂直同步 |
| `clear_color` | `list[float]` | 清屏颜色 `[r, g, b, a]` |

#### 方法

##### `app.init(func) → Callable`

注册初始化回调函数。通常以装饰器形式使用。

```python
@app.init
def on_init(ctx: myth.Engine):
    scene = ctx.create_scene()
    # ... 构建场景
```

- **参数**: `func` — 签名为 `(Engine) -> None` 的回调函数
- **返回**: 原始函数（透传装饰器）

##### `app.update(func) → Callable`

注册每帧更新回调函数。

```python
@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    cube.rotate_y(frame.dt * 0.5)
```

- **参数**: `func` — 签名为 `(Engine, FrameState) -> None` 的回调函数
- **返回**: 原始函数

##### `app.run() → None`

启动应用（阻塞）。进入主事件循环，直到窗口关闭。

---

### Renderer

底层、GUI 无关的渲染器。当需要将 Myth 嵌入外部窗口系统（GLFW、PySide6、wxPython、SDL2 等）时使用此类代替 `App`。

```python
renderer = myth.Renderer(
    render_path: str | RenderPath = RenderPath.BASIC,
    vsync: bool = True,
    clear_color: list[float] = [0.0, 0.0, 0.0, 1.0],
)
```

支持上下文管理器协议：

```python
with myth.Renderer(render_path="hdr") as renderer:
    renderer.init_with_handle(hwnd, 1280, 720)
    # ...
# 自动调用 dispose()
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `render_path` | `str \| RenderPath` | 渲染路径 |
| `vsync` | `bool` | 是否启用垂直同步 |
| `time` | `float` *(只读)* | 自启动以来的总时间（秒） |
| `frame_count` | `int` *(只读)* | 已渲染总帧数 |
| `input` | `Input` *(只读)* | 输入状态代理 |

#### 初始化与生命周期

##### `renderer.init_with_handle(window_handle, width, height) → None`

使用原生平台窗口句柄初始化 GPU。

- **参数**:
  - `window_handle: int` — 平台特定的窗口句柄
    - **Windows**: HWND（`glfw.get_win32_window()` 或 `int(widget.winId())`）
    - **macOS**: NSView 指针
    - **Linux/X11**: X11 Window ID
  - `width: int` — 初始帧缓冲宽度（像素）
  - `height: int` — 初始帧缓冲高度（像素）

##### `renderer.resize(width, height, scale_factor=1.0) → None`

通知渲染器窗口大小已改变。

##### `renderer.dispose() → None`

释放所有 GPU 资源。

#### 渲染控制

##### `renderer.update(dt=None) → None`

推进引擎状态。若 `dt` 为 `None`，则自动从挂钟时间计算。

##### `renderer.render() → None`

渲染一帧并呈现到表面。

##### `renderer.frame(dt=None) → None`

`update()` + `render()` 的便捷合并调用。

#### 场景 / 资产管理

与 `Engine` 类具有相同的场景 API：

##### `renderer.create_scene() → Scene`

创建新场景并设为活跃场景。

##### `renderer.active_scene() → Scene | None`

获取当前活跃场景。

##### `renderer.load_texture(path, color_space="srgb", generate_mipmaps=True) → TextureHandle`

加载 2D 纹理。

##### `renderer.load_hdr_texture(path) → TextureHandle`

加载 HDR 环境纹理（`.hdr` 文件）。

##### `renderer.load_gltf(path) → Object3D`

加载 glTF/GLB 模型，返回根节点。

#### 输入注入

当使用外部窗口系统时，需要将输入事件转发给渲染器：

| 方法 | 说明 |
|------|------|
| `inject_key_down(key: str)` | 注入按键按下事件 |
| `inject_key_up(key: str)` | 注入按键释放事件 |
| `inject_mouse_move(x: float, y: float)` | 注入鼠标移动事件 |
| `inject_mouse_down(button: int)` | 注入鼠标按下（0=左, 1=中, 2=右） |
| `inject_mouse_up(button: int)` | 注入鼠标释放 |
| `inject_scroll(dx: float, dy: float)` | 注入滚轮事件 |

---

## 引擎上下文

### Engine

引擎上下文，在 `@app.init` 和 `@app.update` 回调中可用。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `time` | `float` *(只读)* | 自启动以来的总时间（秒） |
| `frame_count` | `int` *(只读)* | 已渲染总帧数 |
| `input` | `Input` *(只读)* | 输入状态代理 |

#### 方法

##### `engine.create_scene() → Scene`

创建新场景并设为活跃场景。

##### `engine.active_scene() → Scene | None`

获取当前活跃场景。

##### `engine.load_texture(path, color_space="srgb", generate_mipmaps=True) → TextureHandle`

加载 2D 纹理。

- **参数**:
  - `path: str` — 纹理图片文件路径
  - `color_space: str` — `'srgb'` 或 `'linear'`
  - `generate_mipmaps: bool` — 是否生成 Mipmap

##### `engine.load_hdr_texture(path) → TextureHandle`

加载 HDR 环境纹理（如 `.hdr` 文件）。

##### `engine.load_gltf(path) → Object3D`

加载 glTF/GLB 模型并实例化到活跃场景中。返回模型的根 `Object3D` 节点。

##### `engine.set_title(title) → None`

设置窗口标题（仅在使用 `App` 时有效）。

---

### FrameState

每帧状态信息，传入 `@app.update` 回调。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `delta_time` | `float` *(只读)* | 距上一帧的时间间隔（秒） |
| `dt` | `float` *(只读)* | `delta_time` 的别名 |
| `elapsed` | `float` *(只读)* | 自应用启动以来的总时间（秒） |
| `time` | `float` *(只读)* | `elapsed` 的别名 |
| `frame_count` | `int` *(只读)* | 总帧数 |

---

## 场景

### Scene

场景容器，持有物体、灯光、相机和环境设置。通过 `engine.create_scene()` 获取。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `active_camera` | `Object3D \| None` | 当前活跃相机节点 |

#### 添加对象

##### `scene.add_mesh(geometry, material) → Object3D`

向场景添加网格（Mesh）。

- **参数**:
  - `geometry` — 几何体对象（`BoxGeometry`、`SphereGeometry`、`PlaneGeometry` 或 `Geometry`）
  - `material` — 材质对象（`UnlitMaterial`、`PhongMaterial` 或 `PhysicalMaterial`）
- **返回**: 新创建的 `Object3D` 网格节点

##### `scene.add_camera(camera) → Object3D`

向场景添加相机。

- **参数**: `camera` — `PerspectiveCamera` 或 `OrthographicCamera`
- **返回**: `Object3D` 节点。通过 `scene.active_camera = node` 设为活跃相机。

##### `scene.add_light(light) → Object3D`

向场景添加灯光。

- **参数**: `light` — `DirectionalLight`、`PointLight` 或 `SpotLight`
- **返回**: `Object3D` 节点（通过 `.position` 定位灯光）

#### 场景层级

##### `scene.attach(child, parent) → None`

将子节点挂接到父节点。

##### `scene.find_node_by_name(name) → Object3D | None`

按名称查找节点。未找到时返回 `None`。

#### 背景与环境

##### `scene.set_background_color(r, g, b) → None`

设置纯色背景（各分量范围 0.0–1.0）。

```python
scene.set_background_color(0.05, 0.05, 0.1)
```

##### `scene.set_environment_map(tex) → None`

设置环境贴图，用于基于图像的光照（IBL）和天空盒。

```python
hdr = ctx.load_hdr_texture("env.hdr")
scene.set_environment_map(hdr)
```

##### `scene.set_environment_intensity(intensity) → None`

设置环境光照强度。

##### `scene.set_ambient_light(r, g, b) → None`

设置环境光颜色。

#### 后处理

> **注意**: 后处理效果需使用 `RenderPath.HIGH_FIDELITY`（或 `'hdr'`）渲染路径。

##### 色调映射（Tone Mapping）

```python
scene.set_tone_mapping_mode(mode: str) → None
scene.set_tone_mapping(mode: str, exposure: float | None = None) → None
```

支持的模式：`'linear'`、`'neutral'`、`'reinhard'`、`'cineon'`、`'aces'` / `'aces_filmic'`、`'agx'`

```python
scene.set_tone_mapping("aces", exposure=1.2)
```

##### Bloom

```python
scene.set_bloom_enabled(enabled: bool) → None
scene.set_bloom_strength(strength: float) → None   # 例如 0.04
scene.set_bloom_radius(radius: float) → None       # 例如 0.005

# 便捷方法
scene.set_bloom(enabled: bool, strength: float | None = None, radius: float | None = None) → None
```

```python
scene.set_bloom(True, strength=0.04, radius=0.3)
```

##### SSAO（屏幕空间环境光遮蔽）

```python
scene.set_ssao_enabled(enabled: bool) → None
scene.set_ssao_radius(radius: float) → None
scene.set_ssao_bias(bias: float) → None
scene.set_ssao_intensity(intensity: float) → None
```

#### 动画

##### `scene.play_animation(node, name) → None`

播放节点上指定名称的动画片段。

##### `scene.play_if_any_animation(node) → None`

播放节点上的任意可用动画（便捷方法）。

##### `scene.play_any_animation(node) → None`

`play_if_any_animation` 的别名。

##### `scene.list_animations(node) → list[str]`

列出节点上的所有动画片段名称。

##### `scene.get_animation_mixer(node) → AnimationMixer | None`

获取节点的动画混合器（高级控制）。

---

## 场景节点

### Object3D

场景中的 3D 对象（节点）。提供变换、可见性、阴影和命名控制。
通过组件代理访问器（`.camera`、`.light`、`.mesh`）可在运行时检查和修改节点上的 ECS 组件。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `position` | `list[float]` | 位置 `[x, y, z]` |
| `rotation` | `list[float]` | 欧拉旋转（弧度）`[x, y, z]`（XYZ 顺序） |
| `rotation_euler` | `list[float]` | 欧拉旋转（角度）`[x, y, z]`（XYZ 顺序） |
| `scale` | `list[float]` | 缩放 `[x, y, z]` |
| `visible` | `bool` | 是否可见 |
| `cast_shadows` | `bool` | 是否投射阴影（仅对网格有意义） |
| `receive_shadows` | `bool` | 是否接收阴影（仅对网格有意义） |
| `name` | `str \| None` | 节点名称 |
| `camera` | `PerspectiveCameraComponent \| OrthographicCameraComponent \| None` | 相机组件代理 |
| `light` | `DirectionalLightComponent \| PointLightComponent \| SpotLightComponent \| None` | 灯光组件代理 |
| `mesh` | `MeshComponent \| None` | 网格组件代理 |

#### 方法

##### `obj.set_uniform_scale(s) → None`

设置统一缩放（x、y、z 相同值）。

```python
model.set_uniform_scale(2.0)  # 放大两倍
```

##### `obj.rotate_x(angle) → None`

绕局部 X 轴旋转 `angle` 弧度。

##### `obj.rotate_y(angle) → None`

绕局部 Y 轴旋转 `angle` 弧度。

##### `obj.rotate_z(angle) → None`

绕局部 Z 轴旋转 `angle` 弧度。

##### `obj.look_at(target) → None`

旋转节点使其面朝世界空间中的目标位置。

- **参数**: `target: Vec3` — 目标位置 `[x, y, z]`

```python
cam.look_at([0, 0, 0])   # 朝向原点
sun.look_at([0, 0, 0])   # 灯光朝向原点
```

---

## 几何体

### BoxGeometry

立方体（长方体）几何体。

```python
myth.BoxGeometry(
    width: float = 1.0,    # X 轴宽度
    height: float = 1.0,   # Y 轴高度
    depth: float = 1.0,    # Z 轴深度
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `width` | `float` | X 轴宽度 |
| `height` | `float` | Y 轴高度 |
| `depth` | `float` | Z 轴深度 |

---

### SphereGeometry

球体几何体。

```python
myth.SphereGeometry(
    radius: float = 1.0,
    width_segments: int = 32,
    height_segments: int = 16,
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `radius` | `float` | 球体半径 |
| `width_segments` | `int` | 水平分段数 |
| `height_segments` | `int` | 垂直分段数 |

---

### PlaneGeometry

平面几何体。

```python
myth.PlaneGeometry(
    width: float = 1.0,    # X 轴宽度
    height: float = 1.0,   # Z 轴高度
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `width` | `float` | X 轴宽度 |
| `height` | `float` | Z 轴高度 |

---

### Geometry

自定义几何体，从原始顶点数据构建。

```python
geo = myth.Geometry()
```

#### 方法

##### `geo.set_positions(data) → None`

设置顶点位置。数据为扁平列表 `[x0, y0, z0, x1, y1, z1, ...]`。

##### `geo.set_normals(data) → None`

设置顶点法线。数据为扁平列表 `[nx0, ny0, nz0, ...]`。

##### `geo.set_uvs(data) → None`

设置 UV 坐标。数据为扁平列表 `[u0, v0, u1, v1, ...]`。

##### `geo.set_indices(data) → None`

设置三角形索引缓冲。

**示例**:

```python
geo = myth.Geometry()
geo.set_positions([
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
])
geo.set_normals([
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
])
geo.set_uvs([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
geo.set_indices([0, 1, 2])
```

---

## 材质

### UnlitMaterial

无光照（Unlit）材质，仅使用纯色。

```python
myth.UnlitMaterial(
    color: Color = "#ffffff",
    opacity: float = 1.0,
    side: str = "front",        # "front" | "back" | "double"
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `color` | `list[float]` | 漫反射颜色 `[r, g, b]`。可通过 `[r, g, b]`、`(r, g, b)` 或十六进制字符串设置 |
| `opacity` | `float` | 不透明度（0.0–1.0） |

#### 方法

##### `mat.set_map(tex) → None`

设置颜色（漫反射）纹理贴图。

---

### PhongMaterial

Blinn-Phong 材质，支持高光。

```python
myth.PhongMaterial(
    color: Color = "#ffffff",
    specular: Color = "#111111",
    emissive: Color = "#000000",
    shininess: float = 30.0,
    opacity: float = 1.0,
    side: str = "front",
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `color` | `list[float]` | 漫反射颜色 |
| `shininess` | `float` | 高光指数 |
| `opacity` | `float` | 不透明度 |

#### 方法

| 方法 | 说明 |
|------|------|
| `set_map(tex)` | 设置漫反射纹理贴图 |
| `set_normal_map(tex, scale=None)` | 设置法线贴图（可选缩放，默认 1.0） |

---

### PhysicalMaterial

PBR（物理）金属-粗糙度工作流材质。

```python
myth.PhysicalMaterial(
    color: Color = "#ffffff",
    metalness: float = 0.0,
    roughness: float = 0.5,
    emissive: Color = "#000000",
    emissive_intensity: float = 1.0,
    opacity: float = 1.0,
    side: str = "front",
)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `color` | `list[float]` | 基础颜色 `[r, g, b]` |
| `metalness` | `float` | 金属度（0.0–1.0） |
| `roughness` | `float` | 粗糙度（0.0–1.0） |
| `emissive_intensity` | `float` | 自发光强度乘数 |
| `opacity` | `float` | 不透明度（0.0–1.0） |
| `clearcoat` | `float` | 清漆涂层强度 |
| `clearcoat_roughness` | `float` | 清漆涂层粗糙度 |
| `transmission` | `float` | 透射率（用于玻璃等材质） |
| `ior` | `float` | 折射率 |

#### 方法

| 方法 | 说明 |
|------|------|
| `set_map(tex)` | 设置基础颜色纹理贴图 |
| `set_normal_map(tex, scale=None)` | 设置法线贴图 |
| `set_roughness_map(tex)` | 设置粗糙度纹理贴图 |
| `set_metalness_map(tex)` | 设置金属度纹理贴图 |
| `set_emissive_map(tex)` | 设置自发光纹理贴图 |
| `set_ao_map(tex)` | 设置环境光遮蔽纹理贴图 |

---

## 相机

### PerspectiveCamera

透视投影相机。

```python
myth.PerspectiveCamera(
    fov: float = 60.0,            # 垂直视场角（度）
    near: float = 0.1,            # 近裁剪面距离
    far: float = 1000.0,          # 远裁剪面距离
    aspect: float = 0.0,          # 宽高比（0 = 自动检测）
    position: list[float] = ...,  # 初始位置 [x, y, z]
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `fov` | `float` | 垂直视场角（度） |
| `aspect` | `float` | 宽高比（0 = 自动） |
| `near` | `float` | 近裁剪面 |
| `far` | `float` | 远裁剪面 |
| `position` | `list[float]` | 初始位置 |

---

### OrthographicCamera

正交投影相机。

```python
myth.OrthographicCamera(
    size: float = 10.0,           # 正交视图高度
    near: float = 0.1,
    far: float = 1000.0,
    position: list[float] = ...,
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `size` | `float` | 正交视图高度 |
| `near` | `float` | 近裁剪面 |
| `far` | `float` | 远裁剪面 |
| `position` | `list[float]` | 初始位置 |

---

## 灯光

### DirectionalLight

平行光（模拟太阳光）。光线方向由节点的朝向决定。

```python
myth.DirectionalLight(
    color: list[float] = [1.0, 1.0, 1.0],
    intensity: float = 1.0,
    cast_shadows: bool = False,
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `color` | `list[float]` | 光照颜色 `[r, g, b]` |
| `intensity` | `float` | 光照强度乘数 |
| `cast_shadows` | `bool` | 是否投射阴影 |

```python
sun = scene.add_light(myth.DirectionalLight(intensity=3.0, cast_shadows=True))
sun.position = [5, 10, 5]
sun.look_at([0, 0, 0])
```

---

### PointLight

点光源，向所有方向发射光线。

```python
myth.PointLight(
    color: list[float] = [1.0, 1.0, 1.0],
    intensity: float = 1.0,
    range: float = 10.0,       # 最大范围（0 = 无限）
    cast_shadows: bool = False,
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `color` | `list[float]` | 光照颜色 |
| `intensity` | `float` | 光照强度 |
| `range` | `float` | 最大照射范围 |
| `cast_shadows` | `bool` | 是否投射阴影 |

---

### SpotLight

聚光灯，以锥形发射光线。

```python
myth.SpotLight(
    color: list[float] = [1.0, 1.0, 1.0],
    intensity: float = 1.0,
    range: float = 10.0,
    inner_cone: float = 0.3,    # 内锥角（弧度）
    outer_cone: float = 0.5,    # 外锥角（弧度）
    cast_shadows: bool = False,
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `color` | `list[float]` | 光照颜色 |
| `intensity` | `float` | 光照强度 |
| `range` | `float` | 最大照射范围 |
| `inner_cone` | `float` | 内锥角（弧度） |
| `outer_cone` | `float` | 外锥角（弧度） |
| `cast_shadows` | `bool` | 是否投射阴影 |

---

## 组件代理

组件代理是通过 `Object3D.camera`、`Object3D.light` 和 `Object3D.mesh` 属性返回的轻量级句柄。它们提供对场景节点上 ECS 组件的直接读写访问。返回的具体类型取决于组件变体。

### PerspectiveCameraComponent

当节点携带透视相机时，由 `node.camera` 返回。

| 属性 | 类型 | 说明 |
|------|------|------|
| `fov` | `float` | 垂直视场角（度） |
| `aspect` | `float` | 宽高比 |
| `near` | `float` | 近裁剪面 |
| `far` | `float` | 远裁剪面 |
| `antialiasing` | `AntiAliasing` | 抗锯齿配置 |

### OrthographicCameraComponent

当节点携带正交相机时，由 `node.camera` 返回。

| 属性 | 类型 | 说明 |
|------|------|------|
| `size` | `float` | 正交视图半高 |
| `near` | `float` | 近裁剪面 |
| `far` | `float` | 远裁剪面 |
| `antialiasing` | `AntiAliasing` | 抗锯齿配置 |

### DirectionalLightComponent

当节点携带平行光时，由 `node.light` 返回。

| 属性 | 类型 | 说明 |
|------|------|------|
| `color` | `list[float]` | 光照颜色 `[r, g, b]` |
| `intensity` | `float` | 光照强度（勒克斯） |
| `cast_shadows` | `bool` | 是否投射阴影 |

### PointLightComponent

当节点携带点光源时，由 `node.light` 返回。

| 属性 | 类型 | 说明 |
|------|------|------|
| `color` | `list[float]` | 光照颜色 `[r, g, b]` |
| `intensity` | `float` | 光照强度（坎德拉） |
| `range` | `float` | 最大有效范围 |
| `cast_shadows` | `bool` | 是否投射阴影 |

### SpotLightComponent

当节点携带聚光灯时，由 `node.light` 返回。

| 属性 | 类型 | 说明 |
|------|------|------|
| `color` | `list[float]` | 光照颜色 `[r, g, b]` |
| `intensity` | `float` | 光照强度（坎德拉） |
| `range` | `float` | 最大照射范围 |
| `inner_cone` | `float` | 内锥角（弧度） |
| `outer_cone` | `float` | 外锥角（弧度） |
| `cast_shadows` | `bool` | 是否投射阴影 |

### MeshComponent

当节点携带网格时，由 `node.mesh` 返回。

| 属性 | 类型 | 说明 |
|------|------|------|
| `visible` | `bool` | 网格可见性 |
| `cast_shadows` | `bool` | 是否投射阴影 |
| `receive_shadows` | `bool` | 是否接收阴影 |
| `render_order` | `int` | 绘制顺序覆盖 |

**示例 — 运行时组件修改：**

```python
# 运行时调整相机抗锯齿
cam_node.camera.antialiasing = myth.AntiAliasing.taa(feedback_weight=0.9)

# 动态修改灯光强度
sun.light.intensity = 5.0

# 修改网格阴影标志
cube.mesh.cast_shadows = False
```

---

## 纹理

### TextureHandle

已加载纹理的不透明句柄。

通过 `engine.load_texture()` 或 `engine.load_hdr_texture()` 获取，传递给材质的 `set_map()` 等方法使用。

```python
tex = ctx.load_texture("diffuse.png", color_space="srgb", generate_mipmaps=True)
mat.set_map(tex)

hdr = ctx.load_hdr_texture("env.hdr")
scene.set_environment_map(hdr)
```

---

## 控制器

### OrbitControls

Three.js 风格的轨道相机控制器。鼠标左键旋转、右键/Shift+左键平移、滚轮缩放。

```python
myth.OrbitControls(
    position: list[float] = [0.0, 0.0, 5.0],   # 初始相机位置
    target: list[float] = [0.0, 0.0, 0.0],      # 环绕目标点
)
```

#### 属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_damping` | `bool` | — | 启用阻尼（惯性） |
| `damping_factor` | `float` | — | 阻尼系数 |
| `rotate_speed` | `float` | — | 旋转速度 |
| `zoom_speed` | `float` | — | 缩放速度 |
| `pan_speed` | `float` | — | 平移速度 |
| `min_distance` | `float` | — | 最小距离 |
| `max_distance` | `float` | — | 最大距离 |

#### 方法

##### `orbit.update(camera, dt) → None`

更新轨道控制器。**每帧**在 `@app.update` 中调用。

- **参数**:
  - `camera: Object3D` — 相机节点
  - `dt: float` — 帧间隔时间（秒），来自 `frame.delta_time`

##### `orbit.set_target(target) → None`

设置环绕目标点 `[x, y, z]`。

**示例**:

```python
orbit = myth.OrbitControls(position=[3, 3, 5], target=[0, 0.5, 0])

@app.update
def on_update(ctx, frame):
    orbit.update(cam, frame.dt)
```

---

## 输入

### Input

只读输入状态代理。通过 `engine.input` 在 `@app.update` 回调中访问。

#### 键盘方法

| 方法 | 说明 |
|------|------|
| `key(name: str) → bool` | 按键是否正在按住 |
| `key_down(name: str) → bool` | 按键是否在此帧首次按下 |
| `key_up(name: str) → bool` | 按键是否在此帧释放 |

**按键名称**: `'a'`–`'z'`、`'0'`–`'9'`、`'Space'`、`'Enter'`、`'Escape'`、`'Tab'`、`'Shift'`、`'Ctrl'`、`'Alt'`、`'ArrowUp'`、`'ArrowDown'`、`'ArrowLeft'`、`'ArrowRight'`、`'F1'`–`'F12'`

#### 鼠标方法

| 方法 | 说明 |
|------|------|
| `mouse_button(name: str) → bool` | 鼠标按钮是否正在按住 |
| `mouse_button_down(name: str) → bool` | 鼠标按钮是否在此帧首次按下 |
| `mouse_button_up(name: str) → bool` | 鼠标按钮是否在此帧释放 |
| `mouse_position() → list[float]` | 当前鼠标位置 `[x, y]`（窗口像素） |
| `mouse_delta() → list[float]` | 自上一帧的鼠标移动增量 `[dx, dy]` |
| `scroll_delta() → list[float]` | 自上一帧的滚轮增量 `[dx, dy]` |

**鼠标按钮名称**: `'Left'`、`'Right'`、`'Middle'`

**示例**:

```python
@app.update
def on_update(ctx, frame):
    inp = ctx.input

    if inp.key("w"):
        # W 键按住时前进
        pass

    if inp.key_down("Space"):
        # 空格键刚按下
        pass

    if inp.mouse_button("Left"):
        dx, dy = inp.mouse_delta()
        # 鼠标左键拖动
```

---

## 动画

### AnimationMixer

动画混合器，挂载到特定节点，提供高级动画控制。通过 `scene.get_animation_mixer(node)` 获取。

#### 方法

| 方法 | 说明 |
|------|------|
| `list_animations() → list[str]` | 列出所有可用的动画片段名称 |
| `play(name: str)` | 按名称播放动画 |
| `stop(name: str)` | 按名称停止动画 |
| `stop_all()` | 停止所有动画 |

**示例**:

```python
model = ctx.load_gltf("character.glb")
mixer = scene.get_animation_mixer(model)

if mixer:
    anims = mixer.list_animations()
    print(f"可用动画: {anims}")

    mixer.play("Walk")
    # 之后切换动画
    mixer.stop("Walk")
    mixer.play("Run")
```
