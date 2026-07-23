# Python 绑定

除了原生的 Rust API，Myth 还提供了一套 **Python 绑定**，让你能用极少的代码完成快速原型设计与科研可视化——既享受 Python 的便捷，又获得 Rust + wgpu 的底层性能。

## 1. 安装

Python 绑定位于仓库的 [`bindings/python`](https://github.com/panxinmiao/myth/tree/main/bindings/python) 目录，基于 PyO3 构建。请参考该目录下的说明完成安装（通常使用 `maturin` 进行本地构建）。

## 2. 心智模型

Python API 在保留 Myth 核心概念（Scene、Node、Camera、Light、Material）的同时，提供了更符合 Python 直觉的接口风格：使用关键字参数、列表表示向量、装饰器注册回调。

```python
import myth

app = myth.App(
    title="Shadows & Skinning",
    render_path=myth.RenderPath.BASIC,
    vsync=False,
)
```

## 3. 一个完整示例

下面的例子加载一个带阴影的蒙皮角色模型，并播放其动画：

```python
import myth

app = myth.App(title="Shadows & Skinning", render_path=myth.RenderPath.BASIC)
orbit = myth.OrbitControls(position=[0, 1.5, 4], target=[0, 1, 0])
cam = None


@app.init
def on_init(ctx: myth.Engine):
    global cam
    scene = ctx.create_scene()

    # 环境贴图（IBL）
    env_tex = ctx.load_texture("envs/studio.hdr.jpg", color_space="srgb")
    scene.set_environment_map(env_tex)

    # 投射阴影的方向光
    sun = scene.add_light(
        myth.DirectionalLight(color=[1, 1, 1], intensity=5.0, cast_shadows=True)
    )
    sun.position = [0, 12, 6]
    sun.look_at([0, 0, 0])

    # 接收阴影的地面
    ground = scene.add_mesh(
        myth.PlaneGeometry(width=30, height=30),
        myth.PhongMaterial(color=[0.2, 0.3, 0.4], side="double"),
    )
    ground.rotation_euler = [-90, 0, 0]
    ground.receive_shadows = True

    # 加载并播放动画
    model = ctx.load_gltf("Michelle.glb")
    anims = scene.list_animations(model)
    if anims:
        scene.play_animation(model, anims[0])

    # 相机
    cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.1))
    cam.position = [0, 1.5, 4]
    cam.look_at([0, 1, 0])
    scene.active_camera = cam


@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    orbit.update(cam, frame.dt)


app.run()
```

## 4. 适用场景

- **科研与数据可视化：** 在 Python 生态（NumPy / 数据管线）中嵌入高性能的 3D 渲染。
- **快速原型：** 无需编译即可迭代场景布局、相机与光照。
- **教学与演示：** 用最少的样板代码讲解现代渲染概念。

更多示例与完整 API，请参考仓库中的 [Python Bindings 目录](https://github.com/panxinmiao/myth/tree/main/bindings/python)。
