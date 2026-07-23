# Python Bindings

In addition to the native Rust API, Myth provides a set of **Python bindings**, letting you do rapid prototyping and scientific visualization with minimal code — enjoying both Python's convenience and the low-level performance of Rust + wgpu.

## 1. Installation

The Python bindings live in the repository's [`bindings/python`](https://github.com/panxinmiao/myth/tree/main/bindings/python) directory and are built with PyO3. Follow the instructions in that directory to install (typically a local build with `maturin`).

## 2. Mental Model

The Python API preserves Myth's core concepts (Scene, Node, Camera, Light, Material) while offering a more Pythonic interface: keyword arguments, lists for vectors, and decorators for registering callbacks.

```python
import myth

app = myth.App(
    title="Shadows & Skinning",
    render_path=myth.RenderPath.BASIC,
    vsync=False,
)
```

## 3. A Complete Example

The example below loads a skinned character with shadows and plays its animation:

```python
import myth

app = myth.App(title="Shadows & Skinning", render_path=myth.RenderPath.BASIC)
orbit = myth.OrbitControls(position=[0, 1.5, 4], target=[0, 1, 0])
cam = None


@app.init
def on_init(ctx: myth.Engine):
    global cam
    scene = ctx.create_scene()

    # Environment map (IBL)
    env_tex = ctx.load_texture("envs/studio.hdr.jpg", color_space="srgb")
    scene.set_environment_map(env_tex)

    # Shadow-casting directional light
    sun = scene.add_light(
        myth.DirectionalLight(color=[1, 1, 1], intensity=5.0, cast_shadows=True)
    )
    sun.position = [0, 12, 6]
    sun.look_at([0, 0, 0])

    # Shadow-receiving ground plane
    ground = scene.add_mesh(
        myth.PlaneGeometry(width=30, height=30),
        myth.PhongMaterial(color=[0.2, 0.3, 0.4], side="double"),
    )
    ground.rotation_euler = [-90, 0, 0]
    ground.receive_shadows = True

    # Load and play animation
    model = ctx.load_gltf("Michelle.glb")
    anims = scene.list_animations(model)
    if anims:
        scene.play_animation(model, anims[0])

    # Camera
    cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.1))
    cam.position = [0, 1.5, 4]
    cam.look_at([0, 1, 0])
    scene.active_camera = cam


@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    orbit.update(cam, frame.dt)


app.run()
```

## 4. Use Cases

- **Science & data visualization:** Embed high-performance 3D rendering inside the Python ecosystem (NumPy / data pipelines).
- **Rapid prototyping:** Iterate on scene layout, camera, and lighting without compilation.
- **Teaching & demos:** Explain modern rendering concepts with minimal boilerplate.

For more examples and the full API, see the [Python Bindings directory](https://github.com/panxinmiao/myth/tree/main/bindings/python) in the repository.
