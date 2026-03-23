# Myth: High-Performance Python 3D Rendering Engine 🚀

<p align="center">
  <strong>Official Python bindings for <a href="https://github.com/panxinmiao/myth">Myth Engine</a></strong><br>
  Combining the raw rendering power of <b>Rust + WGPU</b> with a modern, <b>Three.js-style</b> API.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/rust-2024%20edition-orange?logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/graphics-wgpu-green" alt="wgpu">
  <img src="https://img.shields.io/badge/build-maturin-purple" alt="maturin">
</p>

---

Whether you need to build physics simulation environments for AI training, render complex
data visualizations, or develop real-time interactive 3D digital humans, `myth-py` delivers
a silky-smooth development experience with industrial-grade rendering quality.

## Core Features

- **Extreme Performance (Powered by Rust)** — Core engine written in Rust on top of next-generation graphics API (WGPU), delivering throughput on par with C++ engines.
- **Easy-to-use object-oriented API** — Expressive scene graph, materials, and geometries. Frontend developers and Python engineers can get started instantly.
- **Modern Rendering Pipeline** — Built-in PBR, HDR environment lighting (IBL), bloom, SSAO, AgX/ACES tone mapping, and state-of-the-art temporal anti-aliasing (TAA).
- **ECS Dynamic Components** — Runtime-modifiable node components via `.camera`, `.light`, `.mesh` proxies, seamlessly integrating with AI logic.
- **Perfect IDE Experience** — Full `.pyi` type stubs for flawless autocompletion and type inference in VSCode and PyCharm.

## 📦 Installation

Supports **Windows**, **macOS**, and **Linux**. Install with pip:

```bash
pip install myth-py
```

### Build from Source

```bash
# 1. Install Rust toolchain: https://rustup.rs
# 2. Clone the repository
git clone https://github.com/panxinmiao/myth.git
cd myth/bindings/python

# 3. Install maturin and build
pip install maturin
maturin develop --release
```

## 🚀 Quickstart

Under 30 lines of code to launch your first Animating 3D Scene:

```python
import myth

app = myth.App(title="Myth-py Quickstart", render_path=myth.RenderPath.BASIC)

cam_node = None
cube_node = None
orbit = myth.OrbitControls()


@app.init
def setup(ctx: myth.Engine):
    global cam_node, cube_node
    scene = ctx.create_scene()

    # Create a PBR metal cube
    geo = myth.BoxGeometry(width=1.0, height=1.0, depth=1.0)
    mat = myth.PhysicalMaterial(color="#5100ff", roughness=0.2, metalness=0.8)
    cube_node = scene.add_mesh(geo, mat)

    # Add camera and activate it
    cam_node = scene.add_camera(myth.PerspectiveCamera(fov=45.0, position=[0, 2, 5]))
    cam_node.look_at([0, 0, 0])
    scene.active_camera = cam_node

    # Directional sunlight with shadows
    sun = scene.add_light(myth.DirectionalLight(color=[1.0, 0.9, 0.8], intensity=3.0))
    sun.position = [5, 5, 2]
    sun.look_at([0, 0, 0])

    scene.set_ambient_light(0.1, 0.1, 0.15)
    scene.set_background_color(0.1, 0.1, 0.15)


@app.update
def on_frame(ctx: myth.Engine, frame: myth.FrameState):
    orbit.update(cam_node, frame.dt)
    cube_node.rotation = [frame.time * 0.3, frame.time * 0.5, 0]


app.run()
```

## 📚 Advanced Features

### One-Click glTF Animation Loading

```python
robot = ctx.load_gltf("assets/robot.glb")
scene.play_if_any_animation(robot)
```

### Dynamic Component Interaction

Thanks to the underlying ECS architecture, you can safely inspect and modify node
components at any time:

```python
light_node = scene.add_light(myth.PointLight(color="#ffffff"))

# In some runtime logic branch:
if light_node.light:
    light_node.light.intensity = 5.0   # Instantly illuminate the scene
    light_node.light.range = 20.0      # Extend the light range
```

### Two Usage Modes

| Mode | Entry Point | Use Case |
|:---|:---|:---|
| **`App`** | `myth.App()` | Quick prototyping — built-in window + event loop |
| **`Renderer`** | `myth.Renderer()` | Embed in **glfw**, **Qt**, **wxPython**, **rendercanvas**, or any window with a native handle |

## Examples

The `examples/` directory contains ready-to-run demos:

| Example | Description |
|:---|:---|
| [`demo.py`](examples/demo.py) | Rotating cube, sphere, and ground with orbit controls |
| [`earth.py`](examples/earth.py) | Textured Earth with normal maps, night lights, and clouds |
| [`bloom_demo.py`](examples/bloom_demo.py) | HDR bloom with glTF model |
| [`shadows.py`](examples/shadows.py) | Directional light shadow mapping |
| [`sponza.py`](examples/sponza.py) | Classic Sponza Atrium scene |
| [`gltf_viewer.py`](examples/gltf_viewer.py) | General-purpose glTF/GLB viewer |
| [`glfw_demo.py`](examples/glfw_demo.py) | Integration with **glfw** |
| [`pyside_demo.py`](examples/pyside_demo.py) | Integration with **PySide6** (Qt) |
| [`rendercanvas_demo.py`](examples/rendercanvas_demo.py) | Integration with **rendercanvas** |

```bash
python examples/demo.py
python examples/gltf_viewer.py path/to/model.glb
```

## 📖 Documentation & Support

| Document | Description |
|:---|:---|
| [`docs/API.md`](docs/API.md) | Full API reference (English) |
| [`docs/API_CN.md`](docs/API_CN.md) | Full API reference (中文) |
| [`docs/UserGuide.md`](docs/UserGuide.md) | User guide & tutorials (English) |
| [`docs/UserGuide_CN.md`](docs/UserGuide_CN.md) | User guide & tutorials (中文) |
| [Myth Engine](https://github.com/panxinmiao/myth) | Core engine repository (Rust + WGPU) |

> For the full API reference, see [`docs/API.md`](docs/API.md) or the [type stubs](myth/__init__.pyi).