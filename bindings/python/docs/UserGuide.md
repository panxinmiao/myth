# Myth Engine Python — User Guide

> **Myth Engine** is a high-performance 3D rendering engine built on wgpu, offering a Three.js-style object-oriented API.  
> This guide walks you through building 3D scenes with Python from the ground up.

---

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [One-Command Install](#one-command-install)
  - [Manual Install](#manual-install)
- [Quick Start](#quick-start)
  - [Your First Scene](#your-first-scene)
  - [Core Concepts](#core-concepts)
- [Application Architecture](#application-architecture)
  - [App Mode — Built-in Window](#app-mode--built-in-window)
  - [Renderer Mode — External Window](#renderer-mode--external-window)
- [Building Scenes](#building-scenes)
  - [Geometry](#geometry)
  - [Material System](#material-system)
  - [Creating Meshes](#creating-meshes)
  - [Scene Hierarchy](#scene-hierarchy)
- [Cameras](#cameras)
  - [Perspective Camera](#perspective-camera)
  - [Orthographic Camera](#orthographic-camera)
  - [Orbit Controls](#orbit-controls)
- [Lights & Shadows](#lights--shadows)
  - [Light Types](#light-types)
  - [Shadows](#shadows)
- [Component Proxies](#component-proxies)
  - [Camera Component](#camera-component)
  - [Light Component](#light-component)
  - [Mesh Component](#mesh-component)
- [Textures](#textures)
  - [Loading Textures](#loading-textures)
  - [Applying Textures](#applying-textures)
- [Environment & Background](#environment--background)
  - [Background Color](#background-color)
  - [Environment Maps & IBL](#environment-maps--ibl)
- [Post-Processing](#post-processing)
  - [Tone Mapping](#tone-mapping)
  - [Bloom](#bloom)
  - [FXAA Anti-Aliasing](#fxaa-anti-aliasing)
  - [SSAO](#ssao)
- [Animation](#animation)
  - [Playing Animations](#playing-animations)
  - [Advanced Animation Control](#advanced-animation-control)
- [Input Handling](#input-handling)
  - [Keyboard Input](#keyboard-input)
  - [Mouse Input](#mouse-input)
- [Loading glTF Models](#loading-gltf-models)
- [Custom Geometry](#custom-geometry)
- [External Window Integration](#external-window-integration)
  - [GLFW Integration](#glfw-integration)
  - [PySide6 (Qt) Integration](#pyside6-qt-integration)
  - [RenderCanvas Integration](#rendercanvas-integration)
- [Render Paths](#render-paths)
- [Best Practices](#best-practices)
- [FAQ](#faq)

---

## Installation

### Prerequisites

- **Rust** toolchain — [rustup.rs](https://rustup.rs)
- **Python 3.9+**
- **myth-engine** source at `../myth-engine` (sibling directory)

### One-Command Install

```bash
# Windows
scripts\install.bat

# Linux / macOS
./scripts/install.sh
```

This will create a virtualenv, install [maturin](https://www.maturin.rs), and build the library in release mode.

### Manual Install

```bash
# 1. Create & activate virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate

# 2. Install maturin
pip install maturin

# 3. Build & install (release)
maturin develop --release
```

### Build Options

```bash
# Release build (default, recommended)
scripts/build.bat release

# Debug build (faster compile, slower runtime)
scripts/build.bat debug

# Build distributable wheel
scripts/build.bat wheel

# Clean all build artifacts
scripts/build.bat clean
```

---

## Quick Start

### Your First Scene

Create a minimal 3D scene with a rotating cube:

```python
import myth

# Create the application
app = myth.App(title="Hello Myth", render_path="basic")

cube = None
cam = None

@app.init
def on_init(ctx):
    global cube, cam
    scene = ctx.create_scene()

    # Add a cube
    cube = scene.add_mesh(
        myth.BoxGeometry(1, 1, 1),
        myth.PhysicalMaterial(color="#ff8033", roughness=0.4),
    )
    cube.position = [0, 0.5, 0]

    # Add a camera
    cam = scene.add_camera(myth.PerspectiveCamera(fov=60))
    cam.position = [3, 3, 5]
    cam.look_at([0, 0, 0])
    scene.active_camera = cam

    # Add a light
    sun = scene.add_light(myth.DirectionalLight(intensity=2.0, cast_shadows=True))
    sun.position = [5, 10, 5]
    sun.look_at([0, 0, 0])

@app.update
def on_update(ctx, frame):
    cube.rotate_y(frame.dt * 0.5)

app.run()
```

Run it:

```bash
python your_script.py
```

### Core Concepts

The Myth Python API is organized around these core concepts:

```
App / Renderer          → Entry point, manages window and render loop
  └── Engine            → Engine context (used in callbacks)
        └── Scene       → Scene, holds all 3D objects
              ├── Object3D (Mesh)     → Geometry + Material
              ├── Object3D (Camera)   → Camera
              └── Object3D (Light)    → Light
```

**Workflow**:
1. Create an `App` (or `Renderer`)
2. Build the scene in the `@app.init` callback: add meshes, cameras, lights
3. Execute per-frame logic in the `@app.update` callback: animation, input handling
4. Call `app.run()` to start the main loop

---

## Application Architecture

Myth provides two usage modes:

### App Mode — Built-in Window

The `App` class automatically creates a window and manages the event loop. Ideal for standalone 3D applications:

```python
import myth

app = myth.App(
    title="My 3D App",
    render_path=myth.RenderPath.HIGH_FIDELITY,
    vsync=True,
)

@app.init
def on_init(ctx: myth.Engine):
    scene = ctx.create_scene()
    # ... build scene

@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    # ... per-frame update

app.run()  # Blocks until window is closed
```

### Renderer Mode — External Window

The `Renderer` class is a GUI-agnostic, low-level renderer. Use it when you need to embed Myth into an existing windowing framework (GLFW, Qt, wxPython, etc.):

```python
import myth

renderer = myth.Renderer(render_path=myth.RenderPath.BASIC)

# Initialize with the external window's native handle
renderer.init_with_handle(window_handle, width, height)

scene = renderer.create_scene()
# ... build scene

# Drive the render loop manually
while running:
    process_events()             # External window event handling
    renderer.frame(dt)           # Update + render one frame

renderer.dispose()               # Release GPU resources
```

> `Renderer` also supports the context manager protocol: `with myth.Renderer(...) as r:`

---

## Building Scenes

### Geometry

Myth provides three built-in geometries and one custom geometry type:

```python
# Box (cuboid)
box = myth.BoxGeometry(width=2, height=1, depth=1)

# Sphere
sphere = myth.SphereGeometry(radius=0.5, width_segments=32, height_segments=16)

# Plane
plane = myth.PlaneGeometry(width=10, height=10)

# Custom geometry
custom = myth.Geometry()
custom.set_positions([0, 0, 0, 1, 0, 0, 0, 1, 0])
custom.set_indices([0, 1, 2])
```

### Material System

Myth provides three materials, from simple to advanced:

#### UnlitMaterial — Unlit

The simplest material, unaffected by lighting. Good for UI elements, debug helpers, etc.

```python
mat = myth.UnlitMaterial(color="#ff0000", opacity=0.8)
```

#### PhongMaterial — Blinn-Phong Lighting

Classic lighting model with diffuse and specular highlights.

```python
mat = myth.PhongMaterial(
    color="#ffffff",
    specular="#aaaaaa",
    shininess=64.0,
)
```

#### PhysicalMaterial — PBR

Physically-based rendering using the metallic-roughness workflow. **Recommended for most scenes.**

```python
mat = myth.PhysicalMaterial(
    color="#ff8033",
    metalness=0.8,        # 0.0 = non-metal, 1.0 = metal
    roughness=0.2,        # 0.0 = smooth, 1.0 = rough
    emissive="#000000",
    emissive_intensity=1.0,
)
```

**Advanced properties**:

```python
mat.clearcoat = 1.0               # Clearcoat layer
mat.clearcoat_roughness = 0.1
mat.transmission = 0.9             # Transmission (glass effect)
mat.ior = 1.5                      # Index of refraction
```

#### Color Formats

All material color parameters accept multiple formats:

```python
# Hex string
mat = myth.PhysicalMaterial(color="#ff8033")

# RGB list (0.0–1.0)
mat = myth.PhysicalMaterial(color=[1.0, 0.5, 0.2])

# RGB tuple
mat = myth.PhysicalMaterial(color=(1.0, 0.5, 0.2))
```

#### Face Culling

Control face culling with the `side` parameter:

| Value | Description |
|-------|-------------|
| `"front"` | Render front faces only (default) |
| `"back"` | Render back faces only |
| `"double"` | Render both sides |

### Creating Meshes

Combine geometry and material into a mesh and add it to the scene:

```python
cube = scene.add_mesh(
    myth.BoxGeometry(1, 1, 1),
    myth.PhysicalMaterial(color="#ff8033", roughness=0.4, metalness=0.3),
)
cube.position = [0, 1, 0]
cube.rotation_euler = [0, 45, 0]    # Rotate 45° (degrees)
cube.scale = [1, 1, 1]
```

### Scene Hierarchy

Use `scene.attach()` to build parent-child relationships:

```python
parent = scene.add_mesh(box_geo, mat)
child = scene.add_mesh(sphere_geo, mat)
scene.attach(child, parent)

# Moving the parent also moves the child
parent.position = [2, 0, 0]
```

Find nodes by name:

```python
node = scene.find_node_by_name("MyNode")
if node:
    node.visible = False
```

---

## Cameras

### Perspective Camera

Simulates human eye / real camera perspective projection (objects appear smaller with distance).

```python
cam_node = scene.add_camera(myth.PerspectiveCamera(
    fov=60,       # Vertical field of view (degrees)
    near=0.1,     # Near clipping plane
    far=1000.0,   # Far clipping plane
    aspect=0.0,   # 0 = auto
))
cam_node.position = [0, 5, 10]
cam_node.look_at([0, 0, 0])
scene.active_camera = cam_node
```

### Orthographic Camera

No perspective effect. Suitable for 2D games, architectural drawings, UI, etc.

```python
cam_node = scene.add_camera(myth.OrthographicCamera(
    size=10.0,    # View height
    near=0.1,
    far=1000.0,
))
cam_node.position = [0, 10, 0]
cam_node.look_at([0, 0, 0])
scene.active_camera = cam_node
```

### Orbit Controls

`OrbitControls` provides Three.js-style mouse interaction:

| Action | Function |
|--------|----------|
| Left-click drag | Rotate |
| Right-click / Shift+left-click drag | Pan |
| Scroll wheel | Zoom |

```python
orbit = myth.OrbitControls(
    position=[3, 3, 5],
    target=[0, 0, 0],
)

# Optional: tune parameters
orbit.rotate_speed = 1.0
orbit.zoom_speed = 1.0
orbit.enable_damping = True
orbit.damping_factor = 0.05

@app.update
def on_update(ctx, frame):
    orbit.update(cam, frame.dt)    # Must be called every frame
```

---

## Lights & Shadows

### Light Types

#### Directional Light

Simulates sunlight — all rays are parallel. Direction is controlled via the node's `position` and `look_at()`.

```python
sun = scene.add_light(myth.DirectionalLight(
    color=[1.0, 0.95, 0.9],
    intensity=3.0,
    cast_shadows=True,
))
sun.position = [5, 10, 5]
sun.look_at([0, 0, 0])
```

#### Point Light

Emits light in all directions from a single point.

```python
bulb = scene.add_light(myth.PointLight(
    color=[1.0, 0.8, 0.6],
    intensity=5.0,
    range=20.0,
))
bulb.position = [0, 3, 0]
```

#### Spot Light

Emits light in a cone shape toward a specific direction.

```python
spot = scene.add_light(myth.SpotLight(
    intensity=10.0,
    range=30.0,
    inner_cone=0.2,      # Inner cone angle (radians)
    outer_cone=0.5,      # Outer cone angle (radians)
    cast_shadows=True,
))
spot.position = [0, 5, 0]
spot.look_at([0, 0, 0])
```

### Shadows

Enable `cast_shadows=True` on lights and control shadow behavior on meshes:

```python
# Light casts shadows
sun = scene.add_light(myth.DirectionalLight(intensity=3.0, cast_shadows=True))

# Mesh casts + receives shadows
cube.cast_shadows = True
cube.receive_shadows = True

# Ground only receives shadows
ground.cast_shadows = False
ground.receive_shadows = True
```

---

## Component Proxies

Every `Object3D` node can carry one or more **components** (camera, light, mesh). After adding a component to a node, you can access its runtime properties through the corresponding **component proxy**:

| Property | Returns | Available when… |
|:---|:---|:---|
| `node.camera` | `PerspectiveCameraComponent` / `OrthographicCameraComponent` | Node was created via `scene.add_camera()` |
| `node.light` | `DirectionalLightComponent` / `PointLightComponent` / `SpotLightComponent` | Node was created via `scene.add_light()` |
| `node.mesh` | `MeshComponent` | Node was created via `scene.add_mesh()` |

All properties return `None` if the component is absent.

### Camera Component

Access and modify camera parameters at runtime:

```python
cam_node = scene.add_camera(myth.PerspectiveCamera(fov=60))
cam_node.position = [0, 5, 10]
scene.active_camera = cam_node

# Later, at runtime:
cam = cam_node.camera  # PerspectiveCameraComponent
if cam:
    cam.fov = 45.0                                                # Change FOV
    cam.near = 0.01                                               # Near clip
    cam.antialiasing = myth.AntiAliasing.taa(feedback_weight=0.9)  # Enable TAA
```

For an orthographic camera:

```python
cam_node = scene.add_camera(myth.OrthographicCamera(size=10.0))
cam = cam_node.camera  # OrthographicCameraComponent
if cam:
    cam.size = 20.0   # Adjust view size
```

### Light Component

Inspect and modify light parameters without replacing the light:

```python
sun = scene.add_light(myth.DirectionalLight(intensity=3.0, cast_shadows=True))
sun.position = [5, 10, 5]

# At runtime:
light = sun.light  # DirectionalLightComponent
if light:
    light.intensity = 5.0
    light.color = [1.0, 0.8, 0.6]
```

Different light types expose different properties:

```python
# PointLight → PointLightComponent
bulb = scene.add_light(myth.PointLight(intensity=5.0, range=20.0))
bulb.light.range = 30.0

# SpotLight → SpotLightComponent
spot = scene.add_light(myth.SpotLight(inner_cone=0.2, outer_cone=0.5))
spot.light.inner_cone = 0.1
spot.light.outer_cone = 0.4
```

### Mesh Component

Control mesh rendering properties:

```python
cube = scene.add_mesh(
    myth.BoxGeometry(1, 1, 1),
    myth.PhysicalMaterial(color="#ff8033"),
)

mesh = cube.mesh  # MeshComponent
if mesh:
    mesh.visible = False         # Hide mesh without removing the node
    mesh.cast_shadows = True
    mesh.receive_shadows = True
    mesh.render_order = 10       # Control draw order
```

> **Tip**: Component proxies are lightweight handles — they read/write directly to the engine's ECS storage with no extra copies.

---

## Textures

### Loading Textures

```python
# Standard texture (sRGB color space, auto-generate mipmaps)
diffuse_tex = ctx.load_texture("textures/diffuse.png")

# Linear color space (for normal maps, roughness maps, etc.)
normal_tex = ctx.load_texture("textures/normal.png", color_space="linear")

# Without mipmaps
tex = ctx.load_texture("textures/ui.png", generate_mipmaps=False)

# HDR environment texture
hdr_tex = ctx.load_hdr_texture("textures/env.hdr")
```

### Applying Textures

Apply textures to different material channels:

```python
mat = myth.PhysicalMaterial(color="#ffffff", roughness=0.5, metalness=0.0)

mat.set_map(diffuse_tex)                       # Base color map
mat.set_normal_map(normal_tex, scale=1.0)      # Normal map
mat.set_roughness_map(roughness_tex)           # Roughness map
mat.set_metalness_map(metalness_tex)           # Metalness map
mat.set_emissive_map(emissive_tex)             # Emissive map
mat.set_ao_map(ao_tex)                         # Ambient occlusion map
```

---

## Environment & Background

### Background Color

```python
scene.set_background_color(0.05, 0.05, 0.1)  # Dark blue background
```

### Environment Maps & IBL

Using an HDR environment map provides both a skybox and image-based lighting (IBL):

```python
hdr = ctx.load_hdr_texture("environment.hdr")
scene.set_environment_map(hdr)
scene.set_environment_intensity(1.5)   # Adjust IBL intensity
```

You can also set a simple ambient light:

```python
scene.set_ambient_light(0.2, 0.2, 0.3)  # Light blue ambient
```

---

## Post-Processing

> Post-processing effects require the `RenderPath.HIGH_FIDELITY` (or `'hdr'` string) render path.

```python
app = myth.App(render_path=myth.RenderPath.HIGH_FIDELITY)
```

### Tone Mapping

Tone mapping converts HDR colors to the displayable LDR range.

```python
scene.set_tone_mapping("aces", exposure=1.2)
```

Available modes:

| Mode | Description |
|------|-------------|
| `'linear'` | Linear mapping (no tone mapping) |
| `'neutral'` | Khronos PBR neutral tone mapping |
| `'reinhard'` | Reinhard mapping |
| `'cineon'` | Cineon film-style |
| `'aces'` | ACES Filmic (recommended, cinematic look) |
| `'agx'` | AgX mapping |

### Bloom

Makes bright areas appear to glow:

```python
scene.set_bloom(True, strength=0.04, radius=0.3)

# Or step by step
scene.set_bloom_enabled(True)
scene.set_bloom_strength(0.04)
scene.set_bloom_radius(0.3)
```

### SSAO

Screen-Space Ambient Occlusion — simulates natural darkening in creases and corners:

```python
scene.set_ssao_enabled(True)
scene.set_ssao_radius(0.5)
scene.set_ssao_intensity(1.0)
scene.set_ssao_bias(0.025)
```

### Recommended Post-Processing Setup

```python
# High-quality photorealistic rendering
scene.set_tone_mapping("aces", exposure=1.0)
scene.set_bloom(True, strength=0.02)
scene.set_ssao_enabled(True)
```

---

## Animation

### Playing Animations

After loading a glTF model, play animations directly:

```python
model = ctx.load_gltf("character.glb")

# Play any available animation (shortcut)
scene.play_any_animation(model)

# Play a specific animation by name
scene.play_animation(model, "Walk")

# List all available animations
anims = scene.list_animations(model)
print(f"Available animations: {anims}")
```

### Advanced Animation Control

Use `AnimationMixer` for fine-grained control:

```python
mixer = scene.get_animation_mixer(model)

if mixer:
    anims = mixer.list_animations()
    mixer.play("Walk")

    # Switch animation
    mixer.stop("Walk")
    mixer.play("Run")

    # Stop all
    mixer.stop_all()
```

---

## Input Handling

### Keyboard Input

```python
@app.update
def on_update(ctx, frame):
    inp = ctx.input

    # Held down detection (continuous)
    if inp.key("w"):
        print("W key held")

    # First-press detection (single frame)
    if inp.key_down("Space"):
        print("Space pressed")

    # Release detection
    if inp.key_up("Escape"):
        print("Escape released")
```

### Mouse Input

```python
@app.update
def on_update(ctx, frame):
    inp = ctx.input

    # Mouse position
    x, y = inp.mouse_position()

    # Mouse movement delta
    dx, dy = inp.mouse_delta()

    # Scroll delta
    sx, sy = inp.scroll_delta()

    # Button state
    if inp.mouse_button("Left"):
        print(f"Left drag: dx={dx}, dy={dy}")

    if inp.mouse_button_down("Right"):
        print("Right click")
```

---

## Loading glTF Models

Myth supports loading glTF 2.0 (`.gltf` and `.glb`) models:

```python
import sys
import myth

app = myth.App(title="glTF Viewer", render_path=myth.RenderPath.HIGH_FIDELITY)

orbit = myth.OrbitControls()
model = None
cam = None

@app.init
def on_init(ctx):
    global model, cam
    scene = ctx.create_scene()

    # Load the model
    model = ctx.load_gltf("path/to/model.glb")

    # Camera
    cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.01))
    cam.position = [0, 1.5, 3]
    cam.look_at([0, 0, 0])
    scene.active_camera = cam

    # Lighting
    sun = scene.add_light(myth.DirectionalLight(intensity=3.0, cast_shadows=True))
    sun.position = [3, 5, 3]

    # Environment & post-processing
    scene.set_background_color(0.15, 0.15, 0.2)
    scene.set_tone_mapping("aces", exposure=1.0)
    scene.set_bloom(True, strength=0.02)

    # Play animations
    scene.play_any_animation(model)

    anims = scene.list_animations(model)
    if anims:
        print(f"Animations: {anims}")

@app.update
def on_update(ctx, frame):
    orbit.update(cam, frame.dt)

app.run()
```

---

## Custom Geometry

When built-in geometries don't meet your needs, build custom geometry from raw vertex data:

```python
geo = myth.Geometry()

# Three vertices of a triangle
geo.set_positions([
    -0.5, -0.5, 0.0,   # Vertex 0
     0.5, -0.5, 0.0,   # Vertex 1
     0.0,  0.5, 0.0,   # Vertex 2
])

# Normals (all facing +Z)
geo.set_normals([
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
])

# UV coordinates
geo.set_uvs([
    0.0, 0.0,
    1.0, 0.0,
    0.5, 1.0,
])

# Indices
geo.set_indices([0, 1, 2])

# Create mesh
mesh = scene.add_mesh(geo, myth.PhysicalMaterial(color="#00ff00"))
```

---

## External Window Integration

### GLFW Integration

Use `Renderer` to embed Myth in a GLFW window:

```python
import glfw
import myth

# GLFW initialization
glfw.init()
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # No OpenGL needed
window = glfw.create_window(1280, 720, "Myth + GLFW", None, None)

# Create the renderer
renderer = myth.Renderer(render_path=myth.RenderPath.BASIC, vsync=False)
hwnd = glfw.get_win32_window(window)  # Windows platform
renderer.init_with_handle(hwnd, 1280, 720)

# Build the scene
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

# Forward input events
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

# Render loop
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

### PySide6 (Qt) Integration

Embed Myth in a Qt application:

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

        # 60fps timer
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
        # ... add meshes, cameras, lights

    def _render(self):
        self._ensure_init()
        if self._initialized:
            self.renderer.frame()

    # Forward input events
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
        return None  # wgpu renders directly
```

### RenderCanvas Integration

Use rendercanvas as the windowing backend:

```python
import myth
from rendercanvas.glfw import GlfwRenderCanvas

canvas = GlfwRenderCanvas(title="Myth + RenderCanvas", size=(1280, 720))

renderer = myth.Renderer(render_path=myth.RenderPath.BASIC)
# ... initialization and scene setup

# Forward rendercanvas events to Myth
@canvas.add_event_handler("pointer_move")
def on_move(event):
    pr = event.get("pixel_ratio", 1.0)
    renderer.inject_mouse_move(event["x"] * pr, event["y"] * pr)

@canvas.add_event_handler("pointer_down")
def on_down(event):
    renderer.inject_mouse_down(0)

# Render loop
while not canvas.get_closed():
    canvas._process_events()
    renderer.frame(dt)

renderer.dispose()
```

---

## Render Paths

Myth provides two render paths:

### Basic

```python
app = myth.App(render_path=myth.RenderPath.BASIC)
```

- Forward rendering + MSAA
- LDR output
- Suitable for simple scenes, rapid prototyping

### High Fidelity

```python
app = myth.App(render_path=myth.RenderPath.HIGH_FIDELITY)
```

- HDR rendering pipeline
- Post-processing support: Bloom, SSAO, tone mapping, FXAA
- Suitable for photorealistic rendering, product visualization

> Legacy strings `'basic'`, `'hdr'`, `'high_fidelity'` are also accepted.

---

## Best Practices

### 1. Separate Initialization from Updates

Place scene construction in `@app.init` and per-frame logic in `@app.update`:

```python
@app.init
def on_init(ctx):
    # Create scene, add objects — runs once

@app.update
def on_update(ctx, frame):
    # Animation, input handling — runs every frame
```

### 2. Use Global Variables for Node References

`@app.init` and `@app.update` are separate callbacks — share `Object3D` references via globals:

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
    cube.rotate_y(frame.dt)        # Use global reference
    orbit.update(cam, frame.dt)
```

### 3. Use delta_time for Frame-Rate Independent Animation

Always use `frame.dt` (not fixed values) to drive animations, ensuring consistent behavior across different frame rates:

```python
@app.update
def on_update(ctx, frame):
    cube.rotate_y(frame.dt * speed)  # ✓ Frame-rate independent
    # cube.rotate_y(0.01)           # ✗ Frame-rate dependent
```

### 4. Choose the Right Render Path

- Simple scenes / rapid prototyping → `RenderPath.BASIC`
- High-quality rendering / post-processing needed → `RenderPath.HIGH_FIDELITY`

### 5. FPS Counter

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

## FAQ

### Q: Running examples gives "module not found"

Make sure you have built and installed in the virtual environment:

```bash
.venv\Scripts\activate.bat
maturin develop --release
```

### Q: Window appears but the screen is black

Check that you've set an active camera:

```python
scene.active_camera = cam_node
```

Also ensure the scene has lights (except when using `UnlitMaterial`).

### Q: Post-processing effects don't work

Make sure you're using the high-fidelity render path:

```python
app = myth.App(render_path=myth.RenderPath.HIGH_FIDELITY)
# or
app = myth.App(render_path="hdr")
```

### Q: Embedded Qt window doesn't update

Ensure:
1. `WA_PaintOnScreen` and `WA_NativeWindow` are set
2. `paintEngine()` returns `None`
3. A `QTimer` periodically calls `renderer.frame()`

### Q: Mouse interaction doesn't work in Renderer mode

`Renderer` mode requires manually injecting input events:

```python
renderer.inject_mouse_move(x, y)
renderer.inject_mouse_down(button)
renderer.inject_mouse_up(button)
renderer.inject_scroll(dx, dy)
renderer.inject_key_down(key)
renderer.inject_key_up(key)
```

### Q: How to lay a plane flat (horizontal)?

`PlaneGeometry` defaults to the XY plane. Rotate it -90° to make it horizontal:

```python
ground = scene.add_mesh(
    myth.PlaneGeometry(width=20, height=20),
    myth.PhysicalMaterial(color="#666666"),
)
ground.rotation_euler = [-90, 0, 0]
```

### Q: Must myth-engine source be in a sibling directory?

Yes, the build depends on `../myth-engine`. Your directory structure should be:

```
your-workspace/
├── myth-python/
└── myth-engine/
```
