# Myth Engine Python — API Reference

> **Myth Engine** is a high-performance 3D rendering engine built on wgpu. This document is the complete API reference for its Python bindings.

---

## Table of Contents

- [Type Aliases](#type-aliases)
- [Enums](#enums)
  - [RenderPath](#renderpath)
- [Application Layer](#application-layer)
  - [App](#app)
  - [Renderer](#renderer)
- [Engine Context](#engine-context)
  - [Engine](#engine)
  - [FrameState](#framestate)
- [Scene](#scene)
  - [Scene](#scene-1)
- [Scene Nodes](#scene-nodes)
  - [Object3D](#object3d)
- [Geometry](#geometry)
  - [BoxGeometry](#boxgeometry)
  - [SphereGeometry](#spheregeometry)
  - [PlaneGeometry](#planegeometry)
  - [Geometry](#geometry-1)
- [Materials](#materials)
  - [UnlitMaterial](#unlitmaterial)
  - [PhongMaterial](#phongmaterial)
  - [PhysicalMaterial](#physicalmaterial)
- [Cameras](#cameras)
  - [PerspectiveCamera](#perspectivecamera)
  - [OrthographicCamera](#orthographiccamera)
- [Lights](#lights)
  - [DirectionalLight](#directionallight)
  - [PointLight](#pointlight)
  - [SpotLight](#spotlight)
- [Component Proxies](#component-proxies)
  - [PerspectiveCameraComponent](#perspectivecameracomponent)
  - [OrthographicCameraComponent](#orthographiccameracomponent)
  - [DirectionalLightComponent](#directionallightcomponent)
  - [PointLightComponent](#pointlightcomponent)
  - [SpotLightComponent](#spotlightcomponent)
  - [MeshComponent](#meshcomponent)
- [Textures](#textures)
  - [TextureHandle](#texturehandle)
- [Controls](#controls)
  - [OrbitControls](#orbitcontrols)
- [Input](#input)
  - [Input](#input-1)
- [Animation](#animation)
  - [AnimationMixer](#animationmixer)

---

## Type Aliases

| Alias | Definition | Description |
|-------|-----------|-------------|
| `Color` | `str \| list[float] \| tuple[float, float, float]` | Color value: hex string `'#RRGGBB'`, `[r, g, b]` list, or `(r, g, b)` tuple |
| `Vec3` | `list[float] \| tuple[float, float, float]` | 3D vector: `[x, y, z]` |

---

## Enums

### RenderPath

Render pipeline path.

| Value | Description |
|-------|-------------|
| `RenderPath.BASIC` | Forward LDR + MSAA |
| `RenderPath.HIGH_FIDELITY` | HDR + post-processing (Bloom, SSAO, tone mapping, etc.) |

Legacy strings `'basic'`, `'hdr'`, `'high_fidelity'` are also accepted.

---

## Application Layer

### App

The main application class. Manages window creation, event loop, and render pipeline.

```python
app = myth.App(
    title: str = "Myth Engine",
    render_path: str | RenderPath = RenderPath.BASIC,
    vsync: bool = True,
    clear_color: list[float] = [0.0, 0.0, 0.0, 1.0],
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `title` | `str` | Window title |
| `render_path` | `str \| RenderPath` | Render path |
| `vsync` | `bool` | Whether vertical sync is enabled |
| `clear_color` | `list[float]` | Clear color `[r, g, b, a]` |

#### Methods

##### `app.init(func) → Callable`

Register an initialization callback. Typically used as a decorator.

```python
@app.init
def on_init(ctx: myth.Engine):
    scene = ctx.create_scene()
    # ... build scene
```

- **Parameters**: `func` — callback with signature `(Engine) -> None`
- **Returns**: The original function (pass-through decorator)

##### `app.update(func) → Callable`

Register a per-frame update callback.

```python
@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    cube.rotate_y(frame.dt * 0.5)
```

- **Parameters**: `func` — callback with signature `(Engine, FrameState) -> None`
- **Returns**: The original function

##### `app.run() → None`

Start the application (blocking). Enters the main event loop until the window is closed.

---

### Renderer

A low-level, GUI-agnostic renderer. Use this instead of `App` when you need to embed Myth into an external windowing system (GLFW, PySide6, wxPython, SDL2, etc.).

```python
renderer = myth.Renderer(
    render_path: str | RenderPath = RenderPath.BASIC,
    vsync: bool = True,
    clear_color: list[float] = [0.0, 0.0, 0.0, 1.0],
)
```

Supports the context manager protocol:

```python
with myth.Renderer(render_path="hdr") as renderer:
    renderer.init_with_handle(hwnd, 1280, 720)
    # ...
# dispose() called automatically
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `render_path` | `str \| RenderPath` | Render path |
| `vsync` | `bool` | Whether vertical sync is enabled |
| `time` | `float` *(read-only)* | Total time since start (seconds) |
| `frame_count` | `int` *(read-only)* | Total frames rendered |
| `input` | `Input` *(read-only)* | Input state proxy |

#### Initialization & Lifecycle

##### `renderer.init_with_handle(window_handle, width, height) → None`

Initialize the GPU with a native platform window handle.

- **Parameters**:
  - `window_handle: int` — Platform-specific window handle
    - **Windows**: HWND (`glfw.get_win32_window()` or `int(widget.winId())`)
    - **macOS**: NSView pointer
    - **Linux/X11**: X11 Window ID
  - `width: int` — Initial framebuffer width in pixels
  - `height: int` — Initial framebuffer height in pixels

##### `renderer.resize(width, height, scale_factor=1.0) → None`

Notify the renderer that the window has been resized.

##### `renderer.dispose() → None`

Release all GPU resources.

#### Render Control

##### `renderer.update(dt=None) → None`

Advance engine state. If `dt` is `None`, auto-calculates from wall clock time.

##### `renderer.render() → None`

Render one frame and present to the surface.

##### `renderer.frame(dt=None) → None`

Convenience: `update()` + `render()` in one call.

#### Scene / Asset Management

Same scene API as the `Engine` class:

##### `renderer.create_scene() → Scene`

Create a new scene and set it as the active scene.

##### `renderer.active_scene() → Scene | None`

Get the currently active scene.

##### `renderer.load_texture(path, color_space="srgb", generate_mipmaps=True) → TextureHandle`

Load a 2D texture.

##### `renderer.load_hdr_texture(path) → TextureHandle`

Load an HDR environment texture (`.hdr` files).

##### `renderer.load_gltf(path) → Object3D`

Load a glTF/GLB model, returns the root node.

#### Input Injection

When using an external windowing system, forward input events to the renderer:

| Method | Description |
|--------|-------------|
| `inject_key_down(key: str)` | Inject a key-down event |
| `inject_key_up(key: str)` | Inject a key-up event |
| `inject_mouse_move(x: float, y: float)` | Inject a mouse move event |
| `inject_mouse_down(button: int)` | Inject mouse press (0=left, 1=middle, 2=right) |
| `inject_mouse_up(button: int)` | Inject mouse release |
| `inject_scroll(dx: float, dy: float)` | Inject scroll event |

---

## Engine Context

### Engine

Engine context, available inside `@app.init` and `@app.update` callbacks.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `time` | `float` *(read-only)* | Total time since start (seconds) |
| `frame_count` | `int` *(read-only)* | Total frames rendered |
| `input` | `Input` *(read-only)* | Input state proxy |

#### Methods

##### `engine.create_scene() → Scene`

Create a new scene and set it as the active scene.

##### `engine.active_scene() → Scene | None`

Get the currently active scene.

##### `engine.load_texture(path, color_space="srgb", generate_mipmaps=True) → TextureHandle`

Load a 2D texture from a file path.

- **Parameters**:
  - `path: str` — Path to the texture image file
  - `color_space: str` — `'srgb'` or `'linear'`
  - `generate_mipmaps: bool` — Whether to generate mipmaps

##### `engine.load_hdr_texture(path) → TextureHandle`

Load an HDR environment texture (e.g. `.hdr` files).

##### `engine.load_gltf(path) → Object3D`

Load a glTF/GLB model and instantiate it in the active scene. Returns the root `Object3D` node.

##### `engine.set_title(title) → None`

Set the window title (only works when using `App`).

---

### FrameState

Per-frame state information, passed to the `@app.update` callback.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `delta_time` | `float` *(read-only)* | Time elapsed since last frame (seconds) |
| `dt` | `float` *(read-only)* | Alias for `delta_time` |
| `elapsed` | `float` *(read-only)* | Total time since application start (seconds) |
| `time` | `float` *(read-only)* | Alias for `elapsed` |
| `frame_count` | `int` *(read-only)* | Total frame count |

---

## Scene

### Scene

A scene container that holds objects, lights, cameras, and environment settings. Obtained via `engine.create_scene()`.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `active_camera` | `Object3D \| None` | The currently active camera node |

#### Adding Objects

##### `scene.add_mesh(geometry, material) → Object3D`

Add a mesh to the scene.

- **Parameters**:
  - `geometry` — A geometry object (`BoxGeometry`, `SphereGeometry`, `PlaneGeometry`, or `Geometry`)
  - `material` — A material object (`UnlitMaterial`, `PhongMaterial`, or `PhysicalMaterial`)
- **Returns**: A new `Object3D` mesh node

##### `scene.add_camera(camera) → Object3D`

Add a camera to the scene.

- **Parameters**: `camera` — `PerspectiveCamera` or `OrthographicCamera`
- **Returns**: An `Object3D` node. Set `scene.active_camera = node` to render from this camera.

##### `scene.add_light(light) → Object3D`

Add a light to the scene.

- **Parameters**: `light` — `DirectionalLight`, `PointLight`, or `SpotLight`
- **Returns**: An `Object3D` node (position the light via `.position`)

#### Scene Hierarchy

##### `scene.attach(child, parent) → None`

Attach a child node to a parent node.

##### `scene.find_node_by_name(name) → Object3D | None`

Find a node by name. Returns `None` if not found.

#### Background & Environment

##### `scene.set_background_color(r, g, b) → None`

Set the background to a solid color (components in 0.0–1.0).

```python
scene.set_background_color(0.05, 0.05, 0.1)
```

##### `scene.set_environment_map(tex) → None`

Set the environment map for image-based lighting (IBL) and skybox.

```python
hdr = ctx.load_hdr_texture("env.hdr")
scene.set_environment_map(hdr)
```

##### `scene.set_environment_intensity(intensity) → None`

Set the environment lighting intensity.

##### `scene.set_ambient_light(r, g, b) → None`

Set the ambient light color.

#### Post-Processing

> **Note**: Post-processing effects require the `RenderPath.HIGH_FIDELITY` (or `'hdr'`) render path.

##### Tone Mapping

```python
scene.set_tone_mapping_mode(mode: str) → None
scene.set_tone_mapping(mode: str, exposure: float | None = None) → None
```

Supported modes: `'linear'`, `'neutral'`, `'reinhard'`, `'cineon'`, `'aces'` / `'aces_filmic'`, `'agx'`

```python
scene.set_tone_mapping("aces", exposure=1.2)
```

##### Bloom

```python
scene.set_bloom_enabled(enabled: bool) → None
scene.set_bloom_strength(strength: float) → None   # e.g. 0.04
scene.set_bloom_radius(radius: float) → None       # e.g. 0.005

# Convenience method
scene.set_bloom(enabled: bool, strength: float | None = None, radius: float | None = None) → None
```

```python
scene.set_bloom(True, strength=0.04, radius=0.3)
```

##### SSAO (Screen-Space Ambient Occlusion)

```python
scene.set_ssao_enabled(enabled: bool) → None
scene.set_ssao_radius(radius: float) → None
scene.set_ssao_bias(bias: float) → None
scene.set_ssao_intensity(intensity: float) → None
```

#### Animation

##### `scene.play_animation(node, name) → None`

Play a named animation clip on a node.

##### `scene.play_if_any_animation(node) → None`

Play any available animation on a node (convenience method).

##### `scene.play_any_animation(node) → None`

Alias for `play_if_any_animation`.

##### `scene.list_animations(node) → list[str]`

List all animation clip names on a node.

##### `scene.get_animation_mixer(node) → AnimationMixer | None`

Get the animation mixer for a node (for advanced control).

---

## Scene Nodes

### Object3D

A 3D object (node) in the scene. Provides transform, visibility, shadow, and naming controls.
Offers component proxy accessors (`.camera`, `.light`, `.mesh`) for runtime inspection and
modification of the attached ECS component.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `position` | `list[float]` | Position `[x, y, z]` |
| `rotation` | `list[float]` | Euler rotation in radians `[x, y, z]` (XYZ order) |
| `rotation_euler` | `list[float]` | Euler rotation in degrees `[x, y, z]` (XYZ order) |
| `scale` | `list[float]` | Scale `[x, y, z]` |
| `visible` | `bool` | Whether this object is visible |
| `cast_shadows` | `bool` | Whether this mesh casts shadows (mesh nodes only) |
| `receive_shadows` | `bool` | Whether this mesh receives shadows (mesh nodes only) |
| `name` | `str \| None` | Node name |
| `camera` | `PerspectiveCameraComponent \| OrthographicCameraComponent \| None` | Camera component proxy (read-only getter) |
| `light` | `DirectionalLightComponent \| PointLightComponent \| SpotLightComponent \| None` | Light component proxy (read-only getter) |
| `mesh` | `MeshComponent \| None` | Mesh component proxy (read-only getter) |

#### Methods

##### `obj.set_uniform_scale(s) → None`

Set uniform scale (same value for x, y, z).

```python
model.set_uniform_scale(2.0)  # Scale up 2×
```

##### `obj.rotate_x(angle) → None`

Rotate around the local X axis by `angle` radians.

##### `obj.rotate_y(angle) → None`

Rotate around the local Y axis by `angle` radians.

##### `obj.rotate_z(angle) → None`

Rotate around the local Z axis by `angle` radians.

##### `obj.look_at(target) → None`

Rotate this node to face a world-space target position.

- **Parameters**: `target: Vec3` — Target position `[x, y, z]`

```python
cam.look_at([0, 0, 0])   # Look at origin
sun.look_at([0, 0, 0])   # Orient light toward origin
```

---

## Geometry

### BoxGeometry

A box (cuboid) geometry.

```python
myth.BoxGeometry(
    width: float = 1.0,    # Width along X axis
    height: float = 1.0,   # Height along Y axis
    depth: float = 1.0,    # Depth along Z axis
)
```

| Property | Type | Description |
|----------|------|-------------|
| `width` | `float` | Width along X axis |
| `height` | `float` | Height along Y axis |
| `depth` | `float` | Depth along Z axis |

---

### SphereGeometry

A sphere geometry.

```python
myth.SphereGeometry(
    radius: float = 1.0,
    width_segments: int = 32,
    height_segments: int = 16,
)
```

| Property | Type | Description |
|----------|------|-------------|
| `radius` | `float` | Sphere radius |
| `width_segments` | `int` | Horizontal segments |
| `height_segments` | `int` | Vertical segments |

---

### PlaneGeometry

A plane geometry.

```python
myth.PlaneGeometry(
    width: float = 1.0,    # Width along X axis
    height: float = 1.0,   # Height along Z axis
)
```

| Property | Type | Description |
|----------|------|-------------|
| `width` | `float` | Width along X axis |
| `height` | `float` | Height along Z axis |

---

### Geometry

A custom geometry built from raw vertex data.

```python
geo = myth.Geometry()
```

#### Methods

##### `geo.set_positions(data) → None`

Set vertex positions as a flat list `[x0, y0, z0, x1, y1, z1, ...]`.

##### `geo.set_normals(data) → None`

Set vertex normals as a flat list `[nx0, ny0, nz0, ...]`.

##### `geo.set_uvs(data) → None`

Set UV coordinates as a flat list `[u0, v0, u1, v1, ...]`.

##### `geo.set_indices(data) → None`

Set the triangle index buffer.

**Example**:

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

## Materials

### UnlitMaterial

An unlit material with flat color.

```python
myth.UnlitMaterial(
    color: Color = "#ffffff",
    opacity: float = 1.0,
    side: str = "front",        # "front" | "back" | "double"
)
```

| Property | Type | Description |
|----------|------|-------------|
| `color` | `list[float]` | Diffuse color `[r, g, b]`. Can be set with `[r, g, b]`, `(r, g, b)`, or hex string |
| `opacity` | `float` | Opacity (0.0–1.0) |

#### Methods

##### `mat.set_map(tex) → None`

Set the color (diffuse) texture map.

---

### PhongMaterial

A Blinn-Phong material with specular highlights.

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

| Property | Type | Description |
|----------|------|-------------|
| `color` | `list[float]` | Diffuse color |
| `shininess` | `float` | Specular exponent |
| `opacity` | `float` | Opacity |

#### Methods

| Method | Description |
|--------|-------------|
| `set_map(tex)` | Set the diffuse texture map |
| `set_normal_map(tex, scale=None)` | Set the normal map (optional scale, default 1.0) |

---

### PhysicalMaterial

A PBR metallic-roughness material.

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

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `color` | `list[float]` | Base color `[r, g, b]` |
| `metalness` | `float` | Metalness factor (0.0–1.0) |
| `roughness` | `float` | Roughness factor (0.0–1.0) |
| `emissive_intensity` | `float` | Emissive intensity multiplier |
| `opacity` | `float` | Opacity (0.0–1.0) |
| `clearcoat` | `float` | Clearcoat strength |
| `clearcoat_roughness` | `float` | Clearcoat roughness |
| `transmission` | `float` | Transmission (for glass-like materials) |
| `ior` | `float` | Index of refraction |

#### Methods

| Method | Description |
|--------|-------------|
| `set_map(tex)` | Set the base color texture map |
| `set_normal_map(tex, scale=None)` | Set the normal map |
| `set_roughness_map(tex)` | Set the roughness texture map |
| `set_metalness_map(tex)` | Set the metalness texture map |
| `set_emissive_map(tex)` | Set the emissive texture map |
| `set_ao_map(tex)` | Set the ambient occlusion texture map |

---

## Cameras

### PerspectiveCamera

A perspective projection camera.

```python
myth.PerspectiveCamera(
    fov: float = 60.0,            # Vertical field of view in degrees
    near: float = 0.1,            # Near clipping plane
    far: float = 1000.0,          # Far clipping plane
    aspect: float = 0.0,          # Aspect ratio (0 = auto-detect)
    position: list[float] = ...,  # Initial position [x, y, z]
)
```

| Property | Type | Description |
|----------|------|-------------|
| `fov` | `float` | Vertical field of view (degrees) |
| `aspect` | `float` | Aspect ratio (0 = auto) |
| `near` | `float` | Near clipping plane |
| `far` | `float` | Far clipping plane |
| `position` | `list[float]` | Initial position |

---

### OrthographicCamera

An orthographic projection camera.

```python
myth.OrthographicCamera(
    size: float = 10.0,           # Orthographic view height
    near: float = 0.1,
    far: float = 1000.0,
    position: list[float] = ...,
)
```

| Property | Type | Description |
|----------|------|-------------|
| `size` | `float` | Orthographic view height |
| `near` | `float` | Near clipping plane |
| `far` | `float` | Far clipping plane |
| `position` | `list[float]` | Initial position |

---

## Lights

### DirectionalLight

A directional light (simulates sunlight). Light direction is determined by the node's orientation.

```python
myth.DirectionalLight(
    color: list[float] = [1.0, 1.0, 1.0],
    intensity: float = 1.0,
    cast_shadows: bool = False,
)
```

| Property | Type | Description |
|----------|------|-------------|
| `color` | `list[float]` | Light color `[r, g, b]` |
| `intensity` | `float` | Light intensity multiplier |
| `cast_shadows` | `bool` | Whether this light casts shadows |

```python
sun = scene.add_light(myth.DirectionalLight(intensity=3.0, cast_shadows=True))
sun.position = [5, 10, 5]
sun.look_at([0, 0, 0])
```

---

### PointLight

A point light that emits in all directions.

```python
myth.PointLight(
    color: list[float] = [1.0, 1.0, 1.0],
    intensity: float = 1.0,
    range: float = 10.0,       # Maximum range (0 = infinite)
    cast_shadows: bool = False,
)
```

| Property | Type | Description |
|----------|------|-------------|
| `color` | `list[float]` | Light color |
| `intensity` | `float` | Light intensity |
| `range` | `float` | Maximum range |
| `cast_shadows` | `bool` | Whether this light casts shadows |

---

### SpotLight

A spotlight that emits in a cone shape.

```python
myth.SpotLight(
    color: list[float] = [1.0, 1.0, 1.0],
    intensity: float = 1.0,
    range: float = 10.0,
    inner_cone: float = 0.3,    # Inner cone angle (radians)
    outer_cone: float = 0.5,    # Outer cone angle (radians)
    cast_shadows: bool = False,
)
```

| Property | Type | Description |
|----------|------|-------------|
| `color` | `list[float]` | Light color |
| `intensity` | `float` | Light intensity |
| `range` | `float` | Maximum range |
| `inner_cone` | `float` | Inner cone angle (radians) |
| `outer_cone` | `float` | Outer cone angle (radians) |
| `cast_shadows` | `bool` | Whether this light casts shadows |

---

## Component Proxies

Component proxies are lightweight handles returned by the `Object3D.camera`, `Object3D.light`, and `Object3D.mesh` properties. They give direct read/write access to the live ECS component attached to a scene node. The concrete type returned depends on the component variant.

### PerspectiveCameraComponent

Returned by `node.camera` when the node carries a perspective camera.

| Property | Type | Description |
|----------|------|-------------|
| `fov` | `float` | Vertical field of view (degrees) |
| `aspect` | `float` | Aspect ratio |
| `near` | `float` | Near clipping plane |
| `far` | `float` | Far clipping plane |
| `antialiasing` | `AntiAliasing` | Anti-aliasing configuration |

### OrthographicCameraComponent

Returned by `node.camera` when the node carries an orthographic camera.

| Property | Type | Description |
|----------|------|-------------|
| `size` | `float` | Orthographic view half-height |
| `near` | `float` | Near clipping plane |
| `far` | `float` | Far clipping plane |
| `antialiasing` | `AntiAliasing` | Anti-aliasing configuration |

### DirectionalLightComponent

Returned by `node.light` when the node carries a directional light.

| Property | Type | Description |
|----------|------|-------------|
| `color` | `list[float]` | Light color `[r, g, b]` |
| `intensity` | `float` | Light intensity (lux) |
| `cast_shadows` | `bool` | Whether this light casts shadows |

### PointLightComponent

Returned by `node.light` when the node carries a point light.

| Property | Type | Description |
|----------|------|-------------|
| `color` | `list[float]` | Light color `[r, g, b]` |
| `intensity` | `float` | Light intensity (candela) |
| `range` | `float` | Maximum effective range |
| `cast_shadows` | `bool` | Whether this light casts shadows |

### SpotLightComponent

Returned by `node.light` when the node carries a spot light.

| Property | Type | Description |
|----------|------|-------------|
| `color` | `list[float]` | Light color `[r, g, b]` |
| `intensity` | `float` | Light intensity (candela) |
| `range` | `float` | Maximum range |
| `inner_cone` | `float` | Inner cone angle (radians) |
| `outer_cone` | `float` | Outer cone angle (radians) |
| `cast_shadows` | `bool` | Whether this light casts shadows |

### MeshComponent

Returned by `node.mesh` when the node carries a mesh.

| Property | Type | Description |
|----------|------|-------------|
| `visible` | `bool` | Mesh visibility |
| `cast_shadows` | `bool` | Whether this mesh casts shadows |
| `receive_shadows` | `bool` | Whether this mesh receives shadows |
| `render_order` | `int` | Draw order override |

**Example — runtime component modification:**

```python
# Adjust camera anti-aliasing at runtime
cam_node.camera.antialiasing = myth.AntiAliasing.taa(feedback_weight=0.9)

# Dynamically change light intensity
sun.light.intensity = 5.0

# Modify mesh shadow flags
cube.mesh.cast_shadows = False
```

---

## Textures

### TextureHandle

An opaque handle to a loaded texture.

Obtain via `engine.load_texture()` or `engine.load_hdr_texture()`, and pass to material methods like `set_map()`.

```python
tex = ctx.load_texture("diffuse.png", color_space="srgb", generate_mipmaps=True)
mat.set_map(tex)

hdr = ctx.load_hdr_texture("env.hdr")
scene.set_environment_map(hdr)
```

---

## Controls

### OrbitControls

Three.js-style orbit camera controls. Left-click rotates, right-click / Shift+left-click pans, scroll wheel zooms.

```python
myth.OrbitControls(
    position: list[float] = [0.0, 0.0, 5.0],   # Initial camera position
    target: list[float] = [0.0, 0.0, 0.0],      # Orbit target point
)
```

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `enable_damping` | `bool` | — | Enable damping (inertia) |
| `damping_factor` | `float` | — | Damping factor |
| `rotate_speed` | `float` | — | Rotation speed |
| `zoom_speed` | `float` | — | Zoom speed |
| `pan_speed` | `float` | — | Pan speed |
| `min_distance` | `float` | — | Minimum distance |
| `max_distance` | `float` | — | Maximum distance |

#### Methods

##### `orbit.update(camera, dt) → None`

Update the orbit controls. Must be called **every frame** in `@app.update`.

- **Parameters**:
  - `camera: Object3D` — The camera node
  - `dt: float` — Delta time in seconds (from `frame.delta_time`)

##### `orbit.set_target(target) → None`

Set the orbit target point `[x, y, z]`.

**Example**:

```python
orbit = myth.OrbitControls(position=[3, 3, 5], target=[0, 0.5, 0])

@app.update
def on_update(ctx, frame):
    orbit.update(cam, frame.dt)
```

---

## Input

### Input

Read-only input state proxy. Access via `engine.input` inside `@app.update` callbacks.

#### Keyboard Methods

| Method | Description |
|--------|-------------|
| `key(name: str) → bool` | Whether the key is currently held down |
| `key_down(name: str) → bool` | Whether the key was first pressed this frame |
| `key_up(name: str) → bool` | Whether the key was released this frame |

**Key names**: `'a'`–`'z'`, `'0'`–`'9'`, `'Space'`, `'Enter'`, `'Escape'`, `'Tab'`, `'Shift'`, `'Ctrl'`, `'Alt'`, `'ArrowUp'`, `'ArrowDown'`, `'ArrowLeft'`, `'ArrowRight'`, `'F1'`–`'F12'`

#### Mouse Methods

| Method | Description |
|--------|-------------|
| `mouse_button(name: str) → bool` | Whether the mouse button is currently held |
| `mouse_button_down(name: str) → bool` | Whether the mouse button was first pressed this frame |
| `mouse_button_up(name: str) → bool` | Whether the mouse button was released this frame |
| `mouse_position() → list[float]` | Current mouse position `[x, y]` in window pixels |
| `mouse_delta() → list[float]` | Mouse movement delta since last frame `[dx, dy]` |
| `scroll_delta() → list[float]` | Scroll wheel delta since last frame `[dx, dy]` |

**Mouse button names**: `'Left'`, `'Right'`, `'Middle'`

**Example**:

```python
@app.update
def on_update(ctx, frame):
    inp = ctx.input

    if inp.key("w"):
        # W key held — move forward
        pass

    if inp.key_down("Space"):
        # Space key just pressed
        pass

    if inp.mouse_button("Left"):
        dx, dy = inp.mouse_delta()
        # Left mouse button drag
```

---

## Animation

### AnimationMixer

Animation mixer attached to a specific node for advanced animation control. Obtained via `scene.get_animation_mixer(node)`.

#### Methods

| Method | Description |
|--------|-------------|
| `list_animations() → list[str]` | List all available animation clip names |
| `play(name: str)` | Play an animation by name |
| `stop(name: str)` | Stop a specific animation by name |
| `stop_all()` | Stop all animations |

**Example**:

```python
model = ctx.load_gltf("character.glb")
mixer = scene.get_animation_mixer(model)

if mixer:
    anims = mixer.list_animations()
    print(f"Available animations: {anims}")

    mixer.play("Walk")
    # Switch animation later
    mixer.stop("Walk")
    mixer.play("Run")
```
