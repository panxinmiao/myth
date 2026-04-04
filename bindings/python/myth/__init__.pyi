"""Type stubs for the ``myth`` Python module — bindings for Myth Engine.

Myth Engine is a high-performance 3D rendering engine built on wgpu.
These stubs provide IDE autocompletion and type checking support.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

# ============================================================================
# Type Aliases
# ============================================================================

ColorInput = Union[str, Sequence[float]]
"""Color specification: hex string ``'#RRGGBB'``, ``[r, g, b]`` list, or ``(r, g, b)`` tuple."""

Vec2Input = Sequence[float]
"""A 2-component vector: ``[x, y]``."""

Vec3Input = Sequence[float]
"""A 3-component vector: ``[x, y, z]``."""

# ============================================================================
# Enums
# ============================================================================

class RenderPath:
    """Render pipeline path.

    - ``RenderPath.BASIC``: Forward LDR + MSAA.
    - ``RenderPath.HIGH_FIDELITY``: HDR + post-processing (bloom, SSAO, tone mapping, …).

    Can also pass legacy strings ``'basic'``, ``'hdr'``, ``'high_fidelity'``.
    """

    BASIC: RenderPath
    """Forward LDR + MSAA."""
    HIGH_FIDELITY: RenderPath
    """HDR + post-processing (bloom, SSAO, tone mapping, etc.)."""

# ============================================================================
# App
# ============================================================================

class App:
    """The main myth application.

    Register ``@app.init`` and ``@app.update`` callbacks, then call ``app.run()``.

    Example::

        import myth

        app = myth.App(title="My App", render_path=myth.RenderPath.HIGH_FIDELITY)

        @app.init
        def init(ctx: myth.Engine) -> None:
            scene = ctx.create_scene()
            ...

        @app.update
        def update(ctx: myth.Engine, frame: myth.FrameState) -> None:
            ...

        app.run()
    """

    title: str
    render_path: Union[str, RenderPath]
    """Render path: ``RenderPath.BASIC`` (forward LDR+MSAA) or ``RenderPath.HIGH_FIDELITY`` (HDR+post-processing). Also accepts legacy strings."""
    vsync: bool
    clear_color: ColorInput
    """Clear color as ``[r, g, b, a]``."""

    def __init__(
        self,
        title: str = "Myth Engine",
        render_path: Union[str, RenderPath] = ...,
        vsync: bool = True,
        clear_color: ColorInput = ...,
    ) -> None: ...
    def init(self, func: Callable[[Engine], None]) -> Callable[[Engine], None]:
        """Register an init callback. Typically used as a decorator ``@app.init``."""
        ...

    def update(
        self, func: Callable[[Engine, FrameState], None]
    ) -> Callable[[Engine, FrameState], None]:
        """Register a per-frame update callback. Typically used as ``@app.update``."""
        ...

    def run(self) -> None:
        """Run the application (blocking). Starts the main event loop."""
        ...

# ============================================================================
# Renderer — GUI-agnostic engine (for glfw, PySide, wxPython, …)
# ============================================================================

class Renderer:
    """Low-level, GUI-agnostic renderer.

    Use this instead of :class:`App` when you want to drive the render loop
    from an external windowing library (glfw, PySide6, wxPython, SDL2, …).

    Example (glfw)::

        import glfw, myth

        renderer = myth.Renderer(render_path=myth.RenderPath.HIGH_FIDELITY)
        glfw.init()
        win = glfw.create_window(1280, 720, "Hello", None, None)
        renderer.init_with_handle(glfw.get_win32_window(win), 1280, 720)

        scene = renderer.create_scene()
        # … build scene …

        while not glfw.window_should_close(win):
            glfw.poll_events()
            renderer.frame()

        renderer.dispose()
    """

    render_path: Union[str, RenderPath]
    vsync: bool

    def __init__(
        self,
        render_path: Union[str, RenderPath] = ...,
        vsync: bool = True,
        clear_color: list[float] = ...,
    ) -> None: ...
    def init_with_handle(
        self,
        window_handle: int,
        width: int,
        height: int,
    ) -> None:
        """Initialize GPU with a native platform window handle.

        Args:
            window_handle: Platform-specific integer:
                - **Windows**: HWND (``glfw.get_win32_window()``, ``int(widget.winId())``)
                - **macOS**: NSView pointer
                - **Linux/X11**: X11 Window ID
            width: Initial framebuffer width in pixels.
            height: Initial framebuffer height in pixels.
        """
        ...

    def resize(self, width: int, height: int, scale_factor: float = 1.0) -> None:
        """Notify the renderer that the window has been resized."""
        ...

    def update(self, dt: Optional[float] = None) -> None:
        """Advance engine state. If *dt* is None, auto-calculates from wall time."""
        ...

    def render(self) -> None:
        """Render one frame and present to the surface."""
        ...

    def frame(self, dt: Optional[float] = None) -> None:
        """Convenience: ``update()`` + ``render()`` in one call."""
        ...

    # ---- Scene / Asset management (same API as Engine) ----

    def create_scene(self) -> Scene:
        """Create a new scene and set it as the active scene."""
        ...

    def active_scene(self) -> Optional[Scene]:
        """Get the currently active scene."""
        ...

    def load_texture(
        self, path: str, color_space: str = "srgb", generate_mipmaps: bool = True
    ) -> TextureHandle: ...
    def load_hdr_texture(self, path: str) -> TextureHandle: ...
    def load_gltf(self, path: str) -> Object3D: ...

    # ---- Input injection ----

    def inject_key_down(self, key: str) -> None:
        """Inject a key-down event (e.g. ``"w"``, ``"Space"``, ``"Escape"``)."""
        ...

    def inject_key_up(self, key: str) -> None: ...
    def inject_mouse_move(self, x: float, y: float) -> None: ...
    def inject_mouse_down(self, button: int) -> None:
        """Inject mouse press. 0=left, 1=middle, 2=right."""
        ...
    def inject_mouse_up(self, button: int) -> None: ...
    def inject_scroll(self, dx: float, dy: float) -> None: ...

    # ---- Timing ----

    @property
    def time(self) -> float: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def input(self) -> Input: ...

    # ---- Headless / Readback ----

    def init_headless(
        self,
        width: int,
        height: int,
        format: Optional[str] = None,
    ) -> None:
        """Initialize the renderer in headless (offscreen) mode.

        No window is created. An offscreen render target of the specified
        dimensions is allocated, suitable for server-side rendering and
        GPU readback.

        Args:
            width: Render target width in pixels.
            height: Render target height in pixels.
            format: Pixel format — ``'rgba8'`` (default) or ``'rgba16float'``
                / ``'rgba16'`` / ``'hdr'`` for HDR readback.
        """
        ...

    def readback_pixels(self) -> bytes:
        """Read back the current render target as raw bytes.

        Returns tightly-packed pixel data (RGBA8 = 4 bytes/px,
        RGBA16Float = 8 bytes/px). Row order is top-to-bottom.
        """
        ...

    # ---- Expert Readback API ----

    def create_readback_stream(self, buffer_count: int = 3) -> ReadbackStream:
        """Create a :class:`ReadbackStream` for non-blocking readback.

        Args:
            buffer_count: Number of ring-buffer slots (default 3).
        """
        ...

    def poll_device(self) -> None:
        """Drive pending GPU callbacks without blocking.

        Call once per frame in a readback-stream loop so that frames
        become available via :meth:`ReadbackStream.try_recv`.
        """
        ...

    # ---- Simple Recording API ----

    def start_recording(self, buffer_count: int = 3) -> None:
        """Begin a recording session with an internal ring-buffer stream.

        Enables :meth:`render_and_record` / :meth:`try_pull_frame` /
        :meth:`flush_recording`.

        Args:
            buffer_count: Number of ring-buffer slots (default 3).

        Raises:
            RuntimeError: If a recording session is already active.
        """
        ...

    def render_and_record(self, dt: Optional[float] = None) -> None:
        """Update, render, and record one frame — all in a single call.

        Equivalent to ``update(dt) → render() → submit → poll_device``.
        Pull completed frames with :meth:`try_pull_frame`.

        Args:
            dt: Delta time in seconds. ``None`` = auto from wall clock.

        Raises:
            RuntimeError: If no recording session is active.
        """
        ...

    def try_pull_frame(self) -> Optional[dict]:
        """Return the next completed frame, or ``None``.

        The returned dict has:
          - ``"pixels"``: ``bytes`` — tightly-packed pixel data.
          - ``"frame_index"``: ``int`` — zero-based index.

        Raises:
            RuntimeError: If no recording session is active.
        """
        ...

    def flush_recording(self) -> list[dict]:
        """Block until all in-flight frames are received, then end the session.

        Returns all remaining frames as a list of dicts.
        """
        ...

    # ---- Lifecycle ----

    def dispose(self) -> None:
        """Release all GPU resources."""
        ...

    def __enter__(self) -> Renderer: ...
    def __exit__(self, *args: object) -> bool: ...

# ============================================================================
# ReadbackStream — Expert-mode non-blocking GPU→CPU readback
# ============================================================================

class ReadbackStream:
    """High-throughput, non-blocking GPU→CPU readback stream.

    Created via :meth:`Renderer.create_readback_stream`. Use this for
    fine-grained control over readback timing. For a simpler API, see
    :meth:`Renderer.start_recording`.

    Example::

        stream = renderer.create_readback_stream(buffer_count=3)
        for _ in range(100):
            renderer.update()
            renderer.render()
            stream.submit(renderer)
            renderer.poll_device()
            frame = stream.try_recv()
            if frame is not None:
                process(frame["pixels"])
        for frame in stream.flush(renderer):
            process(frame["pixels"])
    """

    @property
    def buffer_count(self) -> int:
        """Number of ring-buffer slots."""
        ...

    @property
    def frames_submitted(self) -> int:
        """Total frames submitted so far."""
        ...

    @property
    def dimensions(self) -> tuple[int, int]:
        """Render target dimensions as ``(width, height)``."""
        ...

    def submit(self, renderer: Renderer) -> None:
        """Submit a non-blocking copy from the headless texture.

        Args:
            renderer: The :class:`Renderer` that owns the headless texture.

        Raises:
            RuntimeError: If the ring buffer is full.
        """
        ...

    def try_recv(self) -> Optional[dict]:
        """Return the next ready frame as ``dict``, or ``None``.

        The returned dict has:
          - ``"pixels"``: ``bytes`` — tightly-packed pixel data.
          - ``"frame_index"``: ``int`` — zero-based index.
        """
        ...

    def flush(self, renderer: Renderer) -> list[dict]:
        """Block until all in-flight frames are returned.

        Args:
            renderer: The :class:`Renderer` that owns the GPU device.

        Returns:
            List of frame dicts with ``"pixels"`` and ``"frame_index"``.
        """
        ...

# ============================================================================
# Engine
# ============================================================================

class Engine:
    """Engine context, available inside ``@app.init`` and ``@app.update`` callbacks."""

    @property
    def time(self) -> float:
        """Total elapsed time since start (seconds)."""
        ...

    @property
    def frame_count(self) -> int:
        """Total number of frames rendered."""
        ...

    @property
    def input(self) -> Input:
        """Read-only input state proxy."""
        ...

    def create_scene(self) -> Scene:
        """Create a new scene and set it as the active scene."""
        ...

    def active_scene(self) -> Optional[Scene]:
        """Get the currently active scene, or ``None`` if none exists."""
        ...

    def load_texture(
        self,
        path: str,
        color_space: str = "srgb",
        generate_mipmaps: bool = True,
    ) -> TextureHandle:
        """Load a 2D texture from a file path.

        Args:
            path: File path to the texture image.
            color_space: ``'srgb'`` or ``'linear'``.
            generate_mipmaps: Whether to generate mip maps.
        """
        ...

    def load_hdr_texture(self, path: str) -> TextureHandle:
        """Load an HDR environment texture (e.g. ``.hdr`` files)."""
        ...

    def load_gltf(self, path: str) -> Object3D:
        """Load a glTF/GLB model and instantiate it in the active scene.

        Returns the root ``Object3D`` node of the loaded model.
        """
        ...

    def set_title(self, title: str) -> None:
        """Set the window title (only works when using ``App``)."""
        ...

# ============================================================================
# Scene
# ============================================================================

class Scene:
    """A scene that holds objects, lights, cameras, and environment settings.

    Obtained via ``engine.create_scene()``.
    """

    active_camera: Optional[Object3D]
    """The currently active camera node. Set this to render from a specific camera."""

    def add_mesh(
        self,
        geometry: Union[BoxGeometry, SphereGeometry, PlaneGeometry, Geometry],
        material: Union[UnlitMaterial, PhongMaterial, PhysicalMaterial],
    ) -> Object3D:
        """Add a mesh to the scene.

        Args:
            geometry: A geometry object (BoxGeometry, SphereGeometry, etc.)
            material: A material object (UnlitMaterial, PhongMaterial, etc.)

        Returns:
            An Object3D handle for the created mesh node.
        """
        ...

    def add_camera(
        self,
        camera: Union[PerspectiveCamera, OrthographicCamera],
    ) -> Object3D:
        """Add a camera to the scene.

        Returns an Object3D node handle. Set ``scene.active_camera = node``
        to render from it.
        """
        ...

    def add_light(
        self,
        light: Union[DirectionalLight, PointLight, SpotLight],
    ) -> Object3D:
        """Add a light to the scene.

        Returns an Object3D node for the light (position it with ``.position``).
        """
        ...

    def attach(self, child: Object3D, parent: Object3D) -> None:
        """Attach a child node to a parent node in the scene hierarchy."""
        ...

    def find_node_by_name(self, name: str) -> Optional[Object3D]:
        """Find a node by name. Returns ``None`` if not found."""
        ...

    # ---- Background & Environment ----

    def set_background_color(self, r: float, g: float, b: float) -> None:
        """Set the background to a solid color (components in 0..1)."""
        ...

    def set_environment_map(self, tex: TextureHandle) -> None:
        """Set the environment map for image-based lighting (IBL) and skybox."""
        ...

    def set_environment_intensity(self, intensity: float) -> None:
        """Set the intensity of environment lighting."""
        ...

    def set_ambient_light(self, r: float, g: float, b: float) -> None:
        """Set the ambient light color for environment lighting."""
        ...

    # ---- Post-processing ----

    def set_bloom_enabled(self, enabled: bool) -> None:
        """Enable or disable bloom."""
        ...

    def set_bloom_strength(self, strength: float) -> None:
        """Set bloom strength (e.g. 0.04)."""
        ...

    def set_bloom_radius(self, radius: float) -> None:
        """Set bloom radius (e.g. 0.005)."""
        ...

    def set_ssao_enabled(self, enabled: bool) -> None:
        """Enable or disable Screen-Space Ambient Occlusion."""
        ...

    def set_ssao_radius(self, radius: float) -> None:
        """Set SSAO sampling radius."""
        ...

    def set_ssao_bias(self, bias: float) -> None:
        """Set SSAO bias."""
        ...

    def set_ssao_intensity(self, intensity: float) -> None:
        """Set SSAO intensity."""
        ...

    def set_tone_mapping_mode(self, mode: str) -> None:
        """Set the tone mapping mode.

        Supported modes: ``'linear'``, ``'neutral'``, ``'reinhard'``,
        ``'cineon'``, ``'aces'`` / ``'aces_filmic'``, ``'agx'`` / ``'agx_punchy'``.
        """
        ...

    def set_tone_mapping(self, mode: str, exposure: Optional[float] = None) -> None:
        """Set tone mapping mode with optional exposure.

        This is a convenience combining ``set_tone_mapping_mode`` and
        optional exposure.

        Args:
            mode: One of ``'linear'``, ``'neutral'``, ``'reinhard'``,
                  ``'cineon'``, ``'aces'``, ``'agx'``, ``'agx_punchy'``.
            exposure: Exposure value (reserved for future use).
        """
        ...

    def set_bloom(
        self,
        enabled: bool,
        strength: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> None:
        """Enable/disable bloom with optional strength and radius.

        Args:
            enabled: Whether bloom is enabled.
            strength: Bloom strength (e.g. 0.04).
            radius: Bloom radius (e.g. 0.005).
        """
        ...

    # ---- Animation ----

    def play_animation(self, node: Object3D, name: str) -> None:
        """Play a named animation clip on a node."""
        ...

    def play_if_any_animation(self, node: Object3D) -> None:
        """Play any available animation on a node (simple convenience)."""
        ...

    def play_any_animation(self, node: Object3D) -> None:
        """Alias for ``play_if_any_animation``."""
        ...

    def list_animations(self, node: Object3D) -> list[str]:
        """List animation clip names available on a node."""
        ...

    def get_animation_mixer(self, node: Object3D) -> Optional[AnimationMixer]:
        """Get the animation mixer for a node (for advanced control)."""
        ...

# ============================================================================
# Object3D
# ============================================================================

# ============================================================================
# Component Proxy Types
# ============================================================================

# --- Camera Components ---

class PerspectiveCameraComponent:
    """Runtime proxy for a perspective camera attached to a scene node.

    Obtained via ``node.camera`` when the node carries a perspective camera.
    """

    fov: float
    """Vertical field of view in degrees."""
    aspect: float
    """Width / height aspect ratio."""
    near: float
    """Near clipping plane distance."""
    far: float
    """Far clipping plane distance."""
    antialiasing: AntiAliasing
    """Anti-aliasing configuration."""

class OrthographicCameraComponent:
    """Runtime proxy for an orthographic camera attached to a scene node.

    Obtained via ``node.camera`` when the node carries an orthographic camera.
    """

    size: float
    """Orthographic view half-height."""
    near: float
    """Near clipping plane distance."""
    far: float
    """Far clipping plane distance."""
    antialiasing: AntiAliasing
    """Anti-aliasing configuration."""

AnyCameraComponent = Union[PerspectiveCameraComponent, OrthographicCameraComponent]
"""Union of all camera component proxy types."""

# --- Light Components ---

class DirectionalLightComponent:
    """Runtime proxy for a directional light attached to a scene node.

    Obtained via ``node.light`` when the node carries a directional light.
    """

    @property
    def color(self) -> list[float]:
        """Light color as ``[r, g, b]``."""
        ...
    @color.setter
    def color(self, val: ColorInput) -> None: ...

    intensity: float
    """Light intensity in lux."""
    cast_shadows: bool
    """Whether this light casts shadows."""

class PointLightComponent:
    """Runtime proxy for a point light attached to a scene node.

    Obtained via ``node.light`` when the node carries a point light.
    """

    @property
    def color(self) -> list[float]:
        """Light color as ``[r, g, b]``."""
        ...
    @color.setter
    def color(self, val: ColorInput) -> None: ...

    intensity: float
    """Light intensity in candela."""
    range: float
    """Maximum effective range (0 = infinite)."""
    cast_shadows: bool
    """Whether this light casts shadows."""

class SpotLightComponent:
    """Runtime proxy for a spot light attached to a scene node.

    Obtained via ``node.light`` when the node carries a spot light.
    """

    @property
    def color(self) -> list[float]:
        """Light color as ``[r, g, b]``."""
        ...
    @color.setter
    def color(self, val: ColorInput) -> None: ...

    intensity: float
    """Light intensity in candela."""
    range: float
    """Maximum effective range."""
    inner_cone: float
    """Inner cone angle in radians."""
    outer_cone: float
    """Outer cone angle in radians."""
    cast_shadows: bool
    """Whether this light casts shadows."""

AnyLightComponent = Union[
    DirectionalLightComponent, PointLightComponent, SpotLightComponent
]
"""Union of all light component proxy types."""

# --- Mesh Component ---

class MeshComponent:
    """Runtime proxy for a mesh component attached to a scene node.

    Obtained via ``node.mesh``.
    """

    visible: bool
    """Whether this mesh is visible."""
    cast_shadows: bool
    """Whether this mesh casts shadows."""
    receive_shadows: bool
    """Whether this mesh receives shadows."""
    render_order: int
    """Draw order override."""

# ============================================================================
# Object3D
# ============================================================================

class Object3D:
    """A 3D object (node) in the scene.

    Provides transform controls and component proxy accessors for camera,
    light, and mesh data attached to this node.
    """

    @property
    def position(self) -> list[float]:
        """Position as ``[x, y, z]``."""
        ...

    @position.setter
    def position(self, val: Vec3Input) -> None: ...
    @property
    def rotation(self) -> list[float]:
        """Euler rotation in radians as ``[x, y, z]`` (XYZ order)."""
        ...

    @rotation.setter
    def rotation(self, val: Vec3Input) -> None: ...
    @property
    def scale(self) -> list[float]:
        """Scale as ``[x, y, z]``."""
        ...

    @scale.setter
    def scale(self, val: Vec3Input) -> None: ...

    visible: bool
    """Whether this object is visible."""

    cast_shadows: bool
    """Whether this mesh casts shadows (only meaningful for mesh nodes)."""

    receive_shadows: bool
    """Whether this mesh receives shadows (only meaningful for mesh nodes)."""

    name: Optional[str]
    """Node name."""

    @property
    def rotation_euler(self) -> list[float]:
        """Euler rotation in radians as ``[x, y, z]`` (XYZ order)."""
        ...

    @rotation_euler.setter
    def rotation_euler(self, val: Vec3Input) -> None: ...
    def set_uniform_scale(self, s: float) -> None:
        """Set uniform scale (same value for x, y, z)."""
        ...

    def rotate_x(self, angle: float) -> None:
        """Rotate around the local X axis by ``angle`` radians."""
        ...

    def rotate_y(self, angle: float) -> None:
        """Rotate around the local Y axis by ``angle`` radians."""
        ...

    def rotate_z(self, angle: float) -> None:
        """Rotate around the local Z axis by ``angle`` radians."""
        ...

    def rotate_world_x(self, angle: float) -> None:
        """Rotate around the **world** X axis by ``angle`` radians."""
        ...

    def rotate_world_y(self, angle: float) -> None:
        """Rotate around the **world** Y axis by ``angle`` radians."""
        ...

    def rotate_world_z(self, angle: float) -> None:
        """Rotate around the **world** Z axis by ``angle`` radians."""
        ...

    def look_at(self, target: Vec3Input) -> None:
        """Rotate this node to face a world-space target position."""
        ...

    # ---- Component Proxies ----

    @property
    def light(self) -> Optional[AnyLightComponent]:
        """Access the light component on this node.

        Returns a typed proxy (``DirectionalLightComponent``,
        ``PointLightComponent``, or ``SpotLightComponent``) depending
        on the light kind, or ``None`` if this node has no light.
        """
        ...

    @property
    def camera(self) -> Optional[AnyCameraComponent]:
        """Access the camera component on this node.

        Returns a typed proxy (``PerspectiveCameraComponent`` or
        ``OrthographicCameraComponent``) depending on the projection
        type, or ``None`` if this node has no camera.
        """
        ...

    @property
    def mesh(self) -> Optional[MeshComponent]:
        """Access the mesh component on this node.

        Returns a ``MeshComponent`` proxy, or ``None`` if this node
        has no mesh.
        """
        ...

# ============================================================================
# FrameState
# ============================================================================

class FrameState:
    """Per-frame state information, passed to the ``@app.update`` callback."""

    @property
    def delta_time(self) -> float:
        """Time elapsed since last frame (seconds)."""
        ...

    @property
    def elapsed(self) -> float:
        """Total time since application start (seconds)."""
        ...

    @property
    def frame_count(self) -> int:
        """Total frame count since start."""
        ...

    @property
    def dt(self) -> float:
        """Alias for ``delta_time``."""
        ...

    @property
    def time(self) -> float:
        """Alias for ``elapsed``."""
        ...

# ============================================================================
# Geometry
# ============================================================================

class BoxGeometry:
    """A box (cuboid) geometry.

    Args:
        width: Width along X axis.
        height: Height along Y axis.
        depth: Depth along Z axis.
    """

    width: float
    height: float
    depth: float

    def __init__(
        self, width: float = 1.0, height: float = 1.0, depth: float = 1.0
    ) -> None: ...

class SphereGeometry:
    """A sphere geometry.

    Args:
        radius: Sphere radius.
        width_segments: Horizontal segments.
        height_segments: Vertical segments.
    """

    radius: float
    width_segments: int
    height_segments: int

    def __init__(
        self,
        radius: float = 1.0,
        width_segments: int = 32,
        height_segments: int = 16,
    ) -> None: ...

class PlaneGeometry:
    """A plane geometry.

    Args:
        width: Width along X axis.
        height: Height along Z axis.
    """

    width: float
    height: float

    def __init__(self, width: float = 1.0, height: float = 1.0) -> None: ...

class Geometry:
    """A custom geometry built from raw vertex data.

    Example::

        geo = myth.Geometry()
        geo.set_positions([0, 0, 0, 1, 0, 0, 0, 1, 0])
        geo.set_indices([0, 1, 2])
    """

    def __init__(self) -> None: ...
    def set_positions(self, data: list[float]) -> None:
        """Set vertex positions as a flat list ``[x0, y0, z0, x1, y1, z1, ...]``."""
        ...

    def set_normals(self, data: list[float]) -> None:
        """Set vertex normals as a flat list ``[nx0, ny0, nz0, ...]``."""
        ...

    def set_uvs(self, data: list[float]) -> None:
        """Set UV coordinates as a flat list ``[u0, v0, u1, v1, ...]``."""
        ...

    def set_indices(self, data: list[int]) -> None:
        """Set triangle index buffer."""
        ...

# ============================================================================
# Materials
# ============================================================================

class UnlitMaterial:
    """An unlit material with flat color.

    Args:
        color: Color — hex string ``'#ff0000'``, ``[r, g, b]`` list, or ``(r, g, b)`` tuple.
        opacity: Opacity (0.0–1.0).
        side: Face culling — ``'front'``, ``'back'``, or ``'double'``.
    """

    color: list[float]
    """Diffuse color as ``[r, g, b]``. Can be set with ``[r, g, b]``, ``(r, g, b)``, or hex string."""
    opacity: float
    """Opacity (0.0–1.0)."""

    def __init__(
        self,
        color: ColorInput = "#ffffff",
        opacity: float = 1.0,
        side: str = "front",
    ) -> None: ...
    def set_map(self, tex: TextureHandle) -> None:
        """Set the color (diffuse) texture map."""
        ...

class PhongMaterial:
    """A Blinn-Phong material with specular highlights.

    Args:
        color: Diffuse color — hex string, ``[r, g, b]`` list, or ``(r, g, b)`` tuple.
        specular: Specular color.
        emissive: Emissive color.
        shininess: Specular exponent.
        emissive_intensity: Emissive intensity multiplier.
        opacity: Opacity.
        side: Face culling.
        alpha_mode: Alpha blending mode.
        depth_write: Whether to write to depth buffer.
    """

    color: list[float]
    specular: list[float]
    """Specular highlight color as ``[r, g, b]``."""
    emissive: list[float]
    """Emissive color as ``[r, g, b]``."""
    emissive_intensity: float
    """Emissive intensity multiplier."""
    shininess: float
    opacity: float
    alpha_mode: str
    """Alpha blending mode: ``'opaque'``, ``'blend'``, or ``'mask'``."""
    depth_write: bool
    """Whether this material writes to the depth buffer."""

    def __init__(
        self,
        color: ColorInput = "#ffffff",
        specular: ColorInput = "#111111",
        emissive: ColorInput = "#000000",
        shininess: float = 30.0,
        emissive_intensity: float = 1.0,
        opacity: float = 1.0,
        side: str = "front",
        alpha_mode: str = "opaque",
        depth_write: bool = True,
    ) -> None: ...
    def set_map(self, tex: TextureHandle) -> None:
        """Set the diffuse texture map."""
        ...

    def set_normal_map(
        self, tex: TextureHandle, scale: Optional[list[float]] = None
    ) -> None:
        """Set the normal map with optional scale ``[sx, sy]`` or ``[s]`` (default ``[1, 1]``)."""
        ...

    def set_specular_map(self, tex: TextureHandle) -> None:
        """Set the specular highlight texture map."""
        ...

    def set_emissive_map(self, tex: TextureHandle) -> None:
        """Set the emissive texture map."""
        ...

class PhysicalMaterial:
    """A PBR metallic-roughness material.

    Args:
        color: Base color — hex string, ``[r, g, b]`` list, or ``(r, g, b)`` tuple.
        metalness: Metalness factor (0.0–1.0).
        roughness: Roughness factor (0.0–1.0).
        emissive: Emissive color — hex string, ``[r, g, b]`` list, or ``(r, g, b)`` tuple.
        emissive_intensity: Emissive intensity multiplier.
        opacity: Opacity.
        side: Face culling.
    """

    color: list[float]
    """Base color as ``[r, g, b]``."""
    metalness: float
    roughness: float
    emissive_intensity: float
    opacity: float
    clearcoat: float
    clearcoat_roughness: float
    transmission: float
    ior: float

    def __init__(
        self,
        color: ColorInput = "#ffffff",
        metalness: float = 0.0,
        roughness: float = 0.5,
        emissive: ColorInput = "#000000",
        emissive_intensity: float = 1.0,
        opacity: float = 1.0,
        side: str = "front",
        alpha_mode: str = "opaque",
        depth_write: bool = True,
    ) -> None: ...

    alpha_mode: str
    """Alpha blending mode: ``'opaque'``, ``'blend'``, or ``'mask'``."""
    depth_write: bool
    """Whether this material writes to the depth buffer."""

    def set_map(self, tex: TextureHandle) -> None:
        """Set the base color texture map."""
        ...

    def set_normal_map(self, tex: TextureHandle, scale: Optional[float] = None) -> None:
        """Set the normal map with optional scale."""
        ...

    def set_roughness_map(self, tex: TextureHandle) -> None:
        """Set the roughness texture map."""
        ...

    def set_metalness_map(self, tex: TextureHandle) -> None:
        """Set the metalness texture map."""
        ...

    def set_emissive_map(self, tex: TextureHandle) -> None:
        """Set the emissive texture map."""
        ...

    def set_ao_map(self, tex: TextureHandle) -> None:
        """Set the ambient occlusion texture map."""
        ...

# ============================================================================
# Camera
# ============================================================================

class AntiAliasing:
    """Anti-aliasing configuration modes."""

    @staticmethod
    def none() -> AntiAliasing:
        """No anti-aliasing. Maximum performance."""
        ...

    @staticmethod
    def msaa(samples: int = 4) -> AntiAliasing:
        """Hardware multi-sampling.

        Args:
            samples: Number of samples (e.g., 2, 4, 8). Default is 4.
        """
        ...

    @staticmethod
    def fxaa(quality: Optional[str] = None) -> AntiAliasing:
        """FXAA only. Good for low-end / Web targets.

        Args:
            quality: 'low', 'medium', 'high', or 'extreme'. Defaults to 'medium'.
        """
        ...

    @staticmethod
    def msaa_fxaa(samples: int = 4, quality: Optional[str] = None) -> AntiAliasing:
        """MSAA + FXAA. Best static image quality with zero temporal ghosting."""
        ...

    @staticmethod
    def taa(
        feedback_weight: float = 0.9, sharpen_intensity: float = 0.5
    ) -> AntiAliasing:
        """Temporal Anti-Aliasing (Recommended for PBR).

        Args:
            feedback_weight: History frame blend weight (0.0 - 1.0).
                Higher values produce smoother results but increase ghosting.
            sharpen_intensity: Contrast Adaptive Sharpening intensity (0.0 - 1.0).
        """
        ...

    @staticmethod
    def taa_fxaa(
        feedback_weight: float = 0.9,
        sharpen_intensity: float = 0.5,
        quality: Optional[str] = None,
    ) -> AntiAliasing:
        """TAA + FXAA. TAA handles temporal aliasing, FXAA provides extra smoothing."""
        ...

class PerspectiveCamera:
    """A perspective projection camera.

    Args:
        fov: Vertical field of view in degrees.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        aspect: Width / height aspect ratio. ``0`` (default) = auto-detect from renderer size.
        position: Initial camera position ``[x, y, z]``.
    """

    fov: float
    aspect: float
    near: float
    far: float
    position: list[float]
    anti_aliasing: AntiAliasing

    def __init__(
        self,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 1000.0,
        aspect: float = 0.0,
        position: list[float] = ...,
        anti_aliasing: Optional[AntiAliasing] = None,
    ) -> None: ...

class OrthographicCamera:
    """An orthographic projection camera.

    Args:
        size: Orthographic view height.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        position: Initial camera position ``[x, y, z]``.
    """

    size: float
    near: float
    far: float
    position: list[float]
    anti_aliasing: AntiAliasing

    def __init__(
        self,
        size: float = 10.0,
        near: float = 0.1,
        far: float = 1000.0,
        position: list[float] = ...,
        anti_aliasing: Optional[AntiAliasing] = None,
    ) -> None: ...

# ============================================================================
# Lights
# ============================================================================

class DirectionalLight:
    """A directional light (like the sun).

    Args:
        color: Light color as ``[r, g, b]``.
        intensity: Light intensity multiplier.
        cast_shadows: Whether this light casts shadows.
    """

    color: list[float]
    intensity: float
    cast_shadows: bool

    def __init__(
        self,
        color: list[float] = ...,
        intensity: float = 1.0,
        cast_shadows: bool = False,
    ) -> None: ...

class PointLight:
    """A point light that emits in all directions.

    Args:
        color: Light color as ``[r, g, b]``.
        intensity: Light intensity.
        range: Maximum range (0 = infinite).
        cast_shadows: Whether this light casts shadows.
    """

    color: list[float]
    intensity: float
    range: float
    cast_shadows: bool

    def __init__(
        self,
        color: list[float] = ...,
        intensity: float = 1.0,
        range: float = 10.0,
        cast_shadows: bool = False,
    ) -> None: ...

class SpotLight:
    """A spotlight that emits in a cone shape.

    Args:
        color: Light color as ``[r, g, b]``.
        intensity: Light intensity.
        range: Maximum range.
        inner_cone: Inner cone angle in radians.
        outer_cone: Outer cone angle in radians.
        cast_shadows: Whether this light casts shadows.
    """

    color: list[float]
    intensity: float
    range: float
    inner_cone: float
    outer_cone: float
    cast_shadows: bool

    def __init__(
        self,
        color: list[float] = ...,
        intensity: float = 1.0,
        range: float = 10.0,
        inner_cone: float = 0.3,
        outer_cone: float = 0.5,
        cast_shadows: bool = False,
    ) -> None: ...

# ============================================================================
# Texture
# ============================================================================

class TextureHandle:
    """An opaque handle to a loaded texture.

    Obtain via ``engine.load_texture()`` or ``engine.load_hdr_texture()``.
    Pass to material methods like ``mat.set_map(handle)``.
    """

    def __eq__(self, other: object) -> bool: ...

# ============================================================================
# Controls
# ============================================================================

class OrbitControls:
    """Three.js-style orbit camera controls.

    Args:
        position: Initial camera position ``[x, y, z]``.
        target: Point to orbit around ``[x, y, z]``.
    """

    enable_damping: bool
    damping_factor: float
    rotate_speed: float
    zoom_speed: float
    pan_speed: float
    min_distance: float
    max_distance: float

    def __init__(
        self,
        position: list[float] = ...,
        target: list[float] = ...,
    ) -> None: ...
    def update(self, camera: Object3D, dt: float) -> None:
        """Update the orbit controls. Call every frame in ``@app.update``.

        Args:
            camera: The camera Object3D node.
            dt: Delta time in seconds (from ``frame.delta_time``).
        """
        ...

    def set_target(self, target: list[float]) -> None:
        """Set the orbit target point ``[x, y, z]``."""
        ...

    def fit(self, node: Object3D) -> None:
        """Adjust orbit position and target to frame a given node's bounding box."""
        ...

# ============================================================================
# Input
# ============================================================================

class Input:
    """Read-only input state proxy.

    Access via ``engine.input`` inside ``@app.update`` callbacks.
    """

    def key(self, name: str) -> bool:
        """Returns ``True`` if the key is currently held down.

        Key names: ``'a'``–``'z'``, ``'0'``–``'9'``, ``'Space'``, ``'Enter'``,
        ``'Escape'``, ``'Tab'``, ``'Shift'``, ``'Ctrl'``, ``'Alt'``,
        ``'ArrowUp'``, ``'ArrowDown'``, ``'ArrowLeft'``, ``'ArrowRight'``,
        ``'F1'``–``'F12'``.
        """
        ...

    def key_down(self, name: str) -> bool:
        """Returns ``True`` on the frame the key was first pressed."""
        ...

    def key_up(self, name: str) -> bool:
        """Returns ``True`` on the frame the key was released."""
        ...

    def mouse_button(self, name: str) -> bool:
        """Returns ``True`` if the mouse button is currently held.

        Button names: ``'Left'``, ``'Right'``, ``'Middle'``.
        """
        ...

    def mouse_button_down(self, name: str) -> bool:
        """Returns ``True`` on the frame the mouse button was first pressed."""
        ...

    def mouse_button_up(self, name: str) -> bool:
        """Returns ``True`` on the frame the mouse button was released."""
        ...

    def mouse_position(self) -> list[float]:
        """Current mouse position in window pixels ``[x, y]``."""
        ...

    def mouse_delta(self) -> list[float]:
        """Mouse movement delta since last frame ``[dx, dy]``."""
        ...

    def scroll_delta(self) -> list[float]:
        """Mouse scroll wheel delta since last frame ``[dx, dy]``."""
        ...

# ============================================================================
# Animation
# ============================================================================

class AnimationMixer:
    """Animation mixer attached to a node for advanced animation control.

    Obtain via ``scene.get_animation_mixer(node)``.
    """

    def list_animations(self) -> list[str]:
        """List all animation clip names available on this mixer."""
        ...

    def play(self, name: str) -> None:
        """Play an animation by name."""
        ...

    def stop(self, name: str) -> None:
        """Stop a specific animation by name."""
        ...

    def stop_all(self) -> None:
        """Stop all animations."""
        ...
