"""
Myth Engine — RenderCanvas Integration Demo
=============================================

Demonstrates using Myth's ``Renderer`` API with **rendercanvas** as the
windowing / event backend.

rendercanvas handles window creation and input events; myth engine owns the
GPU surface and drives rendering directly.  Because myth presents frames to
the native surface itself, we bypass rendercanvas's built-in draw scheduling
and run a simple manual loop instead.

Requirements:
    pip install rendercanvas glfw        # glfw is the lightest backend

Usage:
    python examples/rendercanvas_demo.py
"""

import math
import time

import myth

# ── rendercanvas setup ──────────────────────────────────────────
# Use the glfw backend (lightest weight, no extra framework needed).
# You can swap this for rendercanvas.pyside6, rendercanvas.qt, etc.
from rendercanvas.glfw import GlfwRenderCanvas
import glfw

canvas = GlfwRenderCanvas(
    title="Myth Engine — RenderCanvas",
    size=(1280, 720),
)


# ── Helper: get native window handle from rendercanvas ──────────
def get_window_handle(canvas):
    """Extract the native platform window handle from a glfw-based canvas."""
    from rendercanvas.glfw import get_glfw_present_info

    info = get_glfw_present_info(canvas._window)
    return info["window"]


# ── Myth Renderer setup ────────────────────────────────────────
renderer = myth.Renderer(render_path=myth.RenderPath.BASIC, vsync=False)

width, height = canvas.get_physical_size()
hwnd = get_window_handle(canvas)
renderer.init_with_handle(hwnd, width, height)

# ── Scene setup ────────────────────────────────────────────────
scene = renderer.create_scene()

# A rotating cube
cube = scene.add_mesh(
    myth.BoxGeometry(1, 1, 1),
    myth.PhysicalMaterial(color="#ff8033", roughness=0.4, metalness=0.3),
)
cube.position = [0, 1, 0]

# A bobbing sphere
sphere = scene.add_mesh(
    myth.SphereGeometry(radius=0.5),
    myth.PhysicalMaterial(color=[0.2, 0.6, 1.0], roughness=0.3),
)
sphere.position = [2, 0.5, 0]

# Ground plane
ground = scene.add_mesh(
    myth.PlaneGeometry(width=20, height=20),
    myth.PhysicalMaterial(color="#666666", roughness=0.8),
)
ground.rotation_euler = [-90, 0, 0]

# Camera
cam = scene.add_camera(myth.PerspectiveCamera(fov=60, near=0.1))
cam.position = [3, 3, 5]
cam.look_at([0, 0.5, 0])
scene.active_camera = cam

# Lights
sun = scene.add_light(
    myth.DirectionalLight(
        color=[1, 0.95, 0.9],
        intensity=2.0,
        cast_shadows=True,
    )
)
sun.position = [5, 10, 5]
sun.look_at([0, 0, 0])

fill = scene.add_light(
    myth.PointLight(
        color=[0.3, 0.5, 1.0],
        intensity=2.0,
        range=20,
    )
)
fill.position = [-3, 4, -2]

scene.set_background_color(0.05, 0.05, 0.1)
scene.set_tone_mapping("aces", exposure=1.0)

# Orbit controls
orbit = myth.OrbitControls(position=[3, 3, 5], target=[0, 0.5, 0])


# ── Event forwarding: rendercanvas events → myth input system ──
@canvas.add_event_handler("resize")
def on_resize(event):
    w, h = canvas.get_physical_size()
    if w > 0 and h > 0:
        renderer.resize(w, h)


@canvas.add_event_handler("pointer_move")
def on_pointer_move(event):
    # rendercanvas gives logical coordinates; scale to physical
    pr = event.get("pixel_ratio", 1.0)
    renderer.inject_mouse_move(event["x"] * pr, event["y"] * pr)


BUTTON_MAP = {
    1: 0,
    2: 2,
    3: 1,
}  # rendercanvas: 1=left,2=right,3=middle → myth: 0=left,1=middle,2=right


@canvas.add_event_handler("pointer_down")
def on_pointer_down(event):
    btn = BUTTON_MAP.get(event.get("button", 1), 0)
    renderer.inject_mouse_down(btn)


@canvas.add_event_handler("pointer_up")
def on_pointer_up(event):
    btn = BUTTON_MAP.get(event.get("button", 1), 0)
    renderer.inject_mouse_up(btn)


@canvas.add_event_handler("wheel")
def on_scroll(event):
    # rendercanvas wheel events: dx/dy are ~100 per notch
    renderer.inject_scroll(event.get("dx", 0) / 100.0, -event.get("dy", 0) / 100.0)


@canvas.add_event_handler("key_down")
def on_key_down(event):
    key = event.get("key", "")
    if key:
        renderer.inject_key_down(key)


@canvas.add_event_handler("key_up")
def on_key_up(event):
    key = event.get("key", "")
    if key:
        renderer.inject_key_up(key)


# ── Handle resize-during-drag on Windows ─────────────────────────
# On Windows, glfw.poll_events() blocks while the user is dragging
# a window edge. GLFW still fires the framebuffer_size_callback
# though, so we render from there to keep the picture alive.
# We chain with rendercanvas's original callback to keep its internal
# size tracking working.

_orig_on_size = canvas._on_size_change  # bound method


def _on_fb_resize(_win, w, h):
    """GLFW framebuffer-size callback: chain rendercanvas + myth resize/render."""
    _orig_on_size(_win, w, h)  # let rendercanvas update its internal state
    if w > 0 and h > 0:
        renderer.resize(w, h)
        renderer.frame(0)  # render immediately so picture stays alive


glfw.set_framebuffer_size_callback(canvas._window, _on_fb_resize)


# ── Render loop (manual — myth presents directly to the surface) ─
last_time = time.perf_counter()
fps_accum_time = 0.0
fps_accum_frames = 0

print("Scene initialized! Close the window to exit.")

while not canvas.get_closed():
    # Process native events and flush the event queue.
    # _process_events() = _rc_gui_poll() + emit resize + flush pending events.
    # Without the flush, add_event_handler callbacks would never fire.
    canvas._process_events()

    now = time.perf_counter()
    dt = now - last_time
    last_time = now

    # Animate
    cube.rotate_y(dt * 0.5)
    y = 0.5 + 0.3 * math.sin(now * 2.0)
    sphere.position = [2, y, 0]

    # Update orbit controls
    orbit.update(cam, dt)

    # Render (myth presents to its own wgpu surface)
    renderer.frame(dt)

    # FPS counter
    fps_accum_time += dt
    fps_accum_frames += 1
    if fps_accum_time >= 0.5:
        fps = fps_accum_frames / fps_accum_time
        canvas.set_title(f"Myth Engine — RenderCanvas | FPS: {fps:.1f}")
        fps_accum_time = 0.0
        fps_accum_frames = 0

# ── Cleanup ─────────────────────────────────────────────────────
renderer.dispose()
print("Done.")
