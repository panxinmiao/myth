"""
Myth Engine — GLFW Integration Demo
====================================

Demonstrates using Myth's low-level ``Renderer`` API with GLFW,
decoupled from winit. This same pattern works with any GUI library
(PySide6, wxPython, SDL2, Tk, …).

Requirements:
    pip install glfw

Usage:
    python examples/glfw_demo.py
"""

import sys

import glfw  # pip install glfw

import myth


def get_native_window_handle(window):
    """Get the native platform window handle from a GLFW window."""
    if sys.platform == "win32":
        # Windows: HWND via user32.dll (glfw doesn't expose get_win32_window in Python)
        hwnd = glfw.get_win32_window(window)
        return hwnd
    elif sys.platform == "darwin":
        ns_window = glfw.get_cocoa_window(window)
        content_view = ns_window.contentView()
        return content_view.value  # Pointer as int
    else:
        # Linux / X11
        return glfw.get_x11_window(window)


def main():
    # ── GLFW setup ──────────────────────────────────────────────
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)

    # We don't need an OpenGL context (wgpu creates its own Vulkan/DX12 context)
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

    width, height = 1280, 720
    window = glfw.create_window(width, height, "Myth Engine — GLFW", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    # ── Myth Renderer setup ────────────────────────────────────
    renderer = myth.Renderer(render_path=myth.RenderPath.BASIC, vsync=False)

    # Initialize with the native window handle
    hwnd = get_native_window_handle(window)
    renderer.init_with_handle(hwnd, width, height)

    # ── Scene setup (same API as App) ──────────────────────────
    scene = renderer.create_scene()

    # Add a cube
    cube = scene.add_mesh(
        myth.BoxGeometry(1, 1, 1),
        myth.PhysicalMaterial(color="#ff8033", roughness=0.4, metalness=0.3),
    )
    cube.position = [0, 0.5, 0]

    # Ground plane
    ground = scene.add_mesh(
        myth.PlaneGeometry(width=20, height=20),
        myth.PhysicalMaterial(color="#666666", roughness=0.8),
    )
    ground.rotation_euler = [-90, 0, 0]

    # Camera
    cam = scene.add_camera(
        myth.PerspectiveCamera(fov=60, near=0.1, aspect=width / height)
    )
    cam.position = [3, 3, 5]
    cam.look_at([0, 0.5, 0])
    scene.active_camera = cam

    # Lights
    sun = scene.add_light(
        myth.DirectionalLight(color=[1, 0.95, 0.9], intensity=2.0, cast_shadows=True)
    )
    sun.position = [5, 10, 5]
    sun.look_at([0, 0, 0])  # Orient the light toward the scene centre

    scene.set_background_color(0.05, 0.05, 0.1)
    scene.set_tone_mapping("aces")

    # Orbit controls
    orbit = myth.OrbitControls(position=[3, 3, 5], target=[0, 0.5, 0])

    # ── Input forwarding callbacks ────────────────────────────
    def on_resize(win, w, h):
        if w > 0 and h > 0:
            renderer.resize(w, h)

    def on_cursor_pos(win, x, y):
        renderer.inject_mouse_move(x, y)

    # GLFW buttons: 0=left, 1=right, 2=middle
    # Myth buttons: 0=left, 1=middle, 2=right
    GLFW_BUTTON_MAP = {0: 0, 1: 2, 2: 1}

    def on_mouse_button(win, button, action, mods):
        btn = GLFW_BUTTON_MAP.get(button, button)
        if action == glfw.PRESS:
            renderer.inject_mouse_down(btn)
        elif action == glfw.RELEASE:
            renderer.inject_mouse_up(btn)

    def on_scroll(win, xoffset, yoffset):
        renderer.inject_scroll(xoffset, yoffset)

    def on_key(win, key, scancode, action, mods):
        name = glfw.get_key_name(key, scancode)
        if name:
            if action == glfw.PRESS:
                renderer.inject_key_down(name)
            elif action == glfw.RELEASE:
                renderer.inject_key_up(name)

    glfw.set_framebuffer_size_callback(window, on_resize)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    print("Scene initialized! Close the window to exit.")

    # ── Main loop (GLFW-driven, not winit) ─────────────────────
    last_time = glfw.get_time()
    fps_accum_time = 0.0
    fps_accum_frames = 0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        now = glfw.get_time()
        dt = now - last_time
        last_time = now

        # Animate
        cube.rotate_y(dt * 0.5)

        # Update orbit controls
        orbit.update(cam, dt)

        # Render one frame
        renderer.frame(dt)

        # FPS display
        fps_accum_time += dt
        fps_accum_frames += 1
        if fps_accum_time >= 0.5:
            fps = fps_accum_frames / fps_accum_time
            glfw.set_window_title(window, f"Myth Engine \u2014 GLFW | FPS: {fps:.1f}")
            fps_accum_time = 0.0
            fps_accum_frames = 0

    # ── Cleanup ────────────────────────────────────────────────
    renderer.dispose()
    glfw.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
