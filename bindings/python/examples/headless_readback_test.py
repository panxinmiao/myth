"""
Myth Engine — Headless Readback Test (Sync)
============================================

Acceptance Test A: Renders 10 frames in headless mode, reads back each
frame using the synchronous ``readback_pixels()`` API, and verifies the
expected buffer size.

Usage:
    maturin develop --release
    python examples/headless_readback_test.py
"""

import myth

renderer = myth.Renderer(render_path="basic")
renderer.init_headless(256, 256)

# Minimal scene
scene = renderer.create_scene()
cube = scene.add_mesh(myth.BoxGeometry(1, 1, 1), myth.UnlitMaterial(color=[0.2, 0.6, 1.0]))
cam = scene.add_perspective_camera(fov=45, near=0.1)
cam.set_position(0, 2, 5)
cam.look_at([0, 0, 0])
scene.set_active_camera(cam)
scene.add_directional_light(color=[1, 1, 1], intensity=3)

EXPECTED_BYTES = 256 * 256 * 4  # RGBA8

for i in range(10):
    renderer.update(1.0 / 60.0)
    renderer.render()

    pixels = renderer.readback_pixels()
    assert len(pixels) == EXPECTED_BYTES, (
        f"frame {i}: expected {EXPECTED_BYTES} bytes, got {len(pixels)}"
    )
    assert any(b != 0 for b in pixels[:1024]), f"frame {i}: first 1024 bytes all zero"

print(f"Test A passed: 10 synchronous readback frames OK (256×256, {EXPECTED_BYTES} bytes each)")

renderer.dispose()
