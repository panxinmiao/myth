"""
Myth Engine — Simple Recording API Test
=========================================

Demonstrates the **simple mode** recording API:
``start_recording`` → ``render_and_record`` → ``try_pull_frame`` → ``flush_recording``.
No explicit ``submit`` / ``poll_device`` calls required.

Usage:
    maturin develop --release
    python examples/headless_simple_test.py
"""

import myth

renderer = myth.Renderer(render_path="basic")
renderer.init_headless(256, 256)

scene = renderer.create_scene()
scene.add_mesh(myth.BoxGeometry(1, 1, 1), myth.UnlitMaterial(color=[0.2, 0.8, 0.3]))
cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.1))
cam.position = [0, 2, 5]
cam.look_at([0, 0, 0])
scene.active_camera = cam
scene.add_light(myth.DirectionalLight(color=[1, 1, 1], intensity=3))

TOTAL = 50
received = 0

renderer.start_recording(buffer_count=3)
for _ in range(TOTAL):
    renderer.render_and_record(dt=1 / 60)
    frame = renderer.try_pull_frame()
    if frame is not None:
        received += 1
remaining = renderer.flush_recording()
received += len(remaining)

assert received == TOTAL, f"expected {TOTAL}, got {received}"
print(f"Simple mode test passed: {TOTAL} frames recorded and received.")
renderer.dispose()
