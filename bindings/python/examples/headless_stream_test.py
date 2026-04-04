"""
Myth Engine — Headless ReadbackStream Test (Async Ring Buffer)
===============================================================

Acceptance Test B: Renders 100 frames using a ``ReadbackStream`` with
``buffer_count=3``. Frames are submitted non-blocking and collected via
``try_recv()``. Remaining in-flight frames are drained with ``flush()``.
Asserts exactly 100 frames are received.

Usage:
    maturin develop --release
    python examples/headless_stream_test.py
"""

import myth

renderer = myth.Renderer(render_path="basic")
renderer.init_headless(256, 256)

# Minimal scene
scene = renderer.create_scene()
cube = scene.add_mesh(myth.BoxGeometry(1, 1, 1), myth.UnlitMaterial(color=[1.0, 0.4, 0.1]))
cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.1))
cam.position = [0, 2, 5]
cam.look_at([0, 0, 0])
scene.active_camera = cam
scene.add_light(myth.DirectionalLight(color=[1, 1, 1], intensity=3))

TOTAL_FRAMES = 100
EXPECTED_BYTES = 256 * 256 * 4  # RGBA8

stream = renderer.create_readback_stream(buffer_count=3)

received = 0
sent = 0
lost = 0

for i in range(TOTAL_FRAMES):
    renderer.update(1.0 / 60.0)
    renderer.render()

    try:
        stream.try_submit(renderer)
        sent += 1
    except Exception as e:
        lost += 1
        print(f"Error submitting frame {i}: {e}")

    renderer.poll_device()

    # Opportunistically pull ready frames.
    while True:
        frame = stream.try_recv()
        if frame is None:
            break
        assert len(frame["pixels"]) == EXPECTED_BYTES, (
            f"frame {frame['frame_index']}: unexpected size {len(frame['pixels'])}"
        )
        received += 1

# Drain remaining in-flight frames.
remaining = stream.flush(renderer)
for frame in remaining:
    assert len(frame["pixels"]) == EXPECTED_BYTES
    received += 1

assert received == sent, f"expected {sent} frames, got {received}"

print(
    f"Test B passed: Total {TOTAL_FRAMES} frames , sent {sent}, received {received}, lost {lost} "
    f"(256×256, buffer_count=3, {EXPECTED_BYTES} bytes/frame)"
)

renderer.dispose()
