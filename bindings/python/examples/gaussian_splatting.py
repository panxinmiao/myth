"""
Myth Engine — 3D Gaussian Splatting Demo (Python)
==================================================

Loads and renders a 3D Gaussian Splatting point cloud from a compressed NPZ file.

Usage:
    pip install maturin
    maturin develop --release
    python examples/gaussian_splatting.py
"""

import myth

from __asset_utils import get_asset

app = myth.App(
    title="Myth Engine — 3D Gaussian Splatting",
    render_path=myth.RenderPath.HIGH_FIDELITY,
    vsync=True,
)

# Orbit controls — initial camera from training data
orbit = myth.OrbitControls(position=[2.86, 1.52, -0.69], target=[0, 0, 0])

cam = None


@app.init
def on_init(ctx: myth.Engine):
    global cam
    scene = ctx.create_scene()

    # Load the compressed NPZ Gaussian cloud
    cloud = ctx.load_gaussian_npz(get_asset("3dgs/point_cloud.npz"))
    cloud.color_space = "linear"
    scene.add_gaussian_cloud("gaussian_cloud", cloud)

    scene.set_tone_mapping("linear", gamma=1.0 / 2.2)

    # Camera
    cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.1))
    cam.position = [2.86, 1.52, -0.69]
    cam.look_at([0, 0, 0])
    scene.active_camera = cam

    print(f"Loaded Gaussian cloud with {cloud.count} splats")


@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    orbit.update(cam, frame.dt)


app.run()
