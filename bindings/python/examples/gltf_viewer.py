"""
Myth Engine — glTF Viewer
==========================

Load and display a glTF/GLB model with orbit controls.

Usage:
    python examples/gltf_viewer.py path/to/model.glb
"""

import sys
import myth

from __asset_utils import get_asset

# Get model path from command line
model_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else get_asset("DamagedHelmet/glTF/DamagedHelmet.gltf")
)

app = myth.App(
    title="Myth — glTF Viewer", render_path=myth.RenderPath.HIGH_FIDELITY, vsync=True
)
orbit = myth.OrbitControls()

model = None
cam = None


@app.init
def on_init(ctx: myth.Engine):
    global model, cam
    scene = ctx.create_scene()

    # Load the glTF model
    model = ctx.load_gltf(model_path)
    print(f"Loaded: {model_path}")

    # Camera
    cam = scene.add_camera(
        myth.PerspectiveCamera(
            fov=45, near=0.01, anti_aliasing=myth.AntiAliasing.fxaa()
        )
    )
    cam.position = [0, 1.5, 3]
    cam.look_at([0, 0, 0])
    scene.active_camera = cam

    # Lighting
    sun = scene.add_light(myth.DirectionalLight(intensity=3.0, cast_shadows=True))
    sun.position = [3, 5, 3]

    fill = scene.add_light(
        myth.DirectionalLight(
            color=[0.5, 0.6, 0.8],
            intensity=1.0,
        )
    )
    fill.position = [-3, 2, -1]

    scene.set_environment_map(
        ctx.load_texture(get_asset("royal_esplanade_2k.hdr.jpg"), color_space="srgb")
    )

    # Environment & post-processing
    scene.set_background_color(0.15, 0.15, 0.2)
    scene.set_tone_mapping("aces", exposure=1.0)
    scene.set_bloom(True, strength=0.02)

    # Play animations if any
    scene.play_any_animation(model)

    anims = scene.list_animations(model)
    if anims:
        print(f"Animations: {anims}")


@app.update
def on_update(ctx, frame):
    orbit.update(cam, frame.dt)


app.run()
