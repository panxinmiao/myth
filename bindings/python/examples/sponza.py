"""
Myth Engine — Sponza Demo
==========================

Load the Sponza architectural scene (glTF) with IBL environment lighting,
SSAO, and shadow mapping. Ported from myth-engine/examples/sponza.rs.

The Sponza glTF model is automatically loaded from the KhronosGroup
glTF-Sample-Assets repository (requires network access). You can also pass
a local path as a command-line argument.

Usage:
    python examples/sponza.py
    python examples/sponza.py path/to/Sponza.gltf
"""

import sys
import myth

from __asset_utils import get_asset

# ── Configuration ────────────────────────────────────────────────────────────
ENV_MAP_PATH = get_asset("royal_esplanade_2k.hdr.jpg")

# Accept model path from CLI (default: remote Sponza)
DEFAULT_MODEL = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/refs/heads/main/Models/Sponza/glTF/Sponza.gltf"
model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

# ── App ──────────────────────────────────────────────────────────────────────
app = myth.App(
    title="Sponza Lighting Example",
    render_path=myth.RenderPath.HIGH_FIDELITY,
    vsync=True,
)

orbit = myth.OrbitControls(position=[6, 4, 0], target=[0, 2, 0])
cam = None

# FPS counter
fps_interval = 0.5
fps_accum_time = 0.0
fps_accum_frames = 0


@app.init
def on_init(ctx: myth.Engine):
    global cam

    scene = ctx.create_scene()

    # ── Environment map (IBL) ────────────────────────────────────────────
    env_tex = ctx.load_texture(ENV_MAP_PATH, color_space="srgb", generate_mipmaps=False)
    scene.set_environment_map(env_tex)

    # ── SSAO ─────────────────────────────────────────────────────────────
    scene.set_ssao_enabled(True)

    # ── Directional light with shadows ───────────────────────────────────
    sun = scene.add_light(
        myth.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=5.0,
            cast_shadows=True,
        )
    )
    sun.position = [2, 12, 6]
    sun.look_at([0, 0, 0])

    # ── Camera ───────────────────────────────────────────────────────────
    cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.1))
    cam.position = [6, 4, 0]
    cam.look_at([0, 0, 0])
    cam.camera.antialiasing = myth.AntiAliasing.fxaa()
    scene.active_camera = cam

    # ── Load Sponza model ────────────────────────────────────────────────
    print(f"Loading glTF model: {model_path}")
    try:
        ctx.load_gltf(model_path)
        print("Successfully loaded Sponza model!")
    except Exception as e:
        print(f"Failed to load model: {e}")

    # ── Post-processing ──────────────────────────────────────────────────
    scene.set_tone_mapping("aces", exposure=1.0)

    print("Sponza scene initialized!")


@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    global fps_accum_time, fps_accum_frames

    if cam:
        # Orbit controls
        orbit.update(cam, frame.dt)

    # FPS display
    fps_accum_time += frame.dt
    fps_accum_frames += 1
    if fps_accum_time >= fps_interval:
        fps = fps_accum_frames / fps_accum_time
        ctx.set_title(f"Sponza Lighting Example | FPS: {fps:.0f}")
        fps_accum_time = 0.0
        fps_accum_frames = 0


app.run()
