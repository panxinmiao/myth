"""
Myth Engine — Shadows & Skinning Demo
=======================================

Load a skinned character model with shadow-casting directional light and
a ground plane that receives shadows. Ported from myth-engine/examples/shadows.rs.

Usage:
    python examples/shadows.py
    python examples/shadows.py path/to/model.glb
"""

import sys
import myth

from __asset_utils import get_asset

# ── Configuration ────────────────────────────────────────────────────────────
ENV_MAP_PATH = get_asset("envs/royal_esplanade_2k.hdr.jpg")
DEFAULT_MODEL = get_asset("Michelle.glb")
model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

# ── App ──────────────────────────────────────────────────────────────────────
app = myth.App(
    title="Shadows & Skinning",
    render_path=myth.RenderPath.BASIC,
    vsync=False,
)

orbit = myth.OrbitControls(position=[0, 1.5, 4], target=[0, 1, 0])

# Geometry & materials for ground plane
ground_geo = myth.PlaneGeometry(width=30, height=30)
ground_mat = myth.PhongMaterial(
    color=[0.2, 0.3, 0.4],
    side="double",
)

cam = None

# FPS counter
fps_interval = 0.5
fps_accum_time = 0.0
fps_accum_frames = 0


@app.init
def on_init(ctx: myth.Engine):
    global cam

    scene = ctx.create_scene()

    # ── Environment map ──────────────────────────────────────────────────
    env_tex = ctx.load_texture(ENV_MAP_PATH, color_space="srgb", generate_mipmaps=False)
    scene.set_environment_map(env_tex)

    # ── Directional light with shadows ───────────────────────────────────
    sun = scene.add_light(
        myth.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=5.0,
            cast_shadows=True,
        )
    )
    sun.position = [0, 12, 6]
    sun.look_at([0, 0, 0])

    # ── Ground plane ─────────────────────────────────────────────────────
    ground = scene.add_mesh(ground_geo, ground_mat)
    ground.rotation_euler = [-90, 0, 0]  # Rotate to horizontal
    ground.cast_shadows = False
    ground.receive_shadows = True

    # ── Load skinned model ───────────────────────────────────────────────
    print(f"Loading glTF model: {model_path}")
    try:
        model = ctx.load_gltf(model_path)
        print(f"Successfully loaded: {model_path}")

        # List and play animations
        anims = scene.list_animations(model)
        if anims:
            print(f"Animations: {anims}")
            # Try "SambaDance" first, fall back to first animation
            if "SambaDance" in anims:
                scene.play_animation(model, "SambaDance")
            else:
                scene.play_any_animation(model)
        else:
            print("No animations found in the model.")
    except Exception as e:
        print(f"Error loading model: {e}")

    # ── Camera ───────────────────────────────────────────────────────────
    cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.1))
    cam.position = [0, 1.5, 4]
    cam.look_at([0, 1, 0])
    scene.active_camera = cam

    # ── Background ───────────────────────────────────────────────────────
    scene.set_background_color(0.12, 0.12, 0.15)


@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    global fps_accum_time, fps_accum_frames

    # Orbit controls
    orbit.update(cam, frame.dt)

    # FPS display
    fps_accum_time += frame.dt
    fps_accum_frames += 1
    if fps_accum_time >= fps_interval:
        fps = fps_accum_frames / fps_accum_time
        ctx.set_title(f"Shadows & Skinning | FPS: {fps:.1f}")
        fps_accum_time = 0.0
        fps_accum_frames = 0


app.run()
