"""
Myth Engine — Earth Demo
=========================

A rotating Earth globe with texture mapping, normal maps, emissive night lights,
and a cloud layer. Ported from myth-engine/examples/earth.rs.

Usage:
    python examples/earth.py

Assets (from myth-engine/examples/assets/planets/):
    earth_atmos_4096.jpg, earth_specular_2048.jpg,
    earth_lights_2048.png, earth_normal_2048.jpg, earth_clouds_1024.png
"""

import myth

from __asset_utils import get_asset

ASSETS = get_asset("planets")

# ── App ──────────────────────────────────────────────────────────────────────
app = myth.App(
    title="Earth",
    render_path=myth.RenderPath.HIGH_FIDELITY,
    vsync=False,
)

# ── Geometry ─────────────────────────────────────────────────────────────────
earth_geo = myth.SphereGeometry(radius=63.71, width_segments=100, height_segments=50)

# ── Materials (created in on_init after textures are loaded) ─────────────────
orbit = myth.OrbitControls(position=[0, 0, 250], target=[0, 0, 0])

earth = None
clouds = None
cam = None

# FPS counter
fps_interval = 0.5
fps_accum_time = 0.0
fps_accum_frames = 0


@app.init
def on_init(ctx: myth.Engine):
    global earth, clouds, cam

    # ── Load textures ────────────────────────────────────────────────────
    earth_tex = ctx.load_texture(f"{ASSETS}/earth_atmos_4096.jpg", color_space="srgb")
    specular_tex = ctx.load_texture(
        f"{ASSETS}/earth_specular_2048.jpg", color_space="srgb"
    )
    emissive_tex = ctx.load_texture(
        f"{ASSETS}/earth_lights_2048.png", color_space="srgb"
    )
    normal_tex = ctx.load_texture(
        f"{ASSETS}/earth_normal_2048.jpg", color_space="linear"
    )
    clouds_tex = ctx.load_texture(f"{ASSETS}/earth_clouds_1024.png", color_space="srgb")

    # ── Earth material (Phong with textures — matches Rust earth.rs) ────
    earth_mat = myth.PhongMaterial(
        color="#ffffff",
        shininess=10.0,
        emissive=[0.0962, 0.0962, 0.0512],
        emissive_intensity=3.0,
    )
    earth_mat.set_map(earth_tex)
    earth_mat.set_normal_map(normal_tex, scale=[0.85, -0.85])
    earth_mat.set_specular_map(specular_tex)
    earth_mat.set_emissive_map(emissive_tex)

    # ── Cloud material (semi-transparent Phong overlay) ────────────────
    cloud_mat = myth.PhongMaterial(
        color="#ffffff",
        shininess=0.0,
        opacity=0.8,
        side="front",
        alpha_mode="blend",
        depth_write=False,
    )
    cloud_mat.set_map(clouds_tex)

    # ── Scene ────────────────────────────────────────────────────────────
    scene = ctx.create_scene()

    # Earth mesh
    earth = scene.add_mesh(earth_geo, earth_mat)
    earth.rotation_euler = [0, -57, 0]  # initial orientation

    # Cloud mesh (slightly larger sphere)
    clouds = scene.add_mesh(earth_geo, cloud_mat)
    clouds.scale = [1.005, 1.005, 1.005]
    clouds.rotation_euler = [0, 0, 23.5]

    # ── Sun light ────────────────────────────────────────────────────────
    sun = scene.add_light(
        myth.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=1.0,
        )
    )
    sun.position = [300, 0, 100]
    sun.look_at([0, 0, 0])

    # Very dim ambient
    scene.set_ambient_light(0.0001, 0.0001, 0.0001)

    # ── Camera ───────────────────────────────────────────────────────────
    cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.1))
    cam.position = [0, 0, 250]
    cam.look_at([0, 0, 0])
    scene.active_camera = cam

    # ── Background ───────────────────────────────────────────────────────
    # scene.set_background_color(0.0, 0.0, 0.02)

    # ── Post-processing ──────────────────────────────────────────────────
    scene.set_tone_mapping("neutral", exposure=1.0)

    print("Earth scene initialized!")


@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    global fps_accum_time, fps_accum_frames

    dt = frame.dt

    # ── Rotate Earth ─────────────────────────────────────────────────────
    earth.rotate_world_y(0.001 * 60.0 * dt)

    # ── Rotate cloud layer (slightly faster, world-space Y) ──────────────
    clouds.rotate_world_y(0.00125 * 60.0 * dt)

    # ── Orbit controls ───────────────────────────────────────────────────
    orbit.update(cam, dt)

    # ── FPS ──────────────────────────────────────────────────────────────
    fps_accum_time += dt
    fps_accum_frames += 1
    if fps_accum_time >= fps_interval:
        fps = fps_accum_frames / fps_accum_time
        ctx.set_title(f"Earth | FPS: {fps:.1f}")
        fps_accum_time = 0.0
        fps_accum_frames = 0


app.run()
