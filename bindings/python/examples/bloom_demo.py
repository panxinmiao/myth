"""
Myth Engine — Bloom Demo
==========================

Demonstrates the physically-based bloom post-processing effect with
interactive keyboard controls. Ported from myth-engine/examples/bloom.rs.

Controls:
    B       - Toggle bloom on/off
    1 / 2   - Decrease / increase bloom strength
    3 / 4   - Decrease / increase bloom radius
    Up/Down - Adjust exposure
    Mouse   - Orbit camera

Usage:
    python examples/bloom_demo.py
    python examples/bloom_demo.py path/to/model.glb
"""

import sys
import myth

from __asset_utils import get_asset

# ── Configuration ────────────────────────────────────────────────────────────
HDR_ENV = get_asset("blouberg_sunrise_2_1k.hdr")
DEFAULT_MODEL = get_asset("phoenix_bird.glb")
model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

# ── App ──────────────────────────────────────────────────────────────────────
app = myth.App(
    title="Bloom Demo",
    render_path=myth.RenderPath.HIGH_FIDELITY,
    vsync=False,
)

orbit = myth.OrbitControls(position=[0, 0, 3], target=[0, 0, 0])
cam = None

# Bloom state (for title display)
bloom_enabled = True
bloom_strength = 0.4
bloom_radius = 0.005
exposure = 1.0

# FPS counter
fps_interval = 0.5
fps_accum_time = 0.0
fps_accum_frames = 0


@app.init
def on_init(ctx: myth.Engine):
    global cam

    scene = ctx.create_scene()

    # ── HDR environment map ──────────────────────────────────────────────
    env_tex = ctx.load_hdr_texture(HDR_ENV)
    scene.set_environment_map(env_tex)
    scene.set_environment_intensity(1.5)

    # ── Directional light ────────────────────────────────────────────────
    sun = scene.add_light(
        myth.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=3.0,
        )
    )
    sun.position = [3, 5, 3]

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    model = ctx.load_gltf(model_path)
    scene.play_any_animation(model)

    # ── Bloom settings ───────────────────────────────────────────────────
    scene.set_bloom(True, strength=bloom_strength, radius=bloom_radius)

    # ── Tone mapping ─────────────────────────────────────────────────────
    scene.set_tone_mapping("neutral", exposure=exposure)

    # ── Camera ───────────────────────────────────────────────────────────
    cam = scene.add_camera(myth.PerspectiveCamera(fov=45, near=0.1))
    cam.position = [0, 0, 3]
    cam.look_at([0, 0, 0])
    scene.active_camera = cam

    # ── Fit camera to loaded model ───────────────────────────────────────
    orbit.fit(model)

    print("=== Physically-Based Bloom Demo ===")
    print("Controls:")
    print("  B       - Toggle bloom on/off")
    print("  1/2     - Decrease/increase bloom strength")
    print("  3/4     - Decrease/increase bloom radius")
    print("  Up/Down - Adjust exposure")
    print("  Mouse   - Orbit camera")


@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    global bloom_enabled, bloom_strength, bloom_radius, exposure
    global fps_accum_time, fps_accum_frames

    inp = ctx.input
    scene = ctx.active_scene()

    # ── Toggle bloom ─────────────────────────────────────────────────────
    if inp.key_down("b"):
        bloom_enabled = not bloom_enabled
        scene.set_bloom_enabled(bloom_enabled)
        print(f"Bloom: {'ON' if bloom_enabled else 'OFF'}")

    # ── Bloom strength (1 = decrease, 2 = increase) ──────────────────────
    if inp.key_down("1"):
        bloom_strength = max(0.0, bloom_strength - 0.01)
        scene.set_bloom_strength(bloom_strength)
        print(f"Bloom strength: {bloom_strength:.3f}")

    if inp.key_down("2"):
        bloom_strength = min(1.0, bloom_strength + 0.01)
        scene.set_bloom_strength(bloom_strength)
        print(f"Bloom strength: {bloom_strength:.3f}")

    # ── Bloom radius (3 = decrease, 4 = increase) ────────────────────────
    if inp.key_down("3"):
        bloom_radius = max(0.001, bloom_radius - 0.001)
        scene.set_bloom_radius(bloom_radius)
        print(f"Bloom radius: {bloom_radius:.4f}")

    if inp.key_down("4"):
        bloom_radius = min(0.05, bloom_radius + 0.001)
        scene.set_bloom_radius(bloom_radius)
        print(f"Bloom radius: {bloom_radius:.4f}")

    # ── Exposure (Up / Down) ─────────────────────────────────────────────
    if inp.key_down("ArrowUp"):
        exposure += 0.1
        scene.set_tone_mapping("aces", exposure=exposure)
        print(f"Exposure: {exposure:.2f}")

    if inp.key_down("ArrowDown"):
        exposure = max(0.1, exposure - 0.1)
        scene.set_tone_mapping("aces", exposure=exposure)
        print(f"Exposure: {exposure:.2f}")

    # ── Orbit controls ───────────────────────────────────────────────────
    orbit.update(cam, frame.dt)

    # ── FPS ──────────────────────────────────────────────────────────────
    fps_accum_time += frame.dt
    fps_accum_frames += 1
    if fps_accum_time >= fps_interval:
        fps = fps_accum_frames / fps_accum_time
        bloom_status = (
            f"ON s={bloom_strength:.3f} r={bloom_radius:.4f}"
            if bloom_enabled
            else "OFF"
        )
        ctx.set_title(
            f"Bloom Demo | FPS: {fps:.0f} | Bloom: {bloom_status} | Exposure: {exposure:.2f}"
        )
        fps_accum_time = 0.0
        fps_accum_frames = 0


app.run()
