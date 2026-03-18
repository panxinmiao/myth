"""
Myth Engine — Python Demo
==========================

A simple 3D scene with a rotating cube, ground plane, lighting, and orbit controls.

Usage:
    pip install maturin
    maturin develop --release
    python examples/demo.py
"""

import myth

# 1. Create the application
app = myth.App(
    title="Myth Engine — Python Demo",
    render_path="basic",
    vsync=False,
)
app.clear_color = [0.05, 0.05, 0.1, 1.0]

# 2. Define geometry & materials (before or during init)
cube_geo = myth.BoxGeometry(1, 1, 1)
sphere_geo = myth.SphereGeometry(radius=0.5)
ground_geo = myth.PlaneGeometry(width=20, height=20)

cube_mat = myth.PhysicalMaterial(
    color="#ff8033",
    roughness=0.4,
    metalness=0.3,
)

sphere_mat = myth.PhysicalMaterial(
    color="#3399ff",
    roughness=0.3,
    metalness=0.0,
)

ground_mat = myth.PhysicalMaterial(
    color="#666666",
    roughness=0.8,
    metalness=0.0,
)

# 3. Orbit controls
orbit = myth.OrbitControls(position=[3, 3, 5], target=[0, 0.5, 0])

# 4. Store references (handles) for use across callbacks
cube = None
sphere = None
cam = None


@app.init
def on_init(ctx: myth.Engine):
    global cube, sphere, cam
    scene = ctx.create_scene()

    # Add meshes
    cube = scene.add_mesh(cube_geo, cube_mat)
    cube.position = [0, 1, 0]

    sphere = scene.add_mesh(sphere_geo, sphere_mat)
    sphere.position = [2, 0.5, 0]

    ground = scene.add_mesh(ground_geo, ground_mat)
    ground.rotation_euler = [-90, 0, 0]  # Rotate to be horizontal

    # Camera
    cam = scene.add_camera(myth.PerspectiveCamera(fov=60, near=0.1))
    cam.position = [3, 3, 5]
    cam.look_at([0, 0.5, 0])
    scene.active_camera = cam

    # Lights
    sun = scene.add_light(
        myth.DirectionalLight(
            color=[1, 0.95, 0.9],
            intensity=2.0,
            cast_shadows=True,
        )
    )
    sun.position = [5, 10, 5]
    sun.look_at([0, 0, 0])  # Orient the light toward the scene centre

    fill = scene.add_light(
        myth.PointLight(
            color=[0.3, 0.5, 1.0],
            intensity=2.0,
            range=20,
        )
    )
    fill.position = [-3, 4, -2]

    # Background
    scene.set_background_color(0.05, 0.05, 0.1)

    # Post-processing
    scene.set_tone_mapping("aces", exposure=1.0)
    # scene.set_bloom(True, strength=0.02, radius=0.3)

    print("Scene initialized!")


# 5. FPS counter
fps_interval = 0.5  # seconds between updates
fps_accum_time = 0.0
fps_accum_frames = 0


@app.update
def on_update(ctx: myth.Engine, frame: myth.FrameState):
    global fps_accum_time, fps_accum_frames

    # Rotate the cube
    cube.rotate_y(frame.dt * 0.5)

    # Bob the sphere up and down
    import math

    y = 0.5 + 0.3 * math.sin(frame.time * 2.0)
    sphere.position = [2, y, 0]

    # Update orbit controls (mouse interaction)
    orbit.update(cam, frame.dt)

    # FPS display
    fps_accum_time += frame.dt
    fps_accum_frames += 1
    if fps_accum_time >= fps_interval:
        fps = fps_accum_frames / fps_accum_time
        ctx.set_title(f"Myth Engine — Python Demo | FPS: {fps:.1f}")
        fps_accum_time = 0.0
        fps_accum_frames = 0


# 6. Run!
app.run()
