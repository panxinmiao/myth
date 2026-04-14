//! [gallery]
//! name = "All Shadows"
//! category = "Shadows"
//! description = "Combined directional, spot, and point-light shadow casting in one scene."
//! order = 430
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

/// Combined shadow example demonstrating all three shadow-casting light types:
/// - **Directional light** (CSM): casts from the upper-right
/// - **Spot light**: aimed at the ground from above-left
/// - **Point light**: omnidirectional cube shadow, hovering near objects
struct ShadowAllLightsDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    point_light_node: NodeHandle,
    time: f32,
}

impl AppHandler for ShadowAllLightsDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        // ── Ground plane ──────────────────────────────────────────────
        let floor = scene.spawn_plane(
            40.0,
            40.0,
            PhysicalMaterial::new(Vec4::new(0.85, 0.85, 0.88, 1.0)).with_side(Side::Double),
            &engine.assets,
        );
        scene
            .node(&floor)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_cast_shadows(false)
            .set_receive_shadows(true);

        // ── Objects ───────────────────────────────────────────────────

        // Central sphere
        let sphere = scene.spawn_sphere(
            1.0,
            PhysicalMaterial::new(Vec4::new(0.2, 0.7, 1.0, 1.0)),
            &engine.assets,
        );
        scene
            .node(&sphere)
            .set_position(0.0, 1.0, 0.0)
            .set_shadows(true, true);

        // Left cube
        let cube = scene.spawn_box(
            1.2,
            1.2,
            1.2,
            PhysicalMaterial::new(Vec4::new(0.9, 0.3, 0.2, 1.0)),
            &engine.assets,
        );
        scene
            .node(&cube)
            .set_position(-3.5, 0.6, 0.0)
            .set_shadows(true, true);

        // Right cube
        let cube2 = scene.spawn_box(
            0.8,
            2.0,
            0.8,
            PhysicalMaterial::new(Vec4::new(0.3, 0.9, 0.2, 1.0)),
            &engine.assets,
        );
        scene
            .node(&cube2)
            .set_position(3.0, 1.0, -1.5)
            .set_shadows(true, true);

        // ── Directional light (CSM) ───────────────────────────────────
        let mut dir_light = Light::new_directional(Vec3::ONE, 0.5);
        dir_light.cast_shadows = true;
        if let Some(shadow) = dir_light.shadow.as_mut() {
            shadow.map_size = 2048;
            shadow.normal_bias = 0.0;
        }
        let dir_node = scene.add_light(dir_light);
        scene
            .node(&dir_node)
            .set_position(8.0, 12.0, 6.0)
            .look_at(Vec3::ZERO);

        // ── Spot light ────────────────────────────────────────────────
        let mut spot = Light::new_spot(Vec3::new(0.0, 0.95, 0.8), 1.0, 30.0, 0.3, 0.5);
        spot.cast_shadows = true;
        if let Some(shadow) = spot.shadow.as_mut() {
            shadow.map_size = 1024;
            shadow.normal_bias = 0.0;
        }
        let spot_node = scene.add_light(spot);
        scene
            .node(&spot_node)
            .set_position(-6.0, 8.0, 4.0)
            .look_at(Vec3::new(-3.5, 0.0, 0.0));

        // ── Point light (omnidirectional cube shadow) ─────────────────
        let mut point = Light::new_point(Vec3::new(1.0, 0.8, 0.5), 1.0, 20.0);
        point.cast_shadows = true;
        if let Some(shadow) = point.shadow.as_mut() {
            shadow.map_size = 1024;
        }
        let point_light_node = scene.add_light(point);
        scene.node(&point_light_node).set_position(0.0, 3.0, 3.0);

        let point_helper = scene.spawn_sphere(
            0.1,
            PhysicalMaterial::new(Vec4::new(1.0, 0.8, 0.5, 1.0))
                .with_emissive(Vec3::new(1.0, 0.8, 0.5), 1.0),
            &engine.assets,
        );

        scene
            .node(&point_helper)
            .set_cast_shadows(false)
            .set_receive_shadows(false);

        scene.attach(point_helper, point_light_node);

        // ── Camera ────────────────────────────────────────────────────
        let cam = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .node(&cam)
            .set_position(10.0, 8.0, 10.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam);

        Self {
            controls: OrbitControls::new(Vec3::new(10.0, 8.0, 10.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
            point_light_node,
            time: 0.0,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // Orbit the point light slowly
        self.time += frame.dt;
        if let Some(node) = scene.get_node_mut(self.point_light_node) {
            let x = 3.0 * self.time.cos();
            let z = 3.0 * self.time.sin();
            node.transform.position = Vec3::new(x, 3.0, z);
        }

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "All Shadows (Dir + Spot + Point) | FPS: {:.2}",
                fps
            ));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<ShadowAllLightsDemo>()
}
