use myth::prelude::*;
use myth::utils::fps_counter::FpsCounter;

struct ShadowSpotDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for ShadowSpotDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        // Sphere
        let sphere_node =
            scene.spawn_sphere(1.0, PhysicalMaterial::new(Vec4::new(0.2, 0.7, 1.0, 1.0)));
        scene
            .node(&sphere_node)
            .set_position(0.0, 1.0, 0.0)
            .set_shadows(true, true);

        // Floor
        let floor_node = scene.spawn_plane(
            30.0,
            30.0,
            PhysicalMaterial::new(Vec4::new(0.9, 0.9, 0.9, 1.0)).with_side(Side::Double),
        );
        scene
            .node(&floor_node)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_cast_shadows(false)
            .set_receive_shadows(true);

        // Spot light with shadows
        let mut spot = Light::new_spot(Vec3::new(1.0, 0.95, 0.9), 30.0, 40.0, 0.35, 0.55);
        spot.cast_shadows = true;
        if let Some(shadow) = spot.shadow.as_mut() {
            shadow.map_size = 2048;
        }
        let spot_node = scene.add_light(spot);
        scene
            .node(&spot_node)
            .set_position(6.0, 10.0, 4.0)
            .look_at(Vec3::ZERO);

        // Camera
        let cam_node = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .node(&cam_node)
            .set_position(8.0, 6.0, 8.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node);

        Self {
            controls: OrbitControls::new(Vec3::new(8.0, 6.0, 8.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Shadow Spot | FPS: {:.2}", fps));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<ShadowSpotDemo>()
}
