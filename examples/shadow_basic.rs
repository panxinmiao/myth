use myth::prelude::*;
use myth::utils::fps_counter::FpsCounter;

struct ShadowBasicDemo {
    cube_node: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for ShadowBasicDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        // Cube with shadows — spawn + chainable node ops
        let cube_node = scene.spawn_box(
            1.5,
            1.5,
            1.5,
            MeshPhongMaterial::new(Vec4::new(0.9, 0.3, 0.2, 1.0)),
        );
        scene
            .node(&cube_node)
            .set_position(0.0, 3.2, 0.0)
            .set_shadows(true, true);

        // Ground plane
        let ground_node = scene.spawn_plane(
            30.0,
            30.0,
            MeshPhongMaterial::new(Vec4::new(0.8, 0.8, 0.85, 1.0)).with_side(Side::Double),
        );
        scene
            .node(&ground_node)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_cast_shadows(false)
            .set_receive_shadows(true);

        // Directional light with shadows
        let mut dir_light = Light::new_directional(Vec3::ONE, 5.0);
        dir_light.cast_shadows = true;
        if let Some(shadow) = dir_light.shadow.as_mut() {
            shadow.map_size = 2048;
        }
        let light_node = scene.add_light(dir_light);
        scene
            .node(&light_node)
            .set_position(8.0, 12.0, 6.0)
            .look_at(Vec3::ZERO);

        // Camera
        let cam_node = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .node(&cam_node)
            .set_position(8.0, 6.0, 8.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node);

        Self {
            cube_node,
            controls: OrbitControls::new(Vec3::new(8.0, 6.0, 8.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some(node) = scene.get_node_mut(self.cube_node) {
            node.transform.rotation *= Quat::from_rotation_y(1.2 * frame.dt);
        }

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Shadow Basic | FPS: {:.2}", fps));
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
        .run::<ShadowBasicDemo>()
}
