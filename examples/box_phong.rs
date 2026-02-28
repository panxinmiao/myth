use myth::prelude::*;

/// Phong Material Cube Example
struct PhongBox {
    cube_node_id: NodeHandle,
    controls: OrbitControls,
}

impl AppHandler for PhongBox {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let tex_handle =
            engine
                .assets
                .textures
                .add(Texture::create_checkerboard(Some("checker"), 512, 512, 64));

        let scene = engine.scene_manager.create_active();

        // Builder pattern: create material inline with chaining
        let mat = PhongMaterial::new(Vec4::ONE).with_map(tex_handle);
        let cube_node_id = scene.spawn_box(2.0, 2.0, 2.0, mat);

        scene.add_light(Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0));

        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node_id)
            .set_position(0.0, 3.0, 10.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);

        Self {
            cube_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 3.0, 10.0), Vec3::ZERO),
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some(node) = scene.get_node_mut(self.cube_node_id) {
            let rot_y = Quat::from_rotation_y(0.02);
            let rot_x = Quat::from_rotation_x(0.01);
            node.transform.rotation = node.transform.rotation * rot_y * rot_x;
        }

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov, frame.dt);
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new().run::<PhongBox>()
}
