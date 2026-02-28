use myth::prelude::*;

/// Basic Rotating Cube Example
///
struct RotatingCube {
    cube_node_id: NodeHandle,
}

impl AppHandler for RotatingCube {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        // One-liner: geometry + material auto-registered
        let cube_node_id = scene.spawn_box(
            2.0,
            2.0,
            2.0,
            Material::new_unlit(Vec4::new(0.8, 0.3, 0.3, 1.0)),
        );

        let camera_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&camera_node_id)
            .set_position(0.0, 3.0, 20.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(camera_node_id);

        Self { cube_node_id }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };
        if let Some(cube_node) = scene.get_node_mut(self.cube_node_id) {
            let rotation_y = Quat::from_rotation_y(frame.time * 0.5);
            let rotation_x = Quat::from_rotation_x(frame.time * 0.3);
            cube_node.transform.rotation = rotation_y * rotation_x;
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new().run::<RotatingCube>()
}
