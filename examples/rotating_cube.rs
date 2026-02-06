use std::sync::Arc;

use myth::prelude::*;
use winit::window::Window;

/// Basic Rotating Cube Example
///
struct RotatingCube {
    cube_node_id: NodeHandle,
}

impl AppHandler for RotatingCube {
    fn init(engine: &mut MythEngine, _window: &Arc<Window>) -> Self {
        let geometry = Geometry::new_box(2.0, 2.0, 2.0);
        let geo_handle = engine.assets.geometries.add(geometry);

        let material = Material::new_basic(Vec4::new(0.8, 0.3, 0.3, 1.0));
        let mat_handle = engine.assets.materials.add(material);

        let mesh = Mesh::new(geo_handle, mat_handle);
        let scene = engine.scene_manager.create_active();
        let cube_node_id = scene.add_mesh(mesh);

        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let camera_node_id = scene.add_camera(camera);

        if let Some(cam_node) = scene.get_node_mut(camera_node_id) {
            cam_node.transform.position = Vec3::new(0.0, 3.0, 20.0);
            cam_node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(camera_node_id);

        Self { cube_node_id }
    }

    fn update(&mut self, engine: &mut MythEngine, _window: &Arc<Window>, frame: &FrameState) {
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
