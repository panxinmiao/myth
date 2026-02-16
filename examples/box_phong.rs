use myth::prelude::*;

/// Phong Material Cube Example
struct PhongBox {
    cube_node_id: NodeHandle,
    controls: OrbitControls,
}

impl AppHandler for PhongBox {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let geometry = Geometry::new_box(2.0, 2.0, 2.0);
        let texture = Texture::create_checkerboard(Some("checker"), 512, 512, 64);
        let mut mat = Material::new_phong(Vec4::new(1.0, 1.0, 1.0, 1.0));

        let tex_handle = engine.assets.textures.add(texture);

        if let Some(phong) = mat.as_phong_mut() {
            phong.set_map(Some(tex_handle));
        }

        let geo_handle = engine.assets.geometries.add(geometry);
        let mat_handle = engine.assets.materials.add(mat);

        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        let mesh = Mesh::new(geo_handle, mat_handle);
        let cube_node_id = scene.add_mesh(mesh);

        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        scene.add_light(light);

        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 3.0, 10.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

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
