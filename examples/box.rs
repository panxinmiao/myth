use std::sync::Arc;

use myth::prelude::*;
use winit::window::Window;

/// 带纹理的旋转立方体 + 轨道控制器
struct TexturedBox {
    controls: OrbitControls,
}

impl AppHandler for TexturedBox {
    fn init(engine: &mut MythEngine, _window: &Arc<Window>) -> Self {
        // 1. 准备资源
        let geometry = Geometry::new_box(2.0, 2.0, 2.0);
        let texture = Texture::create_checkerboard(Some("checker"), 512, 512, 64);
        let basic_mat = MeshBasicMaterial::new(Vec4::new(1.0, 1.0, 1.0, 1.0));

        // 2. 将资源添加到 AssetServer
        let tex_handle = engine.assets.textures.add(texture);

        basic_mat.set_map(Some(tex_handle));

        let geo_handle = engine.assets.geometries.add(geometry);
        let mat_handle = engine.assets.materials.add(basic_mat);

        let scene = engine.scene_manager.create_active();

        // 3. 创建 Mesh 并加入场景
        let mesh = Mesh::new(geo_handle, mat_handle);
        let cube_node_id = scene.add_mesh(mesh);
        // 4. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 3.0, 10.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        let controls = OrbitControls::new(Vec3::new(0.0, 3.0, 10.0), Vec3::ZERO);

        scene.on_update(move |scene, _input, _dt| {
            // 旋转立方体
            if let Some(node) = scene.get_node_mut(cube_node_id) {
                let rot_y = Quat::from_rotation_y(0.02);
                let rot_x = Quat::from_rotation_x(0.01);
                node.transform.rotation = node.transform.rotation * rot_y * rot_x;
            }
        });

        Self { controls }
    }

    fn update(&mut self, engine: &mut MythEngine, _window: &Arc<Window>, frame: &FrameState) {
        let scene = engine.scene_manager.active_scene_mut().unwrap();
        // 轨道控制器
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov, frame.dt);
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new().run::<TexturedBox>()
}
