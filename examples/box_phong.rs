use glam::{Vec3, Vec4, Quat};
use three::app::{App, AppContext, AppHandler};
use three::resources::{Geometry, Material, Mesh, Texture};
use three::scene::{Camera, NodeHandle, light};
use three::OrbitControls;

/// Phong 材质立方体示例
struct PhongBox {
    cube_node_id: NodeHandle,
    controls: OrbitControls,
}

impl AppHandler for PhongBox {
    fn init(ctx: &mut AppContext) -> Self {
        // 1. 准备资源
        let geometry = Geometry::new_box(2.0, 2.0, 2.0);
        let texture = Texture::create_checkerboard(Some("checker"), 512, 512, 64);
        let mut mat = Material::new_phong(Vec4::new(1.0, 1.0, 1.0, 1.0));

        let tex_handle = ctx.assets.add_texture(texture);

        if let Some(phong) = mat.as_phong_mut() {
            phong.map.texture = Some(tex_handle);
        }
        
        let geo_handle = ctx.assets.add_geometry(geometry);
        let mat_handle = ctx.assets.add_material(mat);

        ctx.scenes.create_active();
        let scene = ctx.scenes.active_scene_mut().unwrap();

        // 2. 创建 Mesh 并加入场景
        let mesh = Mesh::new(geo_handle, mat_handle);
        let cube_node_id = scene.add_mesh(mesh);

        // 3. 添加灯光
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        scene.add_light(light);
        // 4. 设置相机
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

    fn update(&mut self, ctx: &mut AppContext) {
        let Some(scene) = ctx.scenes.active_scene_mut() else{
            return;
        };
        // 旋转立方体
        if let Some(node) = scene.get_node_mut(self.cube_node_id) {
            let rot_y = Quat::from_rotation_y(0.02);
            let rot_x = Quat::from_rotation_x(0.01);
            node.transform.rotation = node.transform.rotation * rot_y * rot_x;
        }

        // 轨道控制器
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new().run::<PhongBox>()
}