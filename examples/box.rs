use glam::{Vec3, Vec4, Quat};
use three::app::{App, AppContext, AppHandler};
use three::resources::{Geometry, Material, Mesh, Texture};
use three::scene::{Camera, NodeIndex};
use three::OrbitControls;

/// 带纹理的旋转立方体 + 轨道控制器
struct TexturedBox {
    cube_node_id: NodeIndex,
    controls: OrbitControls,
}

impl AppHandler for TexturedBox {
    fn init(ctx: &mut AppContext) -> Self {
        // 1. 准备资源
        let geometry = Geometry::new_box(2.0, 2.0, 2.0);
        let texture = Texture::create_checkerboard(Some("checker"), 512, 512, 64);
        let mut basic_mat = Material::new_basic(Vec4::new(1.0, 1.0, 1.0, 1.0));

        // 2. 将资源添加到 AssetServer
        let tex_handle = ctx.assets.add_texture(texture);

        if let Some(basic) = basic_mat.as_basic_mut() {
            basic.set_map(Some(tex_handle));
            basic.uniforms_mut().color = Vec4::new(1.0, 1.0, 1.0, 1.0);
        }
        
        let geo_handle = ctx.assets.add_geometry(geometry);
        let mat_handle = ctx.assets.add_material(basic_mat.into());

        // 3. 创建 Mesh 并加入场景
        let mesh = Mesh::new(geo_handle, mat_handle);
        let cube_node_id = ctx.scene.add_mesh(mesh);

        // 4. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1, 100.0);
        let cam_node_id = ctx.scene.add_camera(camera);
        
        if let Some(node) = ctx.scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 3.0, 10.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        
        ctx.scene.active_camera = Some(cam_node_id);

        let controls = OrbitControls::new(Vec3::new(0.0, 3.0, 10.0), Vec3::ZERO);

        Self { cube_node_id, controls }
    }

    fn update(&mut self, ctx: &mut AppContext) {
        // 旋转立方体
        if let Some(node) = ctx.scene.get_node_mut(self.cube_node_id) {
            let rot_y = Quat::from_rotation_y(0.02);
            let rot_x = Quat::from_rotation_x(0.01);
            node.transform.rotation = node.transform.rotation * rot_y * rot_x;
        }

        // 轨道控制器
        if let Some((transform, camera)) = ctx.scene.query_main_camera_bundle() {
            self.controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new().run::<TexturedBox>()
}