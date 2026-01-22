use glam::{Vec3, Vec4, Quat};
use three::app::{App, AppContext, AppHandler};
use three::resources::{Geometry, Material, Mesh};
use three::scene::{Camera, NodeHandle};

/// 旋转立方体示例
/// 
/// 演示带动画更新的 AppHandler 模式
struct RotatingCube {
    cube_node_id: NodeHandle,
}

impl AppHandler for RotatingCube {
    fn init(ctx: &mut AppContext) -> Self {
        // 1. 创建并添加几何体和材质到资产服务器
        let geometry = Geometry::new_box(2.0, 2.0, 2.0);
        let geo_handle = ctx.assets.add_geometry(geometry);

        let material = Material::new_basic(Vec4::new(0.8, 0.3, 0.3, 1.0));
        let mat_handle = ctx.assets.add_material(material);

        // 2. 创建 Mesh 并添加到场景
        let mesh = Mesh::new(geo_handle, mat_handle);
        let cube_node_id = ctx.scene.add_mesh(mesh);

        // 3. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let camera_node_id = ctx.scene.add_camera(camera);
        
        if let Some(cam_node) = ctx.scene.get_node_mut(camera_node_id) {
            cam_node.transform.position = Vec3::new(0.0, 3.0, 20.0);
            cam_node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        
        ctx.scene.active_camera = Some(camera_node_id);

        Self { cube_node_id }
    }

    fn update(&mut self, ctx: &mut AppContext) {
        // 让立方体旋转
        if let Some(cube_node) = ctx.scene.get_node_mut(self.cube_node_id) {
            let rotation_y = Quat::from_rotation_y(ctx.time * 0.5);
            let rotation_x = Quat::from_rotation_x(ctx.time * 0.3);
            cube_node.transform.rotation = rotation_y * rotation_x;
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new().run::<RotatingCube>()
}
