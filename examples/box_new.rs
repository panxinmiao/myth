use glam::{Vec3, Vec4};

use three::app::App;
use three::resources::{Geometry, Material, Mesh};
use three::scene::{Camera, Light};

fn main() -> anyhow::Result<()> {
    // 初始化日志
    env_logger::init();

    // 创建应用
    let mut app = App::new();

    // === 1. 创建并添加几何体和材质到资产服务器 ===
    let geometry = Geometry::new_box(2.0, 2.0, 2.0);
    let geo_handle = app.assets.add_geometry(geometry);

    let material = Material::new_basic(Vec4::new(0.8, 0.3, 0.3, 1.0));
    let mat_handle = app.assets.add_material(material.into());

    // === 2. 创建 Mesh 并添加到场景 ===
    let mesh = Mesh::new(geo_handle, mat_handle);
    app.scene.add_mesh(mesh);

    // === 3. 设置相机 ===
    let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1, 100.0);
    
    // 添加相机到场景
    let camera_node_id = app.scene.add_camera(camera);
    
    // 通过节点设置相机位置
    if let Some(cam_node) = app.scene.get_node_mut(camera_node_id) {
        cam_node.position = Vec3::new(0.0, 3.0, 10.0);
        // 计算朝向原点的旋转
        let forward = (Vec3::ZERO - cam_node.position).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        cam_node.rotation = glam::Quat::from_mat3(&glam::Mat3::from_cols(
            right,
            up,
            -forward,
        ));
    }
    
    app.active_camera = Some(camera_node_id);

    // === 4. 添加光源 ===
    let light = Light::new_directional(
        Vec3::new(0.0, -1.0, -1.0).normalize(), // 方向
        Vec3::new(1.0, 1.0, 1.0),               // 白光
        1.5                                      // 强度
    );
    app.scene.add_light(light);

    // === 5. 运行应用 ===
    app.run()
}
