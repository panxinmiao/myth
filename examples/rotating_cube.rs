use glam::{Vec3, Vec4, Quat};

use three::app::App;
use three::resources::{Geometry, Material, Mesh};
use three::scene::{Camera};

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
    let cube_node_id = app.scene.add_mesh(mesh);

    // === 3. 设置相机 ===
    let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1, 100.0);
    
    // 添加相机到场景
    let camera_node_id = app.scene.add_camera(camera);
    
    // 通过节点设置相机位置
    if let Some(cam_node) = app.scene.get_node_mut(camera_node_id) {
        cam_node.transform.position = Vec3::new(0.0, 3.0, 20.0);
        cam_node.transform.look_at(Vec3::ZERO, Vec3::Y);
    }
    
    app.scene.active_camera = Some(camera_node_id);


    // === 5. 设置更新回调 - 让立方体旋转 ===
    app.set_update_fn(move |ctx| {
        // 通过 node_id 获取立方体节点
        if let Some(cube_node) = ctx.scene.get_node_mut(cube_node_id) {
            // 绕 Y 轴和 X 轴旋转
            let rotation_y = Quat::from_rotation_y(ctx.time * 0.5); // 每秒 0.5 弧度
            let rotation_x = Quat::from_rotation_x(ctx.time * 0.3);
            cube_node.transform.rotation = rotation_y * rotation_x;
        }
    });

    // === 6. 运行应用 ===
    app.run()?;
    Ok(())
}
