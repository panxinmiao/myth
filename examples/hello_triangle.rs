use glam::{Vec3, Vec4};
use three::app::App;
use three::resources::{Geometry, Attribute, Material, Mesh, Texture};
use three::scene::Camera;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    // 1. 初始化 App
    let mut app = App::new();

    // 2. 构建三角形几何体
    let mut geometry = Geometry::new();
    geometry.set_attribute("position", Attribute::new_planar(&[
        [0.0f32, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
    ], wgpu::VertexFormat::Float32x3));
    
    geometry.set_attribute("uv", Attribute::new_planar(&[
        [0.5f32, 1.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ], wgpu::VertexFormat::Float32x2));

    // 3. 准备材质和纹理
    let texture = Texture::create_solid_color("red_tex", [255, 0, 0, 255]);
    let mut basic_mat = Material::new_basic(Vec4::new(1.0, 1.0, 1.0, 1.0));
    
    // 4. 添加到 AssetServer
    let tex_handle = app.assets.add_texture(texture);

    if let Some(basic) = basic_mat.as_basic_mut() {
        basic.map = Some(tex_handle);
    }

    let geo_handle = app.assets.add_geometry(geometry);
    let mat_handle = app.assets.add_material(basic_mat.into());

    // 5. 创建 Mesh 并加入场景
    let mesh = Mesh::new(geo_handle, mat_handle);
    app.scene.add_mesh(mesh);

    // 6. 设置相机
    let camera = Camera::new_perspective(
        45.0, 
        1280.0 / 720.0, 
        0.1, 
        100.0
    );
    
    // 挂载相机到 Node
    let cam_node_id = app.scene.add_camera(camera);

    // 设置位置
    if let Some(node) = app.scene.get_node_mut(cam_node_id) {
        node.position = Vec3::new(0.0, 0.0, 3.0);
        node.look_at(Vec3::ZERO, Vec3::Y);
    }
    
    // 激活相机
    app.active_camera = Some(cam_node_id);

    // 7. 运行 (不需要 update_fn，因为是静态场景)
    app.run()
}