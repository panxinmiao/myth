use glam::{Vec3, Vec4};
use three::app::{App, AppContext, AppHandler};
use three::resources::{Geometry, Attribute, Material, Mesh, Texture};
use three::scene::Camera;

/// Hello Triangle 示例
/// 
/// 最简单的静态场景示例，演示新的 AppHandler 模式
struct HelloTriangle;

impl AppHandler for HelloTriangle {
    fn init(ctx: &mut AppContext) -> Self {
        // 1. 构建三角形几何体
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

        // 2. 准备材质和纹理
        let texture = Texture::create_solid_color(Some("red_tex"), [255, 0, 0, 255]);
        let mut basic_mat = Material::new_basic(Vec4::new(1.0, 1.0, 1.0, 1.0));
        
        // 3. 添加到 AssetServer
        let tex_handle = ctx.assets.add_texture(texture);

        if let Some(basic) = basic_mat.as_basic_mut() {
            basic.map.texture = Some(tex_handle);
        }

        let geo_handle = ctx.assets.add_geometry(geometry);
        let mat_handle = ctx.assets.add_material(basic_mat);

        ctx.scenes.create_active();
        let scene = ctx.scenes.active_scene_mut().unwrap();

        // 4. 创建 Mesh 并加入场景
        let mesh = Mesh::new(geo_handle, mat_handle);
        scene.add_mesh(mesh);
        // 5. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 3.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        
        scene.active_camera = Some(cam_node_id);

        Self
    }


}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new().run::<HelloTriangle>()
}