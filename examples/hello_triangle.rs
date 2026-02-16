use myth::prelude::*;

/// Hello Triangle Example
///
struct HelloTriangle;

impl AppHandler for HelloTriangle {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        // 1. Create triangle geometry
        let mut geometry = Geometry::new();
        geometry.set_attribute(
            "position",
            myth::Attribute::new_planar(
                &[[0.0f32, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]],
                wgpu::VertexFormat::Float32x3,
            ),
        );

        geometry.set_attribute(
            "uv",
            myth::Attribute::new_planar(
                &[[0.5f32, 1.0], [0.0, 0.0], [1.0, 0.0]],
                wgpu::VertexFormat::Float32x2,
            ),
        );

        // 2. Create basic material with a solid color texture
        let texture = Texture::create_solid_color(Some("red_tex"), [255, 0, 0, 255]);
        let mut basic_mat = Material::new_basic(Vec4::new(1.0, 1.0, 1.0, 1.0));

        // 3. Add resources to AssetServer
        let tex_handle = engine.assets.textures.add(texture);

        if let Some(basic) = basic_mat.as_basic_mut() {
            basic.set_map(Some(tex_handle));
        }

        let geo_handle = engine.assets.geometries.add(geometry);
        let mat_handle = engine.assets.materials.add(basic_mat);

        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();
        // 4. Create Mesh and add to scene
        let mesh = Mesh::new(geo_handle, mat_handle);
        scene.add_mesh(mesh);
        // 5. Set up camera
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

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new().run::<HelloTriangle>()
}
