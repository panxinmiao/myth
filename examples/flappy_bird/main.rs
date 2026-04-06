use myth::{prelude::*, resources::{ImageDimension, PixelFormat}};

/// Hello Triangle Example
///
struct HelloTriangle;

const IMAGE_DATA: &[u8] = include_bytes!("../assets/Flappy Bird Assets/Player/StyleBird1/Bird1-1.png");

impl AppHandler for HelloTriangle {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        // 1. Create quad geometry
        let mut geometry = Geometry::new();
        geometry.set_attribute(
            "position",
            myth::Attribute::new_planar(
                &[
                    [-0.5f32,  0.5, 0.0], // top-left
                    [ 0.5,     0.5, 0.0], // top-right
                    [-0.5,    -0.5, 0.0], // bottom-left
                    [ 0.5,    -0.5, 0.0], // bottom-right
                ],
                myth::VertexFormat::Float32x3,
            ),
        );

        geometry.set_attribute(
            "uv",
            myth::Attribute::new_planar(
                &[
                    [0.0f32, 1.0], // top-left
                    [1.0,    1.0], // top-right
                    [0.0,    0.0], // bottom-left
                    [1.0,    0.0], // bottom-right
                ],
                myth::VertexFormat::Float32x2,
            ),
        );

        geometry.set_indices(&[0, 2, 1, 1, 2, 3]);

        // 2. Create unlit material with a solid color texture
        let image = image::load_from_memory(IMAGE_DATA).expect("failed to decode PNG");
         // flip image since its upside down by default
        let image = image.flipv();
        let decoded = image.to_rgba8();
        let (w, h) = (decoded.width(), decoded.height());
        let image_handle = engine
            .assets
            .images
            .add(Image::new(w, h, 1, ImageDimension::D2, PixelFormat::Rgba8Unorm, Some(decoded.into_raw())));
        let texture = Texture::new_2d(Some("red_tex"), image_handle);
        let mut unlit_mat = Material::new_unlit(Vec4::new(1.0, 1.0, 1.0, 1.0));

        // 3. Add resources to AssetServer
        let tex_handle = engine.assets.textures.add(texture);

        if let Some(unlit) = unlit_mat.as_unlit_mut() {
            unlit.set_map(Some(tex_handle));
        }

        let geo_handle = engine.assets.geometries.add(geometry);
        let mat_handle = engine.assets.materials.add(unlit_mat);

        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();
        // 4. Create Mesh and add to scene
        let mesh = Mesh::new(geo_handle, mat_handle);
        let mesh2 = mesh.clone();
        scene.add_mesh(mesh);
        let mesh2_node = scene.add_mesh(mesh2);
        if let Some(node) = scene.get_node_mut(mesh2_node) {
            node.transform.position = Vec3::new(1.0, 1.0, 0.0);
            node.transform.scale = Vec3::new(1.0, h as f32 / w as f32, 1.0);
        }
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