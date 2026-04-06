use myth::{prelude::*, resources::{ImageDimension, PixelFormat}};
use myth_resources::Key;

/// Flappy Bird Example
///

const BIRD_IMAGE_DATA: &[u8] = include_bytes!("../assets/Flappy Bird Assets/Player/StyleBird1/Bird1-1.png");
const PIPE_IMAGE_DATA: &[u8] = include_bytes!("../assets/Flappy Bird Assets/Tiles/Style 1/PipeStyle1.png");

fn create_bird(engine: &mut Engine) -> Mesh {
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
                    [0.25,    1.0], // top-right
                    [0.0,    0.0], // bottom-left
                    [0.25,    0.0], // bottom-right
                ],
                myth::VertexFormat::Float32x2,
            ),
        );

        geometry.set_indices(&[0, 2, 1, 1, 2, 3]);

        // 2. Create unlit material with a solid color texture
        let image = image::load_from_memory(BIRD_IMAGE_DATA).expect("failed to decode PNG");
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

        Mesh::new(geo_handle, mat_handle)
}

fn create_pipe(engine: &mut Engine, turn_up: bool) -> Mesh {
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
                    [0.25,    1.0], // top-right
                    [0.0,    0.75], // bottom-left
                    [0.25,    0.75], // bottom-right
                ],
                myth::VertexFormat::Float32x2,
            ),
        );

        geometry.set_indices(&[0, 2, 1, 1, 2, 3]);

        // 2. Create unlit material with a solid color texture
        let mut image = image::load_from_memory(PIPE_IMAGE_DATA).expect("failed to decode PNG");
        if turn_up {
            // flip image since its upside down by default
            image = image.flipv();
        }
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

        Mesh::new(geo_handle, mat_handle)
}

const PIPE_SPAWN_INTERVAL: f32 = 2.0; // seconds
const BIRD_SPAWN_POINT : Vec3 = Vec3::new(-1.0, 0.0, 0.0);
struct FlappyBird {
    bird_node: NodeHandle,
    bird_velocity: Vec2,
    pipes: Vec<NodeHandle>,
    pipe_spawn_timer: f32,
}
impl AppHandler for FlappyBird {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {

        let bird_mesh = create_bird(engine);
        let pipe_mesh = create_pipe(engine, true);
        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();        

        let bird_node = scene.add_mesh(bird_mesh);
        if let Some(node) = scene.get_node_mut(bird_node) {
            node.transform.position = BIRD_SPAWN_POINT;
            node.transform.scale = Vec3::new(0.1, 0.1, 0.1);
        }
        let pipe_node = scene.add_mesh(pipe_mesh);
        if let Some(node) = scene.get_node_mut(pipe_node) {
            node.transform.position = Vec3::new(2.0, -0.5, 0.0);
            node.transform.scale = Vec3::new(0.5, 0.5, 1.0);
        }
        // 5. Set up camera
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 3.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        Self {
            bird_node,
            bird_velocity: Vec2::ZERO,
            pipes: vec![pipe_node],
            pipe_spawn_timer: 0.0,
        }

    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        // Apply gravity
        self.bird_velocity.y -= 5.0 * frame.dt;

        // Detect spacebar press (flap)
        if engine.input.get_key_down(Key::Space) {
            // apply upward velocity to bird
            self.bird_velocity.y = 4.0;
        }

        // Update bird position
        if let Some(node) = engine.scene_manager.active_scene_mut().unwrap().get_node_mut(self.bird_node) {
            node.transform.position.y += self.bird_velocity.y * frame.dt;
            if node.transform.position.y < -1.9 || node.transform.position.y > 1.9 {
                println!("Game Over!");
                node.transform.position = BIRD_SPAWN_POINT;
                self.bird_velocity = Vec2::ZERO;
            }
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new().run::<FlappyBird>()
}