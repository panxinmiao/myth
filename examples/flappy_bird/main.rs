use std::collections::VecDeque;

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

fn create_pipe(engine: &mut Engine) -> Mesh {
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
        let image = image::load_from_memory(PIPE_IMAGE_DATA).expect("failed to decode PNG");
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

        // Set face culling to double-sided since pipes can be seen from both sides (flip texture vertically for bottom pipe)
        if let Some(unlit) = unlit_mat.as_unlit_mut() {
            unlit.set_map(Some(tex_handle));
            unlit.set_side(myth::resources::Side::Double);
        }

        let geo_handle = engine.assets.geometries.add(geometry);
        let mat_handle = engine.assets.materials.add(unlit_mat);

        Mesh::new(geo_handle, mat_handle)
}

#[derive(PartialEq, Clone)]
struct PipeDuo {
    top_pipe: NodeHandle,
    bottom_pipe: NodeHandle,
    width: f32,
    gap_y: f32,
    x_position: f32,
}

const PIPE_SPAWN_INTERVAL: f32 = 1.0; // seconds
const BIRD_SPAWN_POINT : Vec3 = Vec3::new(-1.0, 0.0, 0.0);
const PIPE_GAP: f32 = 1.0;
const PIPE_STARTING_X_POSITION: f32 = 2.0;
struct FlappyBird {
    bird_node: NodeHandle,
    bird_velocity: Vec2,
    pipe_mesh : Mesh,
    pipes: Vec<PipeDuo>,
    pipe_spawn_timer: f32,
}

impl AppHandler for FlappyBird {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {

        let bird_mesh = create_bird(engine);
        let pipe_mesh = create_pipe(engine);
        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();        

        let bird_node = scene.add_mesh(bird_mesh);
        if let Some(node) = scene.get_node_mut(bird_node) {
            node.transform.position = BIRD_SPAWN_POINT;
            node.transform.scale = Vec3::new(0.1, 0.1, 0.1);
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
            pipe_mesh,
            pipes: vec![],
            pipe_spawn_timer: PIPE_SPAWN_INTERVAL, // spawn first pipe immediately
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

        // Move pipes leftward and check for off-screen
        let mut to_remove = vec![];
        for pipe_duo in &mut self.pipes {
            pipe_duo.x_position -= 2.0 * frame.dt;
            if let Some(node) = engine.scene_manager.active_scene_mut().unwrap().get_node_mut(pipe_duo.top_pipe) {
                node.transform.position.x = pipe_duo.x_position;
            }
            if let Some(node) = engine.scene_manager.active_scene_mut().unwrap().get_node_mut(pipe_duo.bottom_pipe) {
                node.transform.position.x = pipe_duo.x_position;
            }
            if pipe_duo.x_position < -3.0 {
                    engine.scene_manager.active_scene_mut().unwrap().remove_node(pipe_duo.top_pipe);
                    engine.scene_manager.active_scene_mut().unwrap().remove_node(pipe_duo.bottom_pipe);
                    to_remove.push(pipe_duo.clone());
            }
        }
        // Remove off-screen pipes from tracking
        self.pipes.retain(|duo| !to_remove.contains(duo));


        // Spawn pipes at intervals
        self.pipe_spawn_timer += frame.dt;
        if self.pipe_spawn_timer >= PIPE_SPAWN_INTERVAL {
            self.pipe_spawn_timer = 0.0;
            let top_pipe_mesh = self.pipe_mesh.clone();
            let top_pipe_node = engine.scene_manager.active_scene_mut().unwrap().add_mesh(top_pipe_mesh);
            if let Some(node) = engine.scene_manager.active_scene_mut().unwrap().get_node_mut(top_pipe_node) {
                node.transform.position = Vec3::new(PIPE_STARTING_X_POSITION, -0.5, 0.0);
                node.transform.scale = Vec3::new(0.5, 0.5, 1.0);
            }
            let bottom_pipe_mesh = self.pipe_mesh.clone();
            let bottom_pipe_node = engine.scene_manager.active_scene_mut().unwrap().add_mesh(bottom_pipe_mesh);
            if let Some(node) = engine.scene_manager.active_scene_mut().unwrap().get_node_mut(bottom_pipe_node) {
                node.transform.position = Vec3::new(PIPE_STARTING_X_POSITION, 0.5, 0.0);
                node.transform.scale = Vec3::new(0.5, -0.5, 1.0);
            }
            self.pipes.push(PipeDuo {
                top_pipe: top_pipe_node,
                bottom_pipe: bottom_pipe_node,
                x_position: PIPE_STARTING_X_POSITION,
                width: 0.5,
                gap_y: -0.5,
            });
            println!("Spawned new pipe!");
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new().run::<FlappyBird>()
}