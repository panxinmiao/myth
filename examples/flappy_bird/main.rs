use std::collections::VecDeque;

use myth::{
    prelude::*,
    resources::{ImageDimension, PixelFormat},
};
use myth_resources::Key;
use rand::{Rng, rngs::StdRng};

/// Flappy Bird Example
///

const BIRD_IMAGE_DATA: &[u8] =
    include_bytes!("../assets/Flappy Bird Assets/Player/StyleBird1/Bird1-1.png");
const PIPE_IMAGE_DATA: &[u8] =
    include_bytes!("../assets/Flappy Bird Assets/Tiles/Style 1/PipeStyle1.png");

fn create_bird(engine: &mut Engine) -> Mesh {
    // 1. Create quad geometry
    let mut geometry = Geometry::new();
    geometry.set_attribute(
        "position",
        myth::Attribute::new_planar(
            &[
                [-0.5f32, 0.5, 0.0], // top-left
                [0.5, 0.5, 0.0],     // top-right
                [-0.5, -0.5, 0.0],   // bottom-left
                [0.5, -0.5, 0.0],    // bottom-right
            ],
            myth::VertexFormat::Float32x3,
        ),
    );

    geometry.set_attribute(
        "uv",
        myth::Attribute::new_planar(
            &[
                [0.0f32, 1.0], // top-left
                [0.25, 1.0],   // top-right
                [0.0, 0.0],    // bottom-left
                [0.25, 0.0],   // bottom-right
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
    let image_handle = engine.assets.images.add(Image::new(
        w,
        h,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(decoded.into_raw()),
    ));
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

fn create_pipe_cap(engine: &mut Engine) -> Mesh {
    // 1. Create quad geometry
    let mut geometry = Geometry::new();
    geometry.set_attribute(
        "position",
        myth::Attribute::new_planar(
            &[
                [-0.5f32, 0.5, 0.0], // top-left
                [0.5, 0.5, 0.0],     // top-right
                [-0.5, -0.5, 0.0],   // bottom-left
                [0.5, -0.5, 0.0],    // bottom-right
            ],
            myth::VertexFormat::Float32x3,
        ),
    );

    geometry.set_attribute(
        "uv",
        myth::Attribute::new_planar(
            &[
                [0.0f32, 1.0], // top-left
                [0.25, 1.0],   // top-right
                [0.0, 0.875],  // bottom-left
                [0.25, 0.875], // bottom-right
            ],
            myth::VertexFormat::Float32x2,
        ),
    );

    geometry.set_indices(&[0, 2, 1, 1, 2, 3]);

    // 2. Create unlit material with a solid color texture
    let image = image::load_from_memory(PIPE_IMAGE_DATA).expect("failed to decode PNG");
    let decoded = image.to_rgba8();
    let (w, h) = (decoded.width(), decoded.height());
    let image_handle = engine.assets.images.add(Image::new(
        w,
        h,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(decoded.into_raw()),
    ));
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

fn create_pipe(engine: &mut Engine) -> Mesh {
    // 1. Create quad geometry
    let mut geometry = Geometry::new();
    geometry.set_attribute(
        "position",
        myth::Attribute::new_planar(
            &[
                [-0.5f32, 0.5, 0.0], // top-left
                [0.5, 0.5, 0.0],     // top-right
                [-0.5, -0.5, 0.0],   // bottom-left
                [0.5, -0.5, 0.0],    // bottom-right
            ],
            myth::VertexFormat::Float32x3,
        ),
    );

    geometry.set_attribute(
        "uv",
        myth::Attribute::new_planar(
            &[
                [0.0f32, 0.875], // top-left
                [0.25, 0.875],   // top-right
                [0.0, 0.75],     // bottom-left
                [0.25, 0.75],    // bottom-right
            ],
            myth::VertexFormat::Float32x2,
        ),
    );

    geometry.set_indices(&[0, 2, 1, 1, 2, 3]);

    // 2. Create unlit material with a solid color texture
    let image = image::load_from_memory(PIPE_IMAGE_DATA).expect("failed to decode PNG");
    let decoded = image.to_rgba8();
    let (w, h) = (decoded.width(), decoded.height());
    let image_handle = engine.assets.images.add(Image::new(
        w,
        h,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(decoded.into_raw()),
    ));
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
    top_pipe: Vec<NodeHandle>,
    bottom_pipe: Vec<NodeHandle>,
    width: f32,
    gap_y: f32,
    x_position: f32,
}

const PIPE_SPAWN_INTERVAL: f32 = 1.0; // seconds
const BIRD_SPAWN_POINT: Vec3 = Vec3::new(-1.0, 0.0, 0.0);
const BIRD_SCALE: f32 = 0.25;
const PIPE_GAP: f32 = 0.5;
const PIPE_SCALE_X: f32 = 0.5;
const PIPE_SCALE_Y: f32 = 0.5;
const PIPE_STARTING_X_POSITION: f32 = 2.0;
const PIPE_GAP_CENTER: f32 = 0.0; // y-center of the gap between pipes
struct FlappyBird {
    bird_node: NodeHandle,
    bird_velocity: Vec2,
    pipe_cap_mesh: Mesh,
    pipe_mesh: Mesh,
    pipes: Vec<PipeDuo>,
    pipe_spawn_timer: f32,
}

impl AppHandler for FlappyBird {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let bird_mesh = create_bird(engine);
        let pipe_mesh = create_pipe(engine);
        let pipe_cap_mesh = create_pipe_cap(engine);
        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        let bird_node = scene.add_mesh(bird_mesh);
        if let Some(node) = scene.get_node_mut(bird_node) {
            node.transform.position = BIRD_SPAWN_POINT;
            node.transform.scale = Vec3::new(BIRD_SCALE, BIRD_SCALE, 1.0);
        }
        // 5. Set up camera
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 10.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        Self {
            bird_node,
            bird_velocity: Vec2::ZERO,
            pipe_cap_mesh,
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
            println!(
                "Flap! Position: {:?}",
                engine
                    .scene_manager
                    .active_scene()
                    .unwrap()
                    .get_node(self.bird_node)
                    .unwrap()
                    .transform
                    .position
            );
        }

        // Update bird position
        let mut bird_pos = BIRD_SPAWN_POINT;
        if let Some(node) = engine
            .scene_manager
            .active_scene_mut()
            .unwrap()
            .get_node_mut(self.bird_node)
        {
            node.transform.position.y += self.bird_velocity.y * frame.dt;
            bird_pos = node.transform.position;
            if node.transform.position.y < -2.0 || node.transform.position.y > 2.0 {
                self.game_over(engine);
                return;
            }
        }

        // Check collision with pipes
        if self.collide_with_pipes(bird_pos) {
            self.game_over(engine);
            return;
        }

        // Move pipes leftward and check for off-screen
        let mut to_remove = vec![];
        for pipe_duo in &mut self.pipes {
            pipe_duo.x_position -= 2.0 * frame.dt;
            for top in &pipe_duo.top_pipe {
                if let Some(node) = engine
                    .scene_manager
                    .active_scene_mut()
                    .unwrap()
                    .get_node_mut(*top)
                {
                    node.transform.position.x = pipe_duo.x_position;
                }
            }
            for bottom in &pipe_duo.bottom_pipe {
                if let Some(node) = engine
                    .scene_manager
                    .active_scene_mut()
                    .unwrap()
                    .get_node_mut(*bottom)
                {
                    node.transform.position.x = pipe_duo.x_position;
                }
            }
            if pipe_duo.x_position < -3.0 {
                for top in &pipe_duo.top_pipe {
                    engine
                        .scene_manager
                        .active_scene_mut()
                        .unwrap()
                        .remove_node(*top);
                }
                for bottom in &pipe_duo.bottom_pipe {
                    engine
                        .scene_manager
                        .active_scene_mut()
                        .unwrap()
                        .remove_node(*bottom);
                }
                to_remove.push(pipe_duo.clone());
            }
        }
        // Remove off-screen pipes from tracking
        self.pipes.retain(|duo| !to_remove.contains(duo));

        // Spawn pipes at intervals
        self.pipe_spawn_timer += frame.dt;
        if self.pipe_spawn_timer >= PIPE_SPAWN_INTERVAL {
            self.pipe_spawn_timer = 0.0;
            let gap_y = PIPE_GAP_CENTER;
            let mut rng = rand::rng();
            let top_pipe_amount = rng.next_u32() % 3;
            let bottom_pipe_amount = rng.next_u32() % 3;

            let mut top_pipe_nodes = vec![];

            let mut top_pipe_height = -2.;

            for i in 0..top_pipe_amount {
                let top_pipe_mesh = self.pipe_mesh.clone();
                let top_pipe_node = engine
                    .scene_manager
                    .active_scene_mut()
                    .unwrap()
                    .add_mesh(top_pipe_mesh);
                if let Some(node) = engine
                    .scene_manager
                    .active_scene_mut()
                    .unwrap()
                    .get_node_mut(top_pipe_node)
                {
                    node.transform.position =
                        Vec3::new(PIPE_STARTING_X_POSITION, top_pipe_height, 0.0);
                    node.transform.scale = Vec3::new(PIPE_SCALE_X, PIPE_SCALE_Y, 1.0);
                }
                top_pipe_nodes.push(top_pipe_node);
                top_pipe_height += PIPE_SCALE_Y;
            }

            let top_pipe_cap_mesh = self.pipe_cap_mesh.clone();
            let top_pipe_cap_node = engine
                .scene_manager
                .active_scene_mut()
                .unwrap()
                .add_mesh(top_pipe_cap_mesh);
            if let Some(node) = engine
                .scene_manager
                .active_scene_mut()
                .unwrap()
                .get_node_mut(top_pipe_cap_node)
            {
                node.transform.position = Vec3::new(PIPE_STARTING_X_POSITION, top_pipe_height, 0.0);
                node.transform.scale = Vec3::new(PIPE_SCALE_X, PIPE_SCALE_Y, 1.0);
            }

            top_pipe_nodes.push(top_pipe_cap_node);

            let mut bottom_pipe_nodes = vec![];
            let mut bottom_pipe_height = 2.;

            for i in 0..bottom_pipe_amount {
                let bottom_pipe_mesh = self.pipe_mesh.clone();
                let bottom_pipe_node = engine
                    .scene_manager
                    .active_scene_mut()
                    .unwrap()
                    .add_mesh(bottom_pipe_mesh);
                if let Some(node) = engine
                    .scene_manager
                    .active_scene_mut()
                    .unwrap()
                    .get_node_mut(bottom_pipe_node)
                {
                    node.transform.position =
                        Vec3::new(PIPE_STARTING_X_POSITION, bottom_pipe_height, 0.0);
                    node.transform.scale = Vec3::new(PIPE_SCALE_X, -PIPE_SCALE_Y, 1.0);
                }
                bottom_pipe_nodes.push(bottom_pipe_node);
                bottom_pipe_height -= PIPE_SCALE_Y;
            }

            let bottom_pipe_cap_mesh = self.pipe_cap_mesh.clone();
            let bottom_pipe_cap_node = engine
                .scene_manager
                .active_scene_mut()
                .unwrap()
                .add_mesh(bottom_pipe_cap_mesh);
            if let Some(node) = engine
                .scene_manager
                .active_scene_mut()
                .unwrap()
                .get_node_mut(bottom_pipe_cap_node)
            {
                node.transform.position =
                    Vec3::new(PIPE_STARTING_X_POSITION, bottom_pipe_height, 0.0);
                node.transform.scale = Vec3::new(PIPE_SCALE_X, -PIPE_SCALE_Y, 1.0);
            }

            bottom_pipe_nodes.push(bottom_pipe_cap_node);

            self.pipes.push(PipeDuo {
                top_pipe: top_pipe_nodes,
                bottom_pipe: bottom_pipe_nodes,
                x_position: PIPE_STARTING_X_POSITION,
                width: PIPE_SCALE_X,
                gap_y,
            });
            println!("Spawned new pipe!");
        }
    }
}

impl FlappyBird {
    fn collide_with_pipes(&self, bird_pos: Vec3) -> bool {
        // Collision detection: bird vs pipes (AABB)
        /*
        let bird_half_w = BIRD_SCALE * 0.5;
        let bird_half_h = BIRD_SCALE * 0.5;
        let pipe_half_w = PIPE_SCALE_X * 0.5;
        let pipe_half_h = PIPE_SCALE_Y * 0.5;
        for pipe_duo in &self.pipes {
            // Top pipe
            let top_y = pipe_duo.gap_y - PIPE_GAP * 0.5 - pipe_half_h;
            if (bird_pos.x - pipe_duo.x_position).abs() < bird_half_w + pipe_half_w
                && (bird_pos.y - top_y).abs() < bird_half_h + pipe_half_h
            {
                return true;
            }
            // Bottom pipe
            let bottom_y = pipe_duo.gap_y + PIPE_GAP * 0.5 + pipe_half_h;
            if (bird_pos.x - pipe_duo.x_position).abs() < bird_half_w + pipe_half_w
                && (bird_pos.y - bottom_y).abs() < bird_half_h + pipe_half_h
            {
                return true;
            }
        }
        */
        false
    }

    fn game_over(&mut self, engine: &mut Engine) {
        println!("Game Over!");
        // Reset bird
        if let Some(node) = engine
            .scene_manager
            .active_scene_mut()
            .unwrap()
            .get_node_mut(self.bird_node)
        {
            node.transform.position = BIRD_SPAWN_POINT;
        }
        self.bird_velocity = Vec2::ZERO;
        // Remove all pipes from scene
        for pipe_duo in &self.pipes {
            for top in &pipe_duo.top_pipe {
                engine
                    .scene_manager
                    .active_scene_mut()
                    .unwrap()
                    .remove_node(*top);
            }
            for bottom in &pipe_duo.bottom_pipe {
                engine
                    .scene_manager
                    .active_scene_mut()
                    .unwrap()
                    .remove_node(*bottom);
            }
        }
        self.pipes.clear();
        self.pipe_spawn_timer = 0.0;
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new().run::<FlappyBird>()
}
