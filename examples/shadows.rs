use std::env;
use std::path::Path;
use std::sync::Arc;

use myth::prelude::*;
use myth::utils::FpsCounter;
use winit::window::Window;

/// Skinning Animation Example
///
struct SkinningDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for SkinningDemo {
    fn init(engine: &mut Engine, _window: &Arc<Window>) -> Self {
        // === 1. Parse command line arguments for model path ===
        let args: Vec<String> = env::args().collect();

        // Provide a default model path if none is given
        let default_path = "examples/assets/Michelle.glb";

        let gltf_path_str = if args.len() > 1 {
            &args[1]
        } else {
            println!("Tip: You can pass a model path as an argument.");
            println!("Usage: cargo run --example shadows -- <path_to_gltf>");
            println!("No path provided, loading default: {}", default_path);
            default_path
        };

        let gltf_path = Path::new(gltf_path_str);

        // === 2. Load environment map ===
        let map_path = "examples/assets/royal_esplanade_2k.hdr.jpg";

        let env_texture_handle = engine
            .assets
            .load_texture(map_path, ColorSpace::Srgb, false)
            .expect("Failed to load environment map");

        let scene = engine.scene_manager.create_active();

        scene.environment.set_env_map(Some(env_texture_handle));

        // === 3. Add light ===
        let mut dir_light = Light::new_directional(Vec3::ONE, 5.0);
        dir_light.cast_shadows = true;
        if let Some(shadow) = dir_light.shadow.as_mut() {
            shadow.map_size = 2048;
        }
        let light_node = scene.add_light(dir_light);

        if let Some(node) = scene.get_node_mut(light_node) {
            node.transform.position = Vec3::new(0.0, 12.0, 6.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        let ground_geo = engine
            .assets
            .geometries
            .add(Geometry::new_plane(30.0, 30.0));
        let ground_material = MeshPhongMaterial::new(Vec4::new(0.2, 0.3, 0.4, 1.0));
        ground_material.set_side(Side::Double);
        let ground_mat = engine.assets.materials.add(ground_material);
        let mut ground = Mesh::new(ground_geo, ground_mat);
        ground.receive_shadows = true;
        let ground_node = scene.add_mesh(ground);

        if let Some(node) = scene.get_node_mut(ground_node) {
            node.transform.rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            node.transform.position = Vec3::new(0.0, 0.0, 0.0);
        }

        // === 4. Load glTF model with skinning animation ===
        println!("Loading glTF model from: {:?}", gltf_path);

        let prefab = match GltfLoader::load(gltf_path, engine.assets.clone()) {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Error loading model '{}': {}", gltf_path.display(), e);
                panic!("Failed to load model");
            }
        };
        let gltf_node = scene.instantiate(&prefab);

        println!("Successfully loaded root node: {:?}", gltf_node);

        //  Play skinning animation if available  ---
        if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
            println!("Loaded animations:");

            let animations = mixer.list_animations();

            for anim_name in &animations {
                println!(" - {}", anim_name);
            }

            mixer.play("SambaDance");
        }

        // === 5. Setup Camera ===
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 1.5, 4.0);
            node.transform.look_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        }
        scene.active_camera = Some(cam_node_id);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 1.5, 4.0), Vec3::new(0.0, 1.0, 0.0)),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Skinning Animation | FPS: {:.2}", fps));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RenderSettings {
            vsync: false,
            enable_hdr: false,
            ..Default::default()
        })
        .run::<SkinningDemo>()
}
