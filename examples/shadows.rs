use myth::prelude::*;
use myth::utils::FpsCounter;
use std::env;
use std::path::Path;

/// Skinning Animation Example
///
struct SkinningDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for SkinningDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let args: Vec<String> = env::args().collect();
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

        // Load environment map
        let map_path = "examples/assets/royal_esplanade_2k.hdr.jpg";
        let env_texture_handle = engine
            .assets
            .load_texture(map_path, ColorSpace::Srgb, false)
            .expect("Failed to load environment map");

        let scene = engine.scene_manager.create_active();
        scene.environment.set_env_map(Some(env_texture_handle));

        // Directional light with shadows
        let mut dir_light = Light::new_directional(Vec3::ONE, 5.0);
        dir_light.cast_shadows = true;
        if let Some(shadow) = dir_light.shadow.as_mut() {
            shadow.map_size = 2048;
        }
        let light_node = scene.add_light(dir_light);
        scene
            .node(&light_node)
            .set_position(0.0, 12.0, 6.0)
            .look_at(Vec3::ZERO);

        // Ground plane — spawn + builder material + chainable node
        let ground_node = scene.spawn_plane(
            30.0,
            30.0,
            PhongMaterial::new(Vec4::new(0.2, 0.3, 0.4, 1.0)).with_side(Side::Double),
        );
        scene
            .node(&ground_node)
            .set_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .set_cast_shadows(false)
            .set_receive_shadows(true);

        // Load glTF model
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

        // Play animation
        if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
            println!("Loaded animations:");
            let animations = mixer.list_animations();
            for anim_name in &animations {
                println!(" - {}", anim_name);
            }
            mixer.play("SambaDance");
        }

        // Camera
        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node_id)
            .set_position(0.0, 1.5, 4.0)
            .look_at(Vec3::new(0.0, 1.0, 0.0));
        scene.active_camera = Some(cam_node_id);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 1.5, 4.0), Vec3::new(0.0, 1.0, 0.0)),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
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
        .with_settings(RendererSettings {
            path: RenderPath::BasicForward { msaa_samples: 1 },
            vsync: false,
            ..Default::default()
        })
        .run::<SkinningDemo>()
}
