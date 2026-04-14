//! [gallery]
//! name = "Skinning"
//! category = "Animation"
//! description = "Skinned character animation sample with orbit camera and glTF playback."
//! order = 510
//!

#[cfg(not(target_arch = "wasm32"))]
use std::env;

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

/// Skinning Animation Example
///
struct SkinningDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    model_prefab: PrefabHandle,
    model_loaded: bool,
}

impl AppHandler for SkinningDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let gltf_source = {
            let args: Vec<String> = env::args().collect();
            let default_path = format!("{}Michelle.glb", ASSET_PATH);
            if args.len() > 1 {
                args[1].clone()
            } else {
                println!("Tip: You can pass a model path as an argument.");
                println!("Usage: cargo run --example skinning -- <path_to_gltf>");
                println!("No path provided, loading default: {}", default_path);
                default_path
            }
        };
        #[cfg(target_arch = "wasm32")]
        let gltf_source = format!("{}Michelle.glb", ASSET_PATH);

        // Load environment map
        let env_texture_handle = engine.assets.load_texture(
            format!("{}envs/royal_esplanade_2k.hdr.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            false,
        );

        let scene = engine.scene_manager.create_active();
        scene.environment.set_env_map(Some(env_texture_handle));
        scene.add_light(Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0));

        // Load glTF model
        println!("Loading glTF model from: {gltf_source}");
        let model_prefab = engine.assets.load_gltf(gltf_source);

        // Camera — chainable
        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node_id)
            .set_position(0.0, 1.5, 4.0)
            .look_at(Vec3::new(0.0, 1.0, 0.0));
        scene.active_camera = Some(cam_node_id);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 1.5, 4.0), Vec3::new(0.0, 1.0, 0.0)),
            fps_counter: FpsCounter::new(),
            model_prefab,
            model_loaded: false,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let assets = engine.assets.clone();
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if !self.model_loaded {
            if let Some(prefab) = assets.prefabs.get(self.model_prefab) {
                let gltf_node = scene.instantiate(prefab.as_ref());
                println!("Successfully loaded root node: {:?}", gltf_node);

                if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
                    println!("Loaded animations:");
                    let animations = mixer.list_animations();
                    for anim_name in &animations {
                        println!(" - {}", anim_name);
                    }
                    mixer.play("SambaDance");
                }

                self.model_loaded = true;
            } else if let Some(err) = assets.prefabs.get_error(self.model_prefab) {
                eprintln!("Failed to load skinning demo model: {err}");
                self.model_loaded = true;
            }
        }

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Skinning Animation | FPS: {:.2}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_settings(RendererSettings {
            path: RenderPath::BasicForward,
            vsync: false,
            ..Default::default()
        })
        .run::<SkinningDemo>()
}
