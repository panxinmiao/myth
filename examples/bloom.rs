//! [gallery]
//! name = "Bloom"
//! category = "Post-Processing"
//! description = "Interactive bloom tuning demo driven by a glTF character and HDR lighting."
//! order = 310
//!

//! Physically-Based Bloom Example
//!
//! Demonstrates the engine's physically-based bloom post-processing effect
//! using the DamagedHelmet glTF model with HDR environment lighting.
//!
//! Controls:
//! - Mouse drag: Orbit camera
//! - Scroll: Zoom
//! - 1/2: Decrease/increase bloom strength
//! - 3/4: Decrease/increase bloom radius
//! - K: Toggle Karis average
//! - B: Toggle bloom on/off
//! - Up/Down: Adjust exposure

use myth::prelude::*;
use myth::resources::Key;
use myth::utils::FpsCounter;

#[cfg(not(target_arch = "wasm32"))]
const ASSET_PATH: &str = "examples/assets/";
#[cfg(target_arch = "wasm32")]
const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "assets/",
};

struct BloomDemo {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    model_prefab: PrefabHandle,
    model_loaded: bool,
}

impl AppHandler for BloomDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        // Load HDR environment map for realistic IBL lighting
        let env_texture_handle = engine
            .assets
            .load_hdr_texture(format!("{}envs/blouberg_sunrise_2_1k.hdr", ASSET_PATH));

        let scene = engine.scene_manager.create_active();
        scene.environment.set_env_map(Some(env_texture_handle));
        scene.environment.set_intensity(1.5);

        // Add a bright directional light to create specular highlights
        scene.add_light(Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 3.0));

        // Load the DamagedHelmet model (has nice emissive and specular detail)
        let model_prefab = engine
            .assets
            .load_gltf(format!("{}phoenix_bird.glb", ASSET_PATH));

        let controls = OrbitControls::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO);

        // Configure bloom
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.4);
        scene.bloom.set_radius(0.005);
        scene.bloom.set_karis_average(true);

        // Setup camera
        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node_id)
            .set_position(0.0, 0.0, 3.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);

        println!("=== Physically-Based Bloom Demo ===");
        println!("Controls:");
        println!("  B       - Toggle bloom on/off");
        println!("  1/2     - Decrease/increase bloom strength");
        println!("  3/4     - Decrease/increase bloom radius");
        println!("  K       - Toggle Karis average");
        println!("  Up/Down - Adjust exposure");
        println!("  Mouse   - Orbit camera");

        Self {
            cam_node_id,
            controls,
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
                scene.play_if_any_animation(gltf_node);
                scene.update_matrix_world();
                if let Some(bbox) = scene.get_bbox_of_node(gltf_node, &assets) {
                    self.controls.fit(&bbox);
                }
                self.model_loaded = true;
            } else if let Some(err) = assets.prefabs.get_error(self.model_prefab) {
                eprintln!("Failed to load bloom demo model: {err}");
                self.model_loaded = true;
            }
        }

        // Camera controls
        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        // Interactive bloom controls
        let input = &engine.input;

        // Toggle bloom
        if input.get_key_down(Key::B) {
            let toggled = !scene.bloom.enabled;
            scene.bloom.set_enabled(toggled);
            println!("Bloom: {}", if toggled { "ON" } else { "OFF" });
        }

        // Bloom strength: 1 = decrease, 2 = increase
        if input.get_key_down(Key::Key1) {
            let new_val = (scene.bloom.strength() - 0.01).max(0.0);
            scene.bloom.set_strength(new_val);
            println!("Bloom strength: {:.3}", new_val);
        }
        if input.get_key_down(Key::Key2) {
            let new_val = (scene.bloom.strength() + 0.01).min(1.0);
            scene.bloom.set_strength(new_val);
            println!("Bloom strength: {:.3}", new_val);
        }

        // Bloom radius: 3 = decrease, 4 = increase
        if input.get_key_down(Key::Key3) {
            let new_val = (scene.bloom.radius() - 0.001).max(0.001);
            scene.bloom.set_radius(new_val);
            println!("Bloom radius: {:.4}", new_val);
        }
        if input.get_key_down(Key::Key4) {
            let new_val = (scene.bloom.radius() + 0.001).min(0.05);
            scene.bloom.set_radius(new_val);
            println!("Bloom radius: {:.4}", new_val);
        }

        // Toggle Karis average
        if input.get_key_down(Key::K) {
            let toggled = !scene.bloom.karis_average;
            scene.bloom.set_karis_average(toggled);
            println!("Karis average: {}", if toggled { "ON" } else { "OFF" });
        }

        // Exposure: Up = increase, Down = decrease
        if input.get_key_down(Key::ArrowUp) {
            let new_val = scene.tone_mapping.exposure() + 0.1;
            scene.tone_mapping.set_exposure(new_val);
            println!("Exposure: {:.2}", new_val);
        }
        if input.get_key_down(Key::ArrowDown) {
            let new_val = (scene.tone_mapping.exposure() - 0.1).max(0.1);
            scene.tone_mapping.set_exposure(new_val);
            println!("Exposure: {:.2}", new_val);
        }

        // FPS counter
        if let Some(fps) = self.fps_counter.update() {
            let bloom_status = if scene.bloom.enabled {
                format!(
                    "ON s={:.3} r={:.4}",
                    scene.bloom.strength(),
                    scene.bloom.radius()
                )
            } else {
                "OFF".to_string()
            };
            window.set_title(&format!(
                "Bloom Demo - FPS: {:.0} | Bloom: {} | Exposure: {:.2}",
                fps,
                bloom_status,
                scene.tone_mapping.exposure()
            ));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<BloomDemo>()
}
