//! Showcase Viewer
//! a lightweight glTF viewer tailored for web showcase.
//!
//! Features:
//! 1. Automatically reads model URL from query parameter (?model=...)
//! 2. Automatically computes bounding box and adjusts camera view
//! 3. Automatically plays the first animation (if any)
//! 4. Built-in HDR environment lighting
//!
//! Build command: cargo build --example showcase_viewer --target wasm32-unknown-unknown --release

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use myth::assets::SharedPrefab;
use myth::prelude::*;
use myth::utils::FpsCounter;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(
    inline_js = "export function is_mobile_device() { return /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent); }"
)]
extern "C" {
    fn is_mobile_device() -> bool;
}

// Native fallback for `is_mobile_device` when not running in WASM.
#[cfg(not(target_arch = "wasm32"))]
fn is_mobile_device() -> bool {
    false
}

struct ShowcaseApp {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,

    // State flags
    loading_started: bool,
    model_loaded: bool,
}

#[cfg(not(target_arch = "wasm32"))]
const ASSET_PATH: &str = "examples/assets/";
#[cfg(target_arch = "wasm32")]
const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path, // CI or user-defined path via build_wasm.sh
    None => "assets/",  // Local default path for testing with build_wasm.sh
};

impl AppHandler for ShowcaseApp {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        // 1. Load HDR environment map (fire-and-forget, auto-promoted on next frame)
        let map_path = "envs/royal_esplanade_2k.hdr.jpg";
        let env_map_path = format!("{}{}", ASSET_PATH, map_path);
        let env_handle = engine
            .assets
            .load_texture(env_map_path, ColorSpace::Srgb, false);
        scene.environment.set_env_map(Some(env_handle));

        // Set base ambient light as fallback before HDR loads
        scene.environment.set_ambient_light(Vec3::splat(0.2));

        // 2. Add directional light (auxiliary lighting)
        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 3.0);
        let light_node = scene.add_light(light);
        if let Some(node) = scene.get_node_mut(light_node) {
            node.transform.position = Vec3::new(5.0, 10.0, 5.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        // 3. Set up camera
        let mut camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.01);
        if is_mobile_device() {
            camera.aa_mode = AntiAliasingMode::FXAA(FxaaSettings::default());
        } else {
            camera.aa_mode = AntiAliasingMode::MSAA_FXAA(4, FxaaSettings::default());
        }
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 5.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        scene.active_camera = Some(cam_node_id);

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
            loading_started: false,
            model_loaded: false,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // --- 1. Submit model load (once) ---
        if !self.loading_started {
            self.loading_started = true;

            let url = get_model_url().unwrap_or_else(|| {
                "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb".to_string()
            });

            log::info!("Starting load for: {}", url);
            engine.assets.load_gltf(url);
        }

        // --- 2. Handle completed prefab loads ---
        for loaded in engine.assets.take_loaded_prefabs() {
            log::info!("Model loaded successfully: {}", loaded.source);
            self.instantiate_and_focus(scene, &engine.assets, &loaded.prefab);
            self.model_loaded = true;

            #[cfg(target_arch = "wasm32")]
            {
                use web_sys::window;
                if let Some(win) = window() {
                    if let Some(doc) = win.document() {
                        if let Some(el) = doc.get_element_by_id("loading-overlay") {
                            let _ = el.class_list().add_1("hidden");
                        }
                    }
                }
            }
        }

        // --- 3. Update controls ---
        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        // --- 4. Debug output (optional) ---
        self.fps_counter.update();
    }
}

impl ShowcaseApp {
    fn instantiate_and_focus(
        &mut self,
        scene: &mut Scene,
        assets: &AssetServer,
        prefab: &SharedPrefab,
    ) {
        // 1. Instantiate model
        let root_node = scene.instantiate(prefab);

        // 2. Ensure transform matrices are updated for bounding box calculation
        scene.update_subtree(root_node);

        // 3. Auto-play animation
        if let Some(mixer) = scene.animation_mixers.get_mut(root_node) {
            let anims = mixer.list_animations();
            if let Some(first_anim) = anims.first() {
                log::info!("Auto-playing animation: {}", first_anim);
                mixer.play(first_anim);
            }
        }

        // 4. Calculate bounding box and focus camera
        if let Some(bbox) = scene.get_bbox_of_node(root_node, assets) {
            let center = bbox.center();
            let radius = bbox.size().length() * 0.5;

            log::info!("Model BBox: Center={:?}, Radius={}", center, radius);

            self.controls.set_target(center);
            self.controls
                .set_position(center + Vec3::new(0.0, radius, radius * 2.5));

            if let Some((_, camera)) = scene.query_main_camera_bundle() {
                camera.set_near(radius * 0.01);
            }
        }
    }
}

// --- Platform-related helper functions ---

// Parse URL parameters
#[cfg(target_arch = "wasm32")]
fn get_model_url() -> Option<String> {
    let window = web_sys::window()?;
    let location = window.location();
    let search = location.search().ok()?;
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    params.get("model")
}

#[cfg(not(target_arch = "wasm32"))]
fn get_model_url() -> Option<String> {
    std::env::args().nth(1)
}

// --- Entry point ---

#[cfg(not(target_arch = "wasm32"))]
fn main() -> myth::Result<()> {
    env_logger::init();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _enter = rt.enter();

    App::new()
        .with_title("Myth Showcase Viewer")
        .with_settings(RendererSettings {
            anisotropy_clamp: 4,
            ..Default::default()
        })
        .run::<ShowcaseApp>()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).expect("Failed to init logger");

    App::new()
        .with_settings(RendererSettings {
            anisotropy_clamp: if is_mobile_device() { 1 } else { 4 },
            ..Default::default()
        })
        .run::<ShowcaseApp>()
        .unwrap();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
