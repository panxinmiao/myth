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

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use winit::window::Window;

use myth::assets::SharedPrefab;
use myth::prelude::*;
use myth::utils::FpsCounter;

// Define asset loading events for handling asynchronous results in the update loop
enum AssetEvent {
    ModelLoaded { prefab: SharedPrefab, url: String },
    HdrLoaded(TextureHandle),
}

struct ShowcaseApp {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,

    // State flags
    loading_started: bool,
    model_loaded: bool,

    // Asynchronous communication channels
    rx: Receiver<AssetEvent>,
    tx: Sender<AssetEvent>, // Retain sender to clone for async tasks
}

#[cfg(not(target_arch = "wasm32"))]
const ASSET_PATH: &str = "examples/assets/";
#[cfg(target_arch = "wasm32")]
const ASSET_PATH: &str = "assets/";

impl AppHandler for ShowcaseApp {
    fn init(engine: &mut Engine, _window: &Arc<Window>) -> Self {
        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        // 1. init communication channel
        let (tx, rx) = channel();

        // 2. Load default HDR environment map (async)
        let asset_server = engine.assets.clone();
        let hdr_tx = tx.clone();

        let map_path = "royal_esplanade_2k.hdr.jpg";
        let env_map_path = format!("{}{}", ASSET_PATH, map_path);

        execute_future(async move {
            if let Ok(handle) = asset_server
                .load_texture_async(env_map_path, ColorSpace::Srgb, false)
                .await
            {
                let _ = hdr_tx.send(AssetEvent::HdrLoaded(handle));
            }
        });

        // Set base ambient light as fallback before HDR loads
        scene.environment.set_ambient_color(Vec3::splat(0.2));

        // 3. Add directional light (auxiliary lighting)
        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 3.0);
        let light_node = scene.add_light(light);
        if let Some(node) = scene.get_node_mut(light_node) {
            node.transform.position = Vec3::new(5.0, 10.0, 5.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        // 4. Set up camera
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.01);
        let cam_node_id = scene.add_camera(camera);

        // Initial camera position (will be overridden by auto-focus)
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
            rx,
            tx,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // --- 1. Start loading logic (only once) ---
        if !self.loading_started {
            self.loading_started = true;

            // Get URL parameter
            let url = get_model_url().unwrap_or_else(|| {
                // Default fallback model
                "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb".to_string()
            });

            log::info!("Starting load for: {}", url);

            let assets = engine.assets.clone();
            let tx = self.tx.clone();

            execute_future(async move {
                match GltfLoader::load_async(url.clone(), assets).await {
                    Ok(prefab) => {
                        let _ = tx.send(AssetEvent::ModelLoaded { prefab, url: url });
                    }
                    Err(e) => {
                        log::error!("Failed to load model: {}", e);
                    }
                }
            });
        }

        // --- 2. Handle asynchronous events ---
        while let Ok(event) = self.rx.try_recv() {
            match event {
                AssetEvent::HdrLoaded(handle) => {
                    log::info!("HDR Environment loaded");
                    println!("Setting HDR environment map");
                    scene.environment.set_env_map(Some(handle));
                    scene.environment.set_intensity(1.0);
                }
                AssetEvent::ModelLoaded { prefab, url } => {
                    log::info!("Model loaded successfully: {}", url);
                    self.instantiate_and_focus(scene, &prefab, &engine.assets);
                    self.model_loaded = true;
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
        prefab: &SharedPrefab,
        assets: &AssetServer,
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
                camera.near = radius * 0.01;
                camera.update_projection_matrix();
            }
        }
    }
}

// --- Platform-related helper functions ---

#[cfg(not(target_arch = "wasm32"))]
fn execute_future<F: std::future::Future<Output = ()> + Send + 'static>(f: F) {
    tokio::spawn(f);
}

#[cfg(target_arch = "wasm32")]
fn execute_future<F: std::future::Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}

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
        .with_settings(RenderSettings {
            enable_hdr: true,
            msaa_samples: 4,
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
        .with_settings(RenderSettings {
            enable_hdr: true,
            msaa_samples: 4,
            ..Default::default()
        })
        .run::<ShowcaseApp>()
        .unwrap();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
