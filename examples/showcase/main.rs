//! Myth Engine — Showcase Viewer
//!
//! A cinematic glTF showcase designed for hero demos and open-source launches.
//! Features visual presets (Cinematic / Studio / Daylight), auto-camera,
//! full post-processing pipeline, and an elegant web overlay.
//!
//! Native:  `cargo run --example showcase --release`
//! WASM:    `scripts/build_wasm.bat showcase`

use myth_resources::tone_mapping::AgxLook;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender, channel};

use myth::ToneMappingMode;
use myth::assets::SharedPrefab;
use myth::prelude::*;
use myth::utils::FpsCounter;
use myth_resources::MouseButton;

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

// ── Constants ───────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
const ASSET_PATH: &str = "examples/assets/";
#[cfg(target_arch = "wasm32")]
const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path, // CI or user-defined path via build_wasm.sh
    None => "assets/",  // Local default path for testing with build_wasm.sh
};

const DEFAULT_MODEL: &str = "cute_girl.glb";
const DEFAULT_PRESET: VisualPreset = VisualPreset::Cinematic;

// ── Visual Preset Enum ──────────────────────────────────────────────────────

/// Available rendering presets for the showcase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisualPreset {
    Cinematic,
    Studio,
    Daylight,
}

// ── WASM ↔ JS Bridge ───────────────────────────────────────────────────────

thread_local! {
    static PRESET_COMMAND_QUEUE: RefCell<Vec<VisualPreset>> = RefCell::new(Vec::new());
}

/// Called from JavaScript to switch the active visual preset.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn switch_preset(preset_idx: u32) {
    let preset = match preset_idx {
        0 => VisualPreset::Cinematic,
        1 => VisualPreset::Studio,
        _ => VisualPreset::Daylight,
    };
    PRESET_COMMAND_QUEUE.with(|q| q.borrow_mut().push(preset));
}

// ── Preset Data Model ───────────────────────────────────────────────────────

/// Configuration for a single directional light in the three-point rig.
#[derive(Clone)]
struct LightConfig {
    color: Vec3,
    intensity: f32,
    position: Vec3,
    cast_shadows: bool,
}

/// How the background should be rendered for a given preset.
#[derive(Clone)]
#[allow(dead_code)]
enum PresetBackground {
    /// Solid fill color.
    Color(Vec4),
    /// Top-to-bottom gradient.
    Gradient { top: Vec4, bottom: Vec4 },
    /// HDR equirectangular skybox. Falls back to a gradient when the HDR
    /// texture has not finished loading.
    Skybox {
        intensity: f32,
        fallback_top: Vec4,
        fallback_bottom: Vec4,
    },
}

/// Complete rendering parameters for one visual preset.
///
/// Pure data container — [`ShowcaseApp::apply_preset`] reads from it and
/// writes to the scene. Adding or tuning a preset never requires changing
/// rendering logic.
#[derive(Clone)]
struct RenderPreset {
    // ── Environment
    env_intensity: f32,
    ambient_light: Vec3,
    background: PresetBackground,

    // ── Three-point lighting
    key_light: LightConfig,
    fill_light: LightConfig,
    rim_light: LightConfig,

    // ── Tone mapping
    tone_mapping_mode: ToneMappingMode,
    exposure: f32,
    contrast: f32,
    saturation: f32,
    vignette_intensity: f32,
    vignette_smoothness: f32,
    vignette_color: Vec4,
    chromatic_aberration: f32,
    film_grain: f32,
    lut_contribution: f32,

    // ── Bloom
    bloom_enabled: bool,
    bloom_strength: f32,
    bloom_radius: f32,

    // ── SSAO
    ssao_enabled: bool,
    ssao_radius: f32,
    ssao_intensity: f32,

    // —— Screen-space effects
    ssss_enabled: bool,

    // ── Per-preset resources (filenames relative to envs/ or luts/)
    hdr_filename: Option<&'static str>,
    lut_filename: Option<&'static str>,

    // ── Loaded resource handles (populated asynchronously)
    hdr_handle: Option<TextureHandle>,
    lut_handle: Option<TextureHandle>,
}

/// Builds the default preset table with all rendering parameters.
fn build_presets() -> HashMap<VisualPreset, RenderPreset> {
    let mut presets = HashMap::new();

    let is_mobile = is_mobile_device();

    // ── Cinematic: dark IBL, warm key / cool fill, ACES filmic, heavy bloom
    presets.insert(
        VisualPreset::Cinematic,
        RenderPreset {
            env_intensity: 0.8,
            ambient_light: Vec3::splat(0.02),

            background: PresetBackground::Skybox {
                intensity: 1.0,
                fallback_top: Vec4::new(0.02, 0.02, 0.06, 1.0),
                fallback_bottom: Vec4::new(0.0, 0.0, 0.0, 1.0),
            },

            key_light: LightConfig {
                color: Vec3::new(1.0, 0.85, 0.6),
                intensity: 5.0,
                position: Vec3::new(5.0, 5.0, 5.0),
                cast_shadows: false,
            },
            fill_light: LightConfig {
                color: Vec3::new(0.3, 0.5, 1.0),
                intensity: 1.0,
                position: Vec3::new(-5.0, 2.0, -2.0),
                cast_shadows: false,
            },
            rim_light: LightConfig {
                color: Vec3::ONE,
                intensity: 0.0,
                position: Vec3::ZERO,
                cast_shadows: false,
            },

            tone_mapping_mode: ToneMappingMode::AgX(AgxLook::None),
            exposure: 1.0,
            contrast: 1.1,
            saturation: 1.05,
            vignette_intensity: 0.4,
            vignette_smoothness: 0.6,
            vignette_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            chromatic_aberration: 0.002,
            film_grain: 0.03,
            lut_contribution: 1.0,

            bloom_enabled: !is_mobile,
            bloom_strength: 0.06,
            bloom_radius: 0.005,

            ssao_enabled: !is_mobile,
            ssao_radius: 0.5,
            ssao_intensity: 1.5,

            ssss_enabled: true,

            hdr_filename: Some("blouberg_sunrise_2_1k.hdr"),
            lut_filename: Some("Rec709 Fujifilm 3513DI D65.cube"),
            hdr_handle: None,
            lut_handle: None,
        },
    );

    // ── Studio: no IBL, pure three-point lighting, neutral tone, strong rim
    presets.insert(
        VisualPreset::Studio,
        RenderPreset {
            env_intensity: 0.3,
            ambient_light: Vec3::splat(0.02),
            background: PresetBackground::Color(Vec4::new(0.03, 0.03, 0.04, 1.0)),

            key_light: LightConfig {
                color: Vec3::new(1.0, 0.85, 0.6),
                intensity: 5.0,
                position: Vec3::new(5.0, 5.0, 5.0),
                cast_shadows: false,
            },
            fill_light: LightConfig {
                color: Vec3::new(0.3, 0.5, 1.0),
                intensity: 1.0,
                position: Vec3::new(-5.0, 2.0, -2.0),
                cast_shadows: false,
            },
            rim_light: LightConfig {
                color: Vec3::ONE,
                intensity: 0.0,
                position: Vec3::ZERO,
                cast_shadows: false,
            },

            tone_mapping_mode: ToneMappingMode::AgX(AgxLook::None),
            exposure: 1.2,
            contrast: 1.1,
            saturation: 1.05,
            vignette_intensity: 0.4,
            vignette_smoothness: 0.6,
            vignette_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            chromatic_aberration: 0.002,
            film_grain: 0.0,
            lut_contribution: 0.4,

            bloom_enabled: !is_mobile,
            bloom_strength: 0.06,
            bloom_radius: 0.005,

            ssao_enabled: !is_mobile,
            ssao_radius: 0.5,
            ssao_intensity: 1.5,

            ssss_enabled: true,

            hdr_filename: Some("blouberg_sunrise_2_1k.hdr"),
            lut_filename: Some("Rec709 Kodak 2383 D65.cube"),
            hdr_handle: None,
            lut_handle: None,
        },
    );

    // ── Daylight: full HDR IBL skybox, warm sun key, natural look
    presets.insert(
        VisualPreset::Daylight,
        RenderPreset {
            env_intensity: 3.0,
            ambient_light: Vec3::splat(0.1),
            background: PresetBackground::Skybox {
                intensity: 1.0,
                fallback_top: Vec4::new(0.4, 0.6, 0.9, 1.0),
                fallback_bottom: Vec4::new(0.85, 0.8, 0.7, 1.0),
            },

            key_light: LightConfig {
                color: Vec3::new(1.0, 0.95, 0.9),
                intensity: 2.5,
                position: Vec3::new(1.0, 1.0, 1.0),
                cast_shadows: false,
            },
            fill_light: LightConfig {
                color: Vec3::ONE,
                intensity: 0.0,
                position: Vec3::ZERO,
                cast_shadows: false,
            },
            rim_light: LightConfig {
                color: Vec3::ONE,
                intensity: 0.0,
                position: Vec3::ZERO,
                cast_shadows: false,
            },

            tone_mapping_mode: ToneMappingMode::Neutral,
            exposure: 0.7,
            contrast: 1.0,
            saturation: 1.05,
            vignette_intensity: 0.0,
            vignette_smoothness: 0.5,
            vignette_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            chromatic_aberration: 0.0,
            film_grain: 0.0,
            lut_contribution: 0.0,

            bloom_enabled: !is_mobile,
            bloom_strength: 0.04,
            bloom_radius: 0.005,

            ssao_enabled: false,
            ssao_radius: 0.5,
            ssao_intensity: 1.0,

            ssss_enabled: true,

            hdr_filename: Some("spruit_sunrise_1k.hdr"),
            lut_filename: None,
            hdr_handle: None,
            lut_handle: None,
        },
    );

    presets
}

// ── Async Events ────────────────────────────────────────────────────────────

/// Events sent from async asset-loading tasks back to the main thread.
enum AssetEvent {
    ModelLoaded(SharedPrefab),
    HdrLoaded(&'static str, TextureHandle),
    LutLoaded(&'static str, TextureHandle),
}

// ── Application State ───────────────────────────────────────────────────────

struct ShowcaseApp {
    cam_node_id: NodeHandle,
    key_light_id: NodeHandle,
    fill_light_id: NodeHandle,
    rim_light_id: NodeHandle,

    controls: OrbitControls,
    fps_counter: FpsCounter,

    /// All preset parameters and per-preset resource handles.
    presets: HashMap<VisualPreset, RenderPreset>,
    current_preset: VisualPreset,

    idle_timer: f32,
    orbit_target: Vec3,

    loading_started: bool,
    model_loaded: bool,
    /// `true` once the model and the default preset's critical resources
    /// have all finished loading — at which point the loading overlay is hidden.
    initial_ready: bool,

    rx: Receiver<AssetEvent>,
    tx: Sender<AssetEvent>,
}

// ── AppHandler ──────────────────────────────────────────────────────────────

impl AppHandler for ShowcaseApp {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        let (tx, rx) = channel();
        let presets = build_presets();

        let mut requested_hdrs = HashSet::new();
        let mut requested_luts = HashSet::new();

        // Dispatch async loads for every preset's HDR and LUT in parallel.
        for config in presets.values() {
            if let Some(hdr_file) = config.hdr_filename {
                if requested_hdrs.insert(hdr_file) {
                    let path = format!("{}envs/{}", ASSET_PATH, hdr_file);
                    let tx = tx.clone();
                    let assets = engine.assets.clone();

                    execute_future(async move {
                        let is_hdr = hdr_file.ends_with(".hdr") || hdr_file.ends_with(".exr");
                        let res = if is_hdr {
                            assets.load_hdr_texture_async(path).await
                        } else {
                            assets
                                .load_texture_async(path, ColorSpace::Srgb, false)
                                .await
                        };

                        match res {
                            Ok(h) => {
                                let _ = tx.send(AssetEvent::HdrLoaded(hdr_file, h));
                            }
                            Err(e) => log::error!("{:?} HDR load failed: {}", hdr_file, e),
                        }
                    });
                }
            }
            if let Some(lut_file) = config.lut_filename {
                if requested_luts.insert(lut_file) {
                    let path = format!("{}luts/{}", ASSET_PATH, lut_file);
                    let tx = tx.clone();
                    let assets = engine.assets.clone();
                    execute_future(async move {
                        match assets.load_lut_texture_async(path).await {
                            Ok(h) => {
                                let _ = tx.send(AssetEvent::LutLoaded(lut_file, h));
                            }
                            Err(e) => log::error!("{:?} LUT load failed: {}", lut_file, e),
                        }
                    });
                }
            }
        }

        // Fallback ambient before HDR is ready.
        scene.environment.set_ambient_light(Vec3::splat(0.15));

        // Three-point lighting rig (values overridden by apply_preset).
        let key_light_id = scene.add_light(Light::new_directional(Vec3::ONE, 1.0));
        let fill_light_id = scene.add_light(Light::new_directional(Vec3::ONE, 1.0));
        let rim_light_id = scene.add_light(Light::new_directional(Vec3::ONE, 1.0));

        // Camera.
        let mut camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.01);

        if is_mobile_device() {
            camera.set_aa_mode(AntiAliasingMode::FXAA(FxaaSettings::default()));
        } else {
            camera.set_aa_mode(AntiAliasingMode::MSAA_FXAA(4, FxaaSettings::default()));
        }

        let cam_node_id = scene.add_camera(camera);
        scene.active_camera = Some(cam_node_id);

        // Orbit controls.
        let initial_target = Vec3::new(0.0, 1.5, 0.0);
        let mut controls = OrbitControls::new(Vec3::new(-0.1, 1.45, -0.45), initial_target);
        controls.min_distance = 0.1;
        controls.max_distance = 20.0;
        controls.max_polar_angle = std::f32::consts::FRAC_PI_2 * 1.4;
        controls.min_polar_angle = std::f32::consts::FRAC_PI_2 * 0.6;

        let app = Self {
            cam_node_id,
            key_light_id,
            fill_light_id,
            rim_light_id,
            controls,
            fps_counter: FpsCounter::new(),
            presets,
            current_preset: DEFAULT_PRESET,
            idle_timer: 0.0,
            orbit_target: initial_target,
            loading_started: false,
            model_loaded: false,
            initial_ready: false,
            rx,
            tx,
        };

        app.apply_preset(scene, DEFAULT_PRESET);
        app
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // 1. Kick off model loading once.
        if !self.loading_started {
            self.loading_started = true;
            let model_name = DEFAULT_MODEL.to_string();
            let model_path = format!("{}{}", ASSET_PATH, model_name);
            log::info!("Loading model: {}", model_path);

            let assets = engine.assets.clone();
            let tx = self.tx.clone();
            execute_future(async move {
                match GltfLoader::load_async(model_path, assets).await {
                    Ok(prefab) => {
                        let _ = tx.send(AssetEvent::ModelLoaded(prefab));
                    }
                    Err(e) => log::error!("Model load failed: {}", e),
                }
            });
        }

        // 2. Process async asset events.
        while let Ok(event) = self.rx.try_recv() {
            match event {
                AssetEvent::HdrLoaded(filename, handle) => {
                    let mut needs_refresh = false;
                    for (&preset_key, p) in self.presets.iter_mut() {
                        if p.hdr_filename == Some(filename) {
                            p.hdr_handle = Some(handle.clone());
                            if preset_key == self.current_preset {
                                needs_refresh = true;
                            }
                        }
                    }
                    if needs_refresh {
                        self.apply_preset(scene, self.current_preset);
                    }
                }
                AssetEvent::LutLoaded(filename, handle) => {
                    let mut needs_refresh = false;
                    for (&preset_key, p) in self.presets.iter_mut() {
                        if p.lut_filename == Some(filename) {
                            p.lut_handle = Some(handle.clone());
                            if preset_key == self.current_preset {
                                needs_refresh = true;
                            }
                        }
                    }
                    if needs_refresh {
                        self.apply_preset(scene, self.current_preset);
                    }
                }
                AssetEvent::ModelLoaded(prefab) => {
                    self.instantiate_and_focus(scene, &engine.assets, &prefab);
                    self.model_loaded = true;
                }
            }
        }

        // 3. Smart loading overlay: hide only when critical-path assets are ready.
        if !self.initial_ready && self.check_initial_ready() {
            self.initial_ready = true;
            #[cfg(target_arch = "wasm32")]
            hide_loading_overlay();
        }

        // 4. Handle preset switch commands from JS.
        PRESET_COMMAND_QUEUE.with(|q| {
            let mut queue = q.borrow_mut();
            for preset in queue.drain(..) {
                if preset != self.current_preset {
                    self.current_preset = preset;
                    self.apply_preset(scene, preset);
                }
            }
        });

        // 5. Camera controls + idle auto-rotation.
        let interacting = engine.input.get_mouse_button(MouseButton::Left)
            || engine.input.get_mouse_button(MouseButton::Right)
            || engine.input.scroll_delta().y.abs() > 0.01;

        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);

            if !interacting {
                self.idle_timer += frame.dt;
                if self.idle_timer > 60.0 {
                    let cam_pos = cam_node.transform.position;
                    let rotated = Quat::from_axis_angle(Vec3::Y, 0.08 * frame.dt)
                        * (cam_pos - self.orbit_target)
                        + self.orbit_target;

                    self.controls.set_position(rotated);
                }
            } else {
                self.idle_timer = 0.0;
            }
        }

        self.fps_counter.update();
    }
}

// ── Preset Application ──────────────────────────────────────────────────────

impl ShowcaseApp {
    /// Applies the given preset's rendering parameters to the active scene.
    ///
    /// This is a pure data-binding function — it reads from [`RenderPreset`]
    /// and writes to the scene. No preset-specific branching lives here.
    fn apply_preset(&self, scene: &mut Scene, preset: VisualPreset) {
        let Some(p) = self.presets.get(&preset) else {
            log::warn!("Unknown preset: {:?}", preset);
            return;
        };

        log::info!("Applying preset: {:?}", preset);

        // ── Environment
        if let Some(hdr) = p.hdr_handle {
            scene.environment.set_env_map(Some(hdr));
        } else {
            scene.environment.set_env_map(None::<TextureHandle>);
        }
        scene.environment.set_intensity(p.env_intensity);
        scene.environment.set_ambient_light(p.ambient_light);

        // ── Background
        match &p.background {
            PresetBackground::Color(c) => {
                scene.background.set_mode(BackgroundMode::Color(*c));
            }
            PresetBackground::Gradient { top, bottom } => {
                scene.background.set_mode(BackgroundMode::Gradient {
                    top: *top,
                    bottom: *bottom,
                });
            }
            PresetBackground::Skybox {
                intensity,
                fallback_top,
                fallback_bottom,
            } => {
                if let Some(hdr) = p.hdr_handle {
                    scene
                        .background
                        .set_mode(BackgroundMode::equirectangular(hdr, *intensity));
                } else {
                    scene.background.set_mode(BackgroundMode::Gradient {
                        top: *fallback_top,
                        bottom: *fallback_bottom,
                    });
                }
            }
        }

        // ── Three-point lighting
        let apply_light = |scene: &mut Scene, handle: NodeHandle, cfg: &LightConfig| {
            if let Some(light) = scene.get_light_mut(handle) {
                light.color = cfg.color;
                light.intensity = cfg.intensity;
                light.cast_shadows = cfg.cast_shadows;
            }
            if cfg.intensity > 0.0 {
                if let Some(node) = scene.get_node_mut(handle) {
                    node.transform.position = cfg.position;
                    node.transform.look_at(Vec3::ZERO, Vec3::Y);
                }
            }
        };
        apply_light(scene, self.key_light_id, &p.key_light);
        apply_light(scene, self.fill_light_id, &p.fill_light);
        apply_light(scene, self.rim_light_id, &p.rim_light);

        // ── Tone mapping
        scene.tone_mapping.set_mode(p.tone_mapping_mode);
        scene.tone_mapping.set_exposure(p.exposure);
        scene.tone_mapping.set_contrast(p.contrast);
        scene.tone_mapping.set_saturation(p.saturation);
        scene
            .tone_mapping
            .set_vignette_intensity(p.vignette_intensity);
        scene
            .tone_mapping
            .set_vignette_smoothness(p.vignette_smoothness);
        scene.tone_mapping.set_vignette_color(p.vignette_color);
        scene
            .tone_mapping
            .set_chromatic_aberration(p.chromatic_aberration);
        scene.tone_mapping.set_film_grain(p.film_grain);
        scene.tone_mapping.set_lut_texture(p.lut_handle);
        scene.tone_mapping.set_lut_contribution(p.lut_contribution);

        // ── Bloom
        scene.bloom.set_enabled(p.bloom_enabled);
        scene.bloom.set_strength(p.bloom_strength);
        scene.bloom.set_radius(p.bloom_radius);

        // ── SSAO
        scene.ssao.set_enabled(p.ssao_enabled);
        if p.ssao_enabled {
            scene.ssao.set_radius(p.ssao_radius);
            scene.ssao.set_intensity(p.ssao_intensity);
        }

        scene.screen_space.enable_sss = p.ssss_enabled;
    }

    /// Returns `true` when the model and the default preset's critical resources
    /// have all finished loading — i.e. the initial visual is ready to display.
    fn check_initial_ready(&self) -> bool {
        if !self.model_loaded {
            return false;
        }
        if let Some(p) = self.presets.get(&DEFAULT_PRESET) {
            if p.hdr_filename.is_some() && p.hdr_handle.is_none() {
                return false;
            }
            if p.lut_filename.is_some() && p.lut_handle.is_none() {
                return false;
            }
        }
        true
    }

    /// Instantiates the loaded glTF prefab, auto-plays the first animation,
    /// and repositions the camera to frame the model.
    fn instantiate_and_focus(
        &mut self,
        scene: &mut Scene,
        assets: &AssetServer,
        prefab: &SharedPrefab,
    ) {
        let root_node = scene.instantiate(prefab);
        scene.update_subtree(root_node);

        // Auto-play the first animation (if any).
        if let Some(mixer) = scene.animation_mixers.get_mut(root_node) {
            let anims = mixer.list_animations();
            if let Some(first) = anims.first() {
                log::info!("Auto-playing animation: {}", first);
                mixer.play(first);
            }
        }

        let skin_profile =
            myth::resources::screen_space::SssProfile::new(Vec3::new(0.85, 0.25, 0.15), 0.15);

        if let Some(model_node) = scene.find_node_by_name("Object_9") {
            if let Some(mesh) = scene.meshes.get(model_node) {
                if let Some(material) = assets.materials.get(mesh.material) {
                    if let Some(pbr) = material.as_physical() {
                        if let Some(new_id) = assets.sss_registry.write().add(&skin_profile) {
                            pbr.set_sss_id(Some(new_id));
                        }
                    }
                }
            }
        }
    }
}

// ── WASM Helpers ────────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn hide_loading_overlay() {
    use web_sys::window;
    if let Some(win) = window() {
        if let Some(doc) = win.document() {
            if let Some(el) = doc.get_element_by_id("loading-overlay") {
                let _ = el.class_list().add_1("fade-out");
            }
        }
    }
}

// ── Platform Helpers ────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn execute_future<F: std::future::Future<Output = ()> + Send + 'static>(f: F) {
    tokio::spawn(f);
}

#[cfg(target_arch = "wasm32")]
fn execute_future<F: std::future::Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}

// ── Entry Points ────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn main() -> myth::Result<()> {
    env_logger::init();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _enter = rt.enter();

    App::new()
        .with_title("Myth Engine — Showcase")
        .with_settings(RendererSettings {
            ..Default::default()
        })
        .run::<ShowcaseApp>()
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).unwrap();

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
