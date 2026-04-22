//! glTF Viewer Example (Based on App Module)
//!
//! An interactive glTF/glb file viewer demonstrating how to integrate egui as an external plugin.
//! In fact, this example currently has some privileges because to support the Inspector,
//! it directly accesses the engine's internal data structures.
//! In the future, as the engine evolves, it may become a prototype for the engine's editor/debugger.
//!
//! Features:
//! - Load local glTF/glb files via file dialog
//! - Support loading KhronosGroup glTF-Sample-Assets remote resources
//! - Animation playback control (play/pause, speed adjustment)
//! - Scene Inspector (node tree, material, texture viewer)
//! - Camera orbit control
//! - FPS display
//!
//! Run: cargo run -p gltf_viewer --release
//!
//! # Architecture Notes
//! This example demonstrates the "UI as a Plugin" pattern:
//! - `UiPass` implements `PassNode` trait, can be injected into RDG
//! - Inject UI Pass via `compose_frame()` hook into RDG
//! - Engine core does not depend on egui at all

use std::any::Any;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

use egui::CollapsingHeader;
#[cfg(target_arch = "wasm32")]
use std::sync::Mutex;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use myth::RenderableMaterialTrait;
use myth::assets::SharedPrefab;
use myth::prelude::*;
use myth::renderer::core::{BindingResource, ResourceBuilder};
use myth::resources::texture::TextureSource;
use myth_dev_utils::{FpsCounter, UiPass, UiPassNode};

// winit types needed for on_event downcasting (advanced egui integration)
use winit::event::WindowEvent;
use winit::keyboard::PhysicalKey;

/// Global sender for the WASM JS-bridge file drop handler.
///
/// JavaScript calls [`receive_dropped_file`] which pushes raw bytes here.
/// The main loop drains `wasm_drop_rx` each frame and spawns async load tasks.
#[cfg(target_arch = "wasm32")]
static DROP_SENDER: std::sync::LazyLock<Mutex<Option<Sender<(String, Vec<u8>)>>>> =
    std::sync::LazyLock::new(|| Mutex::new(None));

/// Receives a file dropped onto the browser window from the JS bridge.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn receive_dropped_file(name: String, data: Vec<u8>) {
    if let Ok(guard) = DROP_SENDER.lock() {
        if let Some(sender) = &*guard {
            let _ = sender.send((name, data));
            log::info!("Received dropped file from JS bridge");
        }
    }
}

// ============================================================================
// Remote Model Resources
// ============================================================================

const BASE_URL: &str = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main";
const MODEL_LIST_URL: &str = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/model-index.json";

/// Remote model description
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default)]
    pub screenshot: Option<String>,
    #[serde(default)]
    pub variants: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Loading state
#[derive(Debug, Clone, PartialEq)]
enum LoadingState {
    Idle,
    LoadingList,
    LoadingModel(String),
    Error(String),
}

/// Model source type
#[derive(Debug, Clone)]
enum ModelSource {
    #[cfg(not(target_arch = "wasm32"))]
    Local(PathBuf),
    /// WASM: file data loaded from browser (name, bytes)
    #[cfg(target_arch = "wasm32")]
    Local(String, Vec<u8>),

    Remote(String), // URL
}

// ============================================================================
// Skybox / Background Mode
// ============================================================================

/// Background mode for skybox selection in the viewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkyboxMode {
    /// Default: dark clear color, no skybox
    Off,
    /// Solid color background
    SolidColor,
    /// Vertical gradient background
    Gradient,
    /// Equirectangular HDR panorama as skybox
    Equirectangular,
}

impl SkyboxMode {
    fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::SolidColor => "Solid Color",
            Self::Gradient => "Gradient",
            Self::Equirectangular => "Equirectangular HDR",
        }
    }

    fn all() -> &'static [Self] {
        &[
            Self::Off,
            Self::SolidColor,
            Self::Gradient,
            Self::Equirectangular,
        ]
    }
}

// ============================================================================
// Inspector Related Data Structures
// ============================================================================

/// Target type selectable in Inspector
#[derive(Debug, Clone, PartialEq)]
enum InspectorTarget {
    Node(NodeHandle),
    Material(MaterialHandle),
    Texture(TextureHandle),
}

/// Collected material information
#[derive(Debug, Clone)]
struct MaterialInfo {
    pub handle: MaterialHandle,
    pub name: String,
}

/// Collected texture information
#[derive(Debug, Clone)]
struct TextureInfo {
    pub handle: TextureHandle,
    pub name: String,
}

// ============================================================================
// Anti-Aliasing Inspector Cache
// ============================================================================

/// Lightweight discriminant for the AA mode combo-box (no payload).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum AaModeType {
    None,
    FXAA,
    MSAA,
    MSAA_FXAA,
    TAA,
    TAA_FXAA,
}

impl AaModeType {
    const fn label(self) -> &'static str {
        match self {
            Self::None => "Off",
            Self::FXAA => "FXAA",
            Self::MSAA => "MSAA 4x",
            Self::MSAA_FXAA => "MSAA+FXAA",
            Self::TAA => "TAA",
            Self::TAA_FXAA => "TAA+FXAA",
        }
    }

    /// All variants, in display order.
    const ALL: &[Self] = &[
        Self::None,
        Self::FXAA,
        Self::MSAA,
        Self::MSAA_FXAA,
        Self::TAA,
        Self::TAA_FXAA,
    ];

    /// Post-process variants that require the HighFidelity path.
    fn requires_post_processing(self) -> bool {
        matches!(
            self,
            Self::FXAA | Self::MSAA_FXAA | Self::TAA | Self::TAA_FXAA
        )
    }
}

/// Caches the per-technique settings so that switching modes in the UI does
/// not lose the user's tuned parameters.
struct InspectorAaCache {
    current: AaModeType,
    msaa_samples: u32,
    fxaa: FxaaSettings,
    taa: TaaSettings,
}

impl Default for InspectorAaCache {
    fn default() -> Self {
        Self {
            current: AaModeType::TAA_FXAA,
            msaa_samples: 4,
            fxaa: FxaaSettings::default(),
            taa: TaaSettings::default(),
        }
    }
}

impl InspectorAaCache {
    /// Assemble the full [`AntiAliasingMode`] from the cached parameters.
    fn build_mode(&self) -> AntiAliasingMode {
        match self.current {
            AaModeType::None => AntiAliasingMode::None,
            AaModeType::FXAA => AntiAliasingMode::FXAA(self.fxaa.clone()),
            AaModeType::MSAA => AntiAliasingMode::MSAA(self.msaa_samples),
            AaModeType::MSAA_FXAA => {
                AntiAliasingMode::MSAA_FXAA(self.msaa_samples, self.fxaa.clone())
            }
            AaModeType::TAA => AntiAliasingMode::TAA(self.taa.clone()),
            AaModeType::TAA_FXAA => AntiAliasingMode::TAA_FXAA(self.taa.clone(), self.fxaa.clone()),
        }
    }
}

// ============================================================================
// glTF Viewer Main Structure
// ============================================================================

/// glTF Viewer application state
struct GltfViewer {
    /// UI Pass (egui rendering)
    ui_pass: UiPass,

    /// Currently loaded model root node
    gltf_node: Option<NodeHandle>,
    /// List of available animations
    animations: Vec<String>,
    /// Currently selected animation index
    current_animation: usize,
    /// Whether animation is playing
    is_playing: bool,
    /// Animation playback speed
    playback_speed: f32,
    /// Orbit controller
    controls: OrbitControls,
    /// FPS counter
    fps_counter: FpsCounter,
    /// Current FPS
    current_fps: f32,
    /// Model file path or name (for display)
    model_name: Option<String>,

    // === Unified Async Event Channel ===
    /// Sender for all async events (cloned into background tasks).
    event_tx: Sender<ViewerEvent>,
    /// Receiver for all async events (polled each frame).
    event_rx: Receiver<ViewerEvent>,

    /// Handle of the model currently being loaded via fire-and-forget.
    /// Polled each frame until the prefab is ready or fails.
    pending_prefab: Option<(String, PrefabHandle)>,

    /// WASM-only receiver for raw file bytes from the JS bridge drop handler.
    #[cfg(target_arch = "wasm32")]
    wasm_drop_rx: Receiver<(String, Vec<u8>)>,

    // === Remote Model Related ===
    /// Remote model list
    model_list: Vec<ModelInfo>,
    /// Currently selected remote model index
    selected_model_index: usize,
    /// Loading state
    loading_state: LoadingState,
    /// Preferred glTF variants (by priority)
    preferred_variants: Vec<&'static str>,

    // === Inspector Related ===
    /// Whether to show Inspector
    show_inspector: bool,
    /// Current Inspector selected target
    inspector_target: Option<InspectorTarget>,
    /// Collected material list
    inspector_materials: Vec<MaterialInfo>,
    /// Collected texture list
    inspector_textures: Vec<TextureInfo>,

    // === Render Settings ===
    /// IBL toggle
    ibl_enabled: bool,

    light_node: NodeHandle,
    /// Render path (cached from renderer settings)
    render_path: RenderPath,

    /// VSync toggle (cached from renderer settings)
    vsync: bool,
    /// Anisotropic filtering level (cached from renderer settings)
    anisotropy_clamp: u16,
    /// Anti-aliasing inspector cache (preserves per-technique settings)
    aa_cache: InspectorAaCache,

    vignette_breathing: bool,

    show_ui: bool,

    // === Skybox / Background ===
    /// Current background mode
    skybox_mode: SkyboxMode,
    /// Background solid color [r, g, b, a]
    bg_color: [f32; 4],
    /// Gradient top color
    gradient_top: [f32; 4],
    /// Gradient bottom color
    gradient_bottom: [f32; 4],
    /// Environment texture handle (used for both IBL and Equirectangular skybox)
    env_texture: Option<TextureHandle>,
    /// Skybox intensity multiplier
    skybox_intensity: f32,
    /// Skybox rotation in degrees
    skybox_rotation: f32,
    /// Name of the loaded custom skybox file
    skybox_file_name: Option<String>,

    // === SSS Profiles ===
    sss_profiles: Vec<(
        myth::resources::screen_space::FeatureId,
        String,
        myth::resources::screen_space::SssProfile,
    )>,

    // === Debug View (compile-time gated) ===
    #[cfg(feature = "debug_view")]
    debug_view: myth::prelude::DebugViewSettings,
}

/// Async events delivered from background tasks to the main loop.
///
/// Merges what were previously five separate channels (model-list fetch,
/// prefab load, file dialog, skybox HDR, LUT) into a single typed channel.
enum ViewerEvent {
    /// Remote model list fetched.
    ModelList(Result<Vec<ModelInfo>, String>),
    /// A model has been submitted for fire-and-forget loading via [`AssetServer::load_gltf`].
    /// The main thread stores the handle and polls until the prefab is ready.
    #[cfg(not(target_arch = "wasm32"))]
    ModelLoading { name: String, handle: PrefabHandle },
    /// A prefab was loaded from in-memory bytes (WASM file-drop / file-picker).
    #[cfg(target_arch = "wasm32")]
    PrefabLoaded { name: String, prefab: SharedPrefab },
    /// An environment texture handle is ready to be applied.
    EnvTextureReady { name: String, handle: TextureHandle },
    /// A LUT texture handle is ready to be applied.
    LutTextureReady { name: String, handle: TextureHandle },
}

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

impl AppHandler for GltfViewer {
    fn init(engine: &mut Engine, window: &dyn Window) -> Self {
        // 1. Create UI Pass
        let wgpu_ctx = engine
            .renderer
            .wgpu_ctx()
            .expect("Renderer not initialized");
        // Downcast to winit::Window for egui-winit integration
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");
        let ui_pass = UiPass::new(&wgpu_ctx.device, wgpu_ctx.surface_view_format, winit_window);

        let scene = engine.scene_manager.create_active();

        // Load default HDR environment map (fire-and-forget)
        let env_handle = engine.assets.load_texture(
            format!("{}envs/royal_esplanade_2k.hdr.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            false,
        );
        scene.environment.set_env_map(Some(env_handle));
        scene.environment.set_intensity(3.0);

        // 3. Add directional light
        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 3.0);

        let light_node = scene.add_light(light);
        if let Some(node) = scene.get_node_mut(light_node) {
            node.transform.position = Vec3::new(1.0, 1.0, 1.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        // 4. Set up camera
        let mut camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);

        camera.set_aa_mode(AntiAliasingMode::TAA_FXAA(
            TaaSettings::default(),
            FxaaSettings::default(),
        ));

        let cam_node_id = scene.add_camera(camera);
        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 1.0, 5.0);
            node.transform.look_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        }
        scene.active_camera = Some(cam_node_id);

        // 5. Create unified event channel
        let (event_tx, event_rx) = channel();

        #[cfg(target_arch = "wasm32")]
        let (wasm_drop_tx, wasm_drop_rx) = channel();

        #[cfg(target_arch = "wasm32")]
        {
            if let Ok(mut guard) = DROP_SENDER.lock() {
                *guard = Some(wasm_drop_tx);
            }
        }

        let mut viewer = Self {
            ui_pass,
            gltf_node: None,
            animations: Vec::new(),
            current_animation: 0,
            is_playing: true,
            playback_speed: 1.0,
            controls: OrbitControls::new(Vec3::new(0.0, 1.0, 5.0), Vec3::new(0.0, 1.0, 0.0)),
            fps_counter: FpsCounter::new(),
            current_fps: 0.0,
            model_name: None,

            // Async events
            event_tx,
            event_rx,
            pending_prefab: None,
            #[cfg(target_arch = "wasm32")]
            wasm_drop_rx,

            // Remote models
            model_list: Vec::new(),
            selected_model_index: 0,
            loading_state: LoadingState::Idle,
            preferred_variants: vec!["glTF-Binary", "glTF-Embedded", "glTF"],

            // Inspector
            show_inspector: false,
            inspector_target: None,
            inspector_materials: Vec::new(),
            inspector_textures: Vec::new(),

            // Render settings
            ibl_enabled: true,
            light_node: light_node,
            render_path: RenderPath::default(),
            vsync: false, // examples default to vsync off
            anisotropy_clamp: 1,
            aa_cache: InspectorAaCache::default(),

            vignette_breathing: false,

            show_ui: true,

            // Skybox / Background
            skybox_mode: SkyboxMode::Off,
            bg_color: [0.03, 0.03, 0.03, 1.0],
            gradient_top: [0.05, 0.05, 0.25, 1.0],
            gradient_bottom: [0.7, 0.45, 0.2, 1.0],
            env_texture: Some(env_handle),
            skybox_intensity: 1.0,
            skybox_rotation: 0.0,
            skybox_file_name: None,

            sss_profiles: Vec::new(),

            #[cfg(feature = "debug_view")]
            debug_view: myth::prelude::DebugViewSettings::default(),
        };

        // 6. Fetch remote model list in the background
        viewer.fetch_model_list();

        viewer
    }

    fn on_event(&mut self, _engine: &mut Engine, window: &dyn Window, event: &dyn Any) -> bool {
        // Downcast the event to winit::WindowEvent
        let Some(event) = event.downcast_ref::<WindowEvent>() else {
            return false;
        };

        // Tab 键切换 UI 显示
        if let WindowEvent::KeyboardInput { event, .. } = event {
            let PhysicalKey::Code(code) = event.physical_key else {
                return false;
            };
            if code == winit::keyboard::KeyCode::Tab
                && event.state == winit::event::ElementState::Pressed
            {
                self.show_ui = !self.show_ui;
                return true;
            }
        }

        // Downcast window for egui-winit integration
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");

        // UI 优先处理事件
        if self.ui_pass.handle_input(winit_window, event) {
            return true;
        }

        // Handle window resize
        if let WindowEvent::Resized(size) = event {
            let scale_factor = window.scale_factor();
            self.ui_pass.resize(size.width, size.height, scale_factor);
        }

        false
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let time = engine.time();
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // 0. Process async load results
        self.process_load_results(scene, &engine.assets);

        // 1. 更新 FPS
        if let Some(fps) = self.fps_counter.update() {
            self.current_fps = fps;

            let title = if let Some(name) = &self.model_name {
                format!("glTF Viewer - {} | FPS: {:.0}", name, self.current_fps)
            } else {
                format!("glTF Viewer | FPS: {:.0}", self.current_fps)
            };
            window.set_title(&title);
        }

        // 2. 更新动画播放速度
        if let Some(gltf_node) = self.gltf_node
            && let Some(mixer) = scene.animation_mixers.get_mut(gltf_node)
        {
            mixer.time_scale = self.playback_speed;
        }

        // 3. 相机控制
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        if self.vignette_breathing {
            let bpm = 30.0;
            let period = 60.0 / bpm;
            let t = time % period;

            let pulse = (-t * 3.0).exp();
            let vignette_intensity = 0.0 + 0.5 * pulse;
            scene
                .tone_mapping
                .set_vignette_intensity(vignette_intensity);
        }

        // 4. 构建 UI (requires winit window for egui-winit integration)
        if self.show_ui {
            let winit_window = window
                .as_any()
                .downcast_ref::<winit::window::Window>()
                .expect("Expected winit window backend");
            self.ui_pass.begin_frame(winit_window);
            let egui_ctx = self.ui_pass.context().clone();
            self.handle_drag_and_drop(&egui_ctx, engine.assets.clone());
            self.render_ui(engine);
            self.ui_pass.end_frame(winit_window);
        }
    }

    fn render(&mut self, engine: &mut Engine, _window: &dyn Window) {
        use myth::renderer::graph::core::{GraphBlackboard, HookStage};
        let Some(composer) = engine.compose_frame() else {
            return;
        };

        if self.show_ui {
            // Resolve pending engine texture registrations before the RDG
            // prepare phase (which no longer has access to ResourceManager).
            self.ui_pass
                .resolve_textures(composer.device(), composer.resource_manager());

            let ui_pass_ref = &mut self.ui_pass;

            composer
                .add_custom_pass(HookStage::AfterPostProcess, move |rdg, bb| {
                    let new_surface = rdg.add_pass("UI_Pass", |builder| {
                        let out = builder.mutate_texture(bb.surface_out, "Surface_With_UI");
                        let node = UiPassNode {
                            pass: ui_pass_ref,
                            target_tex: out,
                        };
                        (node, out)
                    });

                    GraphBlackboard {
                        surface_out: new_surface,
                        ..bb
                    }
                })
                .render();
        } else {
            composer.render();
        }
    }
}

impl GltfViewer {
    // ========================================================================
    // Skybox / Background
    // ========================================================================

    /// Applies the current skybox/background mode to the scene.
    ///
    /// This is a static-style helper to avoid borrow conflicts inside egui closures:
    /// all parameters are passed explicitly instead of through `&mut self`.
    fn apply_skybox(
        scene: &mut Scene,
        mode: SkyboxMode,
        bg_color: &[f32; 4],
        gradient_top: &[f32; 4],
        gradient_bottom: &[f32; 4],
        texture: Option<TextureHandle>,
        intensity: f32,
        rotation_deg: f32,
    ) {
        let bg_mode = match mode {
            SkyboxMode::Off => BackgroundMode::Color(Vec4::new(0.03, 0.03, 0.03, 1.0)),
            SkyboxMode::SolidColor => BackgroundMode::Color(Vec4::from_array(*bg_color)),
            SkyboxMode::Gradient => BackgroundMode::Gradient {
                top: Vec4::from_array(*gradient_top),
                bottom: Vec4::from_array(*gradient_bottom),
            },
            SkyboxMode::Equirectangular => {
                if let Some(tex) = texture {
                    BackgroundMode::equirectangular(tex, intensity)
                } else {
                    // No texture available — fall back to dark color
                    BackgroundMode::Color(Vec4::new(0.03, 0.03, 0.03, 1.0))
                }
            }
        };
        scene.background.set_mode(bg_mode);

        // Apply rotation for texture modes
        if mode == SkyboxMode::Equirectangular {
            scene.background.set_rotation(rotation_deg.to_radians());
        }
    }

    // ========================================================================
    // Model Loading
    // ========================================================================

    /// Fetches the remote model list in the background.
    fn fetch_model_list(&mut self) {
        self.loading_state = LoadingState::LoadingList;
        let tx = self.event_tx.clone();

        execute_future(async move {
            let result = fetch_remote_model_list().await;
            let _ = tx.send(ViewerEvent::ModelList(result));
        });
    }

    /// Processes all pending async events and polls in-flight model loads.
    fn process_load_results(&mut self, scene: &mut Scene, assets: &AssetServer) {
        // --- Drain the unified event channel ---
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                ViewerEvent::ModelList(Ok(list)) => {
                    log::info!("Loaded {} models from remote", list.len());
                    self.model_list = list;
                    self.loading_state = LoadingState::Idle;
                }
                ViewerEvent::ModelList(Err(e)) => {
                    log::error!("Failed to load model list: {}", e);
                    self.loading_state = LoadingState::Error(e);
                }
                #[cfg(not(target_arch = "wasm32"))]
                ViewerEvent::ModelLoading { name, handle } => {
                    self.loading_state = LoadingState::LoadingModel(name.clone());
                    self.pending_prefab = Some((name, handle));
                }
                #[cfg(target_arch = "wasm32")]
                ViewerEvent::PrefabLoaded { name, prefab } => {
                    self.instantiate_prefab(scene, assets, name, prefab);
                }
                ViewerEvent::EnvTextureReady { name, handle } => {
                    log::info!("Environment texture ready: {}", name);
                    self.env_texture = Some(handle);
                    self.skybox_file_name = Some(name);
                    scene.environment.set_env_map(Some(handle));
                    if self.skybox_mode == SkyboxMode::Equirectangular {
                        Self::apply_skybox(
                            scene,
                            self.skybox_mode,
                            &self.bg_color,
                            &self.gradient_top,
                            &self.gradient_bottom,
                            self.env_texture,
                            self.skybox_intensity,
                            self.skybox_rotation,
                        );
                    }
                }
                ViewerEvent::LutTextureReady { name, handle } => {
                    log::info!("LUT texture ready: {}", name);
                    scene.tone_mapping.set_lut_texture(Some(handle));
                }
            }
        }

        // --- Poll pending fire-and-forget model load ---
        if let Some((ref name, handle)) = self.pending_prefab {
            if let Some(prefab_arc) = assets.prefabs.get(handle) {
                let name = name.clone();
                let prefab = (*prefab_arc).clone();
                self.pending_prefab = None;
                self.instantiate_prefab(scene, assets, name, prefab);
            } else if let Some(err) = assets.prefabs.get_error(handle) {
                log::error!("Failed to load model '{}': {}", name, err);
                self.loading_state = LoadingState::Error(err);
                self.pending_prefab = None;
            }
        }

        // --- WASM: process raw file bytes from JS bridge drop handler ---
        #[cfg(target_arch = "wasm32")]
        while let Ok((name, data)) = self.wasm_drop_rx.try_recv() {
            self.load_model_from_bytes(name, data, assets);
        }
    }

    /// Instantiates a loaded prefab into the scene, replacing any previous model.
    fn instantiate_prefab(
        &mut self,
        scene: &mut Scene,
        assets: &AssetServer,
        display_name: String,
        prefab: SharedPrefab,
    ) {
        // Clear previous model
        if let Some(gltf_node) = self.gltf_node {
            scene.remove_node(gltf_node);
        }
        self.gltf_node = None;
        self.animations.clear();
        self.inspector_materials.clear();
        self.inspector_textures.clear();
        self.inspector_target = None;

        // Instantiate new model
        let gltf_node = scene.instantiate(&prefab);

        self.gltf_node = Some(gltf_node);
        self.model_name = Some(display_name.clone());
        self.current_animation = 0;

        // Auto-play the first animation clip if available
        if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
            self.animations = mixer.list_animations();
            if let Some(clip_name) = self.animations.first() {
                mixer.play(clip_name);
            }
        }

        // Propagate transforms through the subtree
        scene.update_subtree(gltf_node);

        // Fit camera to the model bounding box
        if let Some(bbox) = scene.get_bbox_of_node(gltf_node, assets) {
            let center = bbox.center();
            let radius = bbox.size().length() * 0.5;
            if let Some((_transform, camera)) = scene.query_main_camera_bundle() {
                camera.set_near(radius / 100.0);
                self.controls.set_target(center);
                // let distance = radius / (camera.fov / 2.0).tan();
                // self.controls.set_position(center + Vec3::new(0.0, radius, distance * 1.25));
                self.controls
                    .set_position(center + Vec3::new(0.0, radius, radius * 2.5));
            }
        }

        // Collect Inspector data
        self.collect_inspector_targets(scene, assets, gltf_node);

        self.loading_state = LoadingState::Idle;
        log::info!("Instantiated model: {}", display_name);
    }

    /// Loads a model from a URL or local path using the engine's fire-and-forget API.
    ///
    /// For path/URL sources the [`AssetServer`] handles I/O, parsing and
    /// deduplication internally.  The returned [`PrefabHandle`] is stored in
    /// `pending_prefab` and polled each frame.
    fn load_model(&mut self, source: ModelSource, assets: &AssetServer) {
        match source {
            ModelSource::Remote(url) => {
                let display_name = url.rsplit('/').next().unwrap_or("Remote Model").to_string();
                self.loading_state = LoadingState::LoadingModel(display_name.clone());
                let handle = assets.load_gltf(url);
                self.pending_prefab = Some((display_name, handle));
            }

            #[cfg(not(target_arch = "wasm32"))]
            ModelSource::Local(path) => {
                let display_name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "Unknown".to_string());
                self.loading_state = LoadingState::LoadingModel(display_name.clone());
                let handle = assets.load_gltf(path.to_string_lossy().to_string());
                self.pending_prefab = Some((display_name, handle));
            }

            #[cfg(target_arch = "wasm32")]
            ModelSource::Local(name, data) => {
                self.load_model_from_bytes(name, data, assets);
            }
        }
    }

    /// Loads a model from in-memory bytes (WASM file-drop / file-picker).
    ///
    /// Because byte-based loading has no URI for deduplication, this path
    /// still uses an async task with [`GltfLoader::load_from_bytes`].
    #[cfg(target_arch = "wasm32")]
    fn load_model_from_bytes(&mut self, name: String, data: Vec<u8>, assets: &AssetServer) {
        self.loading_state = LoadingState::LoadingModel(name.clone());
        let tx = self.event_tx.clone();
        let assets = assets.clone();

        execute_future(async move {
            match GltfLoader::load_from_bytes(data, assets).await {
                Ok(prefab) => {
                    let _ = tx.send(ViewerEvent::PrefabLoaded { name, prefab });
                }
                Err(e) => {
                    log::error!("Failed to load model from bytes: {}", e);
                }
            }
        });
    }

    /// Builds a download URL for the given remote model index.
    fn build_remote_url(&self, model_index: usize) -> Option<String> {
        let model = self.model_list.get(model_index)?;

        for variant in &self.preferred_variants {
            if let Some(filename) = model.variants.get(*variant) {
                return Some(format!(
                    "{}/Models/{}/{}/{}",
                    BASE_URL, model.name, variant, filename
                ));
            }
        }

        None
    }

    /// Handles file drag-and-drop onto the viewer (Native & WASM).
    fn handle_drag_and_drop(&mut self, ctx: &egui::Context, assets: AssetServer) {
        // Visual hint while a file hovers over the window
        if !ctx.input(|i| i.raw.hovered_files.is_empty()) {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("file_drop_overlay"),
            ));
            let screen_rect = ctx.content_rect();

            // 绘制半透明覆盖层提示用户松手
            painter.rect_filled(screen_rect, 0.0, egui::Color32::from_black_alpha(100));
            painter.text(
                screen_rect.center(),
                egui::Align2::CENTER_CENTER,
                "📂 Drop glTF/GLB file here",
                egui::FontId::proportional(32.0),
                egui::Color32::WHITE,
            );
        }

        // 2. 检查是否有文件被放下
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                // 取最后一个拖入的文件（如果用户一次拖多个，通常只加载一个）
                if let Some(file) = i.raw.dropped_files.last() {
                    self.process_dropped_file(file, assets);
                }
            }
        });
    }

    /// Processes a file dropped onto the viewer window.
    fn process_dropped_file(&mut self, file: &egui::DroppedFile, assets: AssetServer) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = &file.path {
            log::info!("File dropped (Native): {:?}", path);
            self.load_model(ModelSource::Local(path.clone()), &assets);
        }

        #[cfg(target_arch = "wasm32")]
        if let Some(bytes) = &file.bytes {
            log::info!("File dropped (WASM): {}, {} bytes", file.name, bytes.len());
            self.load_model(
                ModelSource::Local(file.name.clone(), bytes.to_vec()),
                &assets,
            );
        } else {
            log::warn!(
                "Dropped file has no data. Native path: {:?}, Bytes present: {}",
                file.path,
                file.bytes.is_some()
            );
        }
    }

    // ========================================================================
    // Inspector 数据收集
    // ========================================================================

    /// 收集场景中的材质和纹理信息
    fn collect_inspector_targets(&mut self, scene: &Scene, assets: &AssetServer, root: NodeHandle) {
        self.inspector_materials.clear();
        self.inspector_textures.clear();

        let mut visited_materials = std::collections::HashSet::new();
        let mut visited_textures = std::collections::HashSet::new();

        // 遍历所有节点
        let mut stack = vec![root];
        while let Some(node_handle) = stack.pop() {
            // 收集子节点
            if let Some(node) = scene.get_node(node_handle) {
                stack.extend(node.children().iter().cloned());
            }

            // 收集 Mesh 的材质
            if let Some(mesh) = scene.get_mesh(node_handle) {
                let mat_handle = mesh.material;

                if !visited_materials.contains(&mat_handle) {
                    visited_materials.insert(mat_handle);

                    let mat_name = assets
                        .materials
                        .get(mat_handle)
                        .and_then(|m| m.name.clone())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("Material_{:?}", mat_handle));

                    self.inspector_materials.push(MaterialInfo {
                        handle: mat_handle,
                        name: mat_name.clone(),
                    });

                    // 收集材质使用的纹理
                    if let Some(material) = assets.materials.get(mat_handle) {
                        self.collect_textures_from_material(
                            &material,
                            &mat_name,
                            assets,
                            &mut visited_textures,
                        );
                    }
                }
            }
        }
    }

    /// 从材质中收集纹理信息
    fn collect_textures_from_material(
        &mut self,
        material: &Material,
        mat_name: &str,
        assets: &AssetServer,
        visited: &mut std::collections::HashSet<TextureHandle>,
    ) {
        // 使用通用方式收集纹理：通过 visit_textures trait 方法
        let mut collected = Vec::new();
        material.as_renderable().visit_textures(&mut |tex_source| {
            if let myth::resources::texture::TextureSource::Asset(handle) = tex_source
                && !visited.contains(handle)
            {
                visited.insert(*handle);
                collected.push(*handle);
            }
        });

        for (i, tex_handle) in collected.into_iter().enumerate() {
            let texture_name = assets
                .textures
                .get(tex_handle)
                .and_then(|t| t.name.clone())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{}:texture_{}", mat_name, i));

            self.inspector_textures.push(TextureInfo {
                handle: tex_handle,
                name: texture_name,
            });
        }
    }

    // ========================================================================
    // UI 渲染
    // ========================================================================

    fn render_ui(&mut self, engine: &mut Engine) {
        let egui_ctx = self.ui_pass.context().clone();

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // 主控制面板
        self.render_control_panel(&egui_ctx, scene, &engine.assets, &mut engine.renderer);

        // Inspector 面板
        if self.show_inspector {
            self.render_inspector(&egui_ctx, scene, &engine.assets);
        }
    }

    /// 渲染主控制面板
    fn render_control_panel(
        &mut self,
        ctx: &egui::Context,
        scene: &mut Scene,
        assets: &AssetServer,
        renderer: &mut myth::Renderer,
    ) {
        egui::Window::new("Control Panel (Press Tab to Toggle)")
            .default_pos([10.0, 10.0])
            .default_size([320.0, 200.0])
            .show(ctx, |ui| {
                egui::ScrollArea::both()
                    .min_scrolled_height(600.0)
                    .show(ui, |ui| {
                        // ===== 远程模型加载 =====
                        CollapsingHeader::new("🌐 KhronosGroup glTF-Sample-Assets (Remote)")
                            .default_open(true)
                            .show(ui, |ui| {
                                let is_loading = matches!(
                                    self.loading_state,
                                    LoadingState::LoadingList | LoadingState::LoadingModel(_)
                                );

                                ui.add_enabled_ui(!is_loading, |ui| {
                                    ui.horizontal(|ui| {
                                        let model_names: Vec<_> = self
                                            .model_list
                                            .iter()
                                            .map(|m| m.name.as_str())
                                            .collect();
                                        ui.label("Model:");

                                        let combo =
                                            egui::ComboBox::from_id_salt("remote_model_selector")
                                                .width(180.0)
                                                .selected_text(
                                                    model_names
                                                        .get(self.selected_model_index)
                                                        .copied()
                                                        .unwrap_or("Select a model..."),
                                                );

                                        combo.show_ui(ui, |ui| {
                                            ui.set_min_width(250.0);
                                            for (i, name) in model_names.iter().enumerate() {
                                                ui.selectable_value(
                                                    &mut self.selected_model_index,
                                                    i,
                                                    *name,
                                                );
                                            }
                                        });

                                        if ui.button("Load").clicked()
                                            && let Some(url) =
                                                self.build_remote_url(self.selected_model_index)
                                        {
                                            self.load_model(ModelSource::Remote(url), assets);
                                        }
                                    });
                                });

                                // 显示加载状态
                                match &self.loading_state {
                                    LoadingState::LoadingList => {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.label("Loading model list...");
                                        });
                                    }
                                    LoadingState::LoadingModel(name) => {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.label(format!("Loading {}...", name));
                                        });
                                    }
                                    LoadingState::Error(e) => {
                                        ui.colored_label(
                                            egui::Color32::RED,
                                            format!("⚠ Error: {}", e),
                                        );
                                    }
                                    LoadingState::Idle => {}
                                }

                                ui.label(format!("{} models available", self.model_list.len()));
                            });

                        ui.separator();

                        // ===== Local File Loading =====
                        CollapsingHeader::new("📁 Local File")
                            .default_open(true)
                            .show(ui, |ui| {
                                if ui.button("Open glTF/glb File...").clicked() {
                                    let assets_clone = assets.clone();
                                    let tx = self.event_tx.clone();

                                    execute_future(async move {
                                        let file = rfd::AsyncFileDialog::new()
                                            .add_filter("glTF", &["gltf", "glb"])
                                            .pick_file()
                                            .await;

                                        if let Some(file_handle) = file {
                                            #[cfg(not(target_arch = "wasm32"))]
                                            {
                                                let path = file_handle
                                                    .path()
                                                    .to_string_lossy()
                                                    .to_string();
                                                let name = file_handle
                                                    .path()
                                                    .file_name()
                                                    .map(|n| n.to_string_lossy().to_string())
                                                    .unwrap_or_else(|| "Unknown".to_string());
                                                let handle = assets_clone.load_gltf(path);
                                                let _ = tx.send(ViewerEvent::ModelLoading {
                                                    name,
                                                    handle,
                                                });
                                            }

                                            #[cfg(target_arch = "wasm32")]
                                            {
                                                let data = file_handle.read().await;
                                                let name = file_handle.file_name();
                                                match GltfLoader::load_from_bytes(
                                                    data,
                                                    assets_clone,
                                                )
                                                .await
                                                {
                                                    Ok(prefab) => {
                                                        let _ =
                                                            tx.send(ViewerEvent::PrefabLoaded {
                                                                name,
                                                                prefab,
                                                            });
                                                    }
                                                    Err(e) => {
                                                        log::error!(
                                                            "Failed to load model from bytes: {}",
                                                            e
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    });
                                }

                                if let Some(name) = &self.model_name {
                                    ui.label(format!("Current: {}", name));
                                } else {
                                    ui.label("No model loaded");
                                }
                                #[cfg(target_arch = "wasm32")]
                                {
                                    ui.separator();
                                    ui.label("💡 Tip: GLB format recommended");
                                    ui.label("(contains all data in one file)");
                                }
                            });

                        ui.separator();

                        // ===== 天空盒/背景 =====
                        CollapsingHeader::new("🌌 Skybox / Background")
                            .default_open(true)
                            .show(ui, |ui| {
                                let mut bg_changed = false;

                                // --- 模式选择 ---
                                ui.horizontal(|ui| {
                                    ui.label("Mode:");
                                    egui::ComboBox::from_id_salt("skybox_mode_selector")
                                        .width(160.0)
                                        .selected_text(self.skybox_mode.label())
                                        .show_ui(ui, |ui| {
                                            for &mode in SkyboxMode::all() {
                                                if ui
                                                    .selectable_value(
                                                        &mut self.skybox_mode,
                                                        mode,
                                                        mode.label(),
                                                    )
                                                    .changed()
                                                {
                                                    bg_changed = true;
                                                }
                                            }
                                        });
                                });

                                // --- 模式相关参数 ---
                                match self.skybox_mode {
                                    SkyboxMode::Off => {}
                                    SkyboxMode::SolidColor => {
                                        ui.horizontal(|ui| {
                                            ui.label("Color:");
                                            if ui
                                                .color_edit_button_rgba_unmultiplied(
                                                    &mut self.bg_color,
                                                )
                                                .changed()
                                            {
                                                bg_changed = true;
                                            }
                                        });
                                    }
                                    SkyboxMode::Gradient => {
                                        ui.horizontal(|ui| {
                                            ui.label("Top:");
                                            if ui
                                                .color_edit_button_rgba_unmultiplied(
                                                    &mut self.gradient_top,
                                                )
                                                .changed()
                                            {
                                                bg_changed = true;
                                            }
                                        });
                                        ui.horizontal(|ui| {
                                            ui.label("Bottom:");
                                            if ui
                                                .color_edit_button_rgba_unmultiplied(
                                                    &mut self.gradient_bottom,
                                                )
                                                .changed()
                                            {
                                                bg_changed = true;
                                            }
                                        });
                                    }
                                    SkyboxMode::Equirectangular => {
                                        // 显示当前纹理状态
                                        if let Some(name) = &self.skybox_file_name {
                                            ui.label(format!("Env: {}", name));
                                        } else if self.env_texture.is_some() {
                                            ui.label("(Using default env map)");
                                        } else {
                                            ui.colored_label(
                                                egui::Color32::YELLOW,
                                                "⚠ No HDR texture loaded",
                                            );
                                        }

                                        // 亮度滑块
                                        ui.horizontal(|ui| {
                                            ui.label("Intensity:");
                                            if ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut self.skybox_intensity,
                                                        0.1..=5.0,
                                                    )
                                                    .step_by(0.1)
                                                    .logarithmic(true),
                                                )
                                                .changed()
                                            {
                                                scene
                                                    .background
                                                    .set_intensity(self.skybox_intensity);
                                            }
                                        });

                                        // 旋转滑块
                                        ui.horizontal(|ui| {
                                            ui.label("Rotation:");
                                            if ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut self.skybox_rotation,
                                                        0.0..=360.0,
                                                    )
                                                    .step_by(1.0)
                                                    .suffix("°"),
                                                )
                                                .changed()
                                            {
                                                let radians = self.skybox_rotation.to_radians();
                                                scene.background.set_rotation(radians);
                                                scene.environment.rotation = radians;
                                            }
                                        });
                                    }
                                }

                                // 应用模式变更
                                if bg_changed {
                                    Self::apply_skybox(
                                        scene,
                                        self.skybox_mode,
                                        &self.bg_color,
                                        &self.gradient_top,
                                        &self.gradient_bottom,
                                        self.env_texture,
                                        self.skybox_intensity,
                                        self.skybox_rotation,
                                    );
                                }

                                ui.separator();
                                // --- IBL 环境贴图 ---
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut self.ibl_enabled, "IBL").changed() {
                                        scene.environment.set_intensity(if self.ibl_enabled {
                                            3.0
                                        } else {
                                            0.0
                                        });
                                    }

                                    if self.ibl_enabled {
                                        ui.add(
                                            egui::Slider::new(
                                                &mut scene.environment.intensity,
                                                0.1..=5.0,
                                            )
                                            .step_by(0.1)
                                            .logarithmic(true),
                                        );
                                    }

                                    // --- Load HDR file button (updates both IBL and skybox) ---
                                    if ui.button("📂 Load HDR...").clicked() {
                                        let tx = self.event_tx.clone();
                                        let assets_clone = assets.clone();
                                        execute_future(async move {
                                            let file = rfd::AsyncFileDialog::new()
                                                .add_filter(
                                                    "HDR & Images",
                                                    &["hdr", "jpg", "jpeg", "png"],
                                                )
                                                .pick_file()
                                                .await;

                                            if let Some(file_handle) = file {
                                                let name = {
                                                    #[cfg(not(target_arch = "wasm32"))]
                                                    {
                                                        file_handle
                                                            .path()
                                                            .file_name()
                                                            .map(|n| {
                                                                n.to_string_lossy().to_string()
                                                            })
                                                            .unwrap_or_else(|| {
                                                                "Unknown".to_string()
                                                            })
                                                    }
                                                    #[cfg(target_arch = "wasm32")]
                                                    {
                                                        file_handle.file_name()
                                                    }
                                                };

                                                let is_hdr = name.ends_with(".hdr")
                                                    || name.ends_with(".hdr.jpg");

                                                // Native: fire-and-forget via path
                                                #[cfg(not(target_arch = "wasm32"))]
                                                {
                                                    let path_str = file_handle
                                                        .path()
                                                        .to_string_lossy()
                                                        .to_string();
                                                    let handle = if is_hdr {
                                                        assets_clone.load_hdr_texture(path_str)
                                                    } else {
                                                        assets_clone.load_texture(
                                                            path_str,
                                                            ColorSpace::Srgb,
                                                            true,
                                                        )
                                                    };
                                                    let _ = tx.send(ViewerEvent::EnvTextureReady {
                                                        name,
                                                        handle,
                                                    });
                                                }

                                                // WASM: load from bytes (async)
                                                #[cfg(target_arch = "wasm32")]
                                                {
                                                    let data = file_handle.read().await;
                                                    let result = if is_hdr {
                                                        assets_clone
                                                            .load_hdr_texture_from_bytes_async(
                                                                &name, data,
                                                            )
                                                            .await
                                                    } else {
                                                        assets_clone
                                                            .load_texture_from_bytes_async(
                                                                &name,
                                                                data,
                                                                ColorSpace::Srgb,
                                                                true,
                                                            )
                                                            .await
                                                    };
                                                    match result {
                                                        Ok(handle) => {
                                                            let _ = tx.send(
                                                                ViewerEvent::EnvTextureReady {
                                                                    name,
                                                                    handle,
                                                                },
                                                            );
                                                        }
                                                        Err(e) => {
                                                            log::error!(
                                                                "Failed to load HDR texture: {}",
                                                                e
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        });
                                    }
                                });

                                // --- Light 光源 ---
                                ui.horizontal(|ui| {
                                    if let Some(light_bundle) =
                                        scene.get_light_bundle(self.light_node)
                                    {
                                        ui.checkbox(&mut light_bundle.1.visible, "Light");
                                        if light_bundle.1.visible {
                                            ui.add(
                                                egui::Slider::new(
                                                    &mut light_bundle.0.intensity,
                                                    0.1..=5.0,
                                                )
                                                .step_by(0.1)
                                                .logarithmic(true),
                                            );
                                            ui.checkbox(
                                                &mut light_bundle.0.cast_shadows,
                                                "Cast Shadows",
                                            );
                                        }
                                    }
                                });
                            });

                        ui.separator();

                        // ===== 动画控制 =====
                        CollapsingHeader::new("🎬 Animation")
                            .default_open(true)
                            .show(ui, |ui| {
                                if self.animations.is_empty() {
                                    ui.label("No animations available");
                                } else {
                                    // 播放控制
                                    ui.horizontal(|ui| {
                                        if ui
                                            .button(if self.is_playing {
                                                "⏸ Pause"
                                            } else {
                                                "▶ Play"
                                            })
                                            .clicked()
                                        {
                                            self.is_playing = !self.is_playing;
                                            if let Some(gltf_node) = self.gltf_node
                                                && let Some(mixer) =
                                                    scene.animation_mixers.get_mut(gltf_node)
                                            {
                                                mixer.enabled = self.is_playing;
                                            }
                                        }

                                        ui.label("Speed:");
                                        ui.add(
                                            egui::Slider::new(&mut self.playback_speed, 0.0..=2.0)
                                                .step_by(0.1)
                                                .suffix("x"),
                                        );
                                    });

                                    ui.separator();

                                    if let Some(gltf_node) = self.gltf_node {
                                        if let Some(mixer) =
                                            scene.animation_mixers.get_mut(gltf_node)
                                        {
                                            // checkbox for each animation clip
                                            for anim in &self.animations {
                                                // if let Some(action) = mixer.get_control_by_name(anim) {
                                                ui.horizontal(|ui| {
                                                    if let Some(action) =
                                                        mixer.get_control_by_name(anim)
                                                    {
                                                        let is_active = action.is_active();
                                                        let mut current_active = is_active;

                                                        ui.checkbox(&mut current_active, "");

                                                        let name = if anim.len() > 20 {
                                                            format!(
                                                                "{}... ({:.2}s)",
                                                                &anim[..20],
                                                                action.time
                                                            )
                                                        } else {
                                                            format!(
                                                                "{} ({:.2}s)",
                                                                anim, action.time
                                                            )
                                                        };

                                                        if ui
                                                            .selectable_label(current_active, name)
                                                            .clicked()
                                                        {
                                                            mixer.stop_all();
                                                            current_active = !current_active;
                                                        }

                                                        if current_active != is_active {
                                                            if current_active {
                                                                mixer.play(anim);
                                                            } else {
                                                                mixer.stop(anim);
                                                            }
                                                        }
                                                    }
                                                });
                                            }
                                        }
                                    }
                                }
                            });

                        ui.separator();

                        // ===== 渲染设置 =====
                        CollapsingHeader::new("⚙ Rendering").show(ui, |ui| {
                            let mut settings_changed = false;

                            // --- Render Path 选择 ---
                            let is_hf = self.render_path.supports_post_processing();
                            ui.horizontal(|ui| {
                                ui.label("Render Path:");
                                egui::ComboBox::from_id_salt("render_path_selector")
                                    .width(140.0)
                                    .selected_text(if is_hf {
                                        "High Fidelity"
                                    } else {
                                        "Basic Forward"
                                    })
                                    .show_ui(ui, |ui| {
                                        if ui.selectable_label(is_hf, "High Fidelity").clicked()
                                            && !is_hf
                                        {
                                            self.render_path = RenderPath::HighFidelity;
                                            settings_changed = true;
                                        }
                                        if ui.selectable_label(!is_hf, "Basic Forward").clicked()
                                            && is_hf
                                        {
                                            self.render_path = RenderPath::BasicForward;
                                            settings_changed = true;
                                        }
                                    });
                            });

                            #[cfg(not(target_arch = "wasm32"))]
                            // --- VSync toggle ---
                            if ui.checkbox(&mut self.vsync, "VSync").changed() {
                                settings_changed = true;
                            }

                            // --- Anisotropy slider ---
                            ui.horizontal(|ui| {
                                ui.label("Anisotropy:");
                                let mut aniso_index = match self.anisotropy_clamp {
                                    1 => 0,
                                    2 => 1,
                                    4 => 2,
                                    8 => 3,
                                    _ => 4,
                                };
                                const ANISO_VALUES: [u16; 5] = [1, 2, 4, 8, 16];
                                const ANISO_LABELS: [&str; 5] =
                                    ["Off (1×)", "2×", "4×", "8×", "16×"];
                                egui::ComboBox::from_id_salt("anisotropy_selector")
                                    .width(80.0)
                                    .selected_text(ANISO_LABELS[aniso_index])
                                    .show_ui(ui, |ui| {
                                        for (i, label) in ANISO_LABELS.iter().enumerate() {
                                            if ui
                                                .selectable_value(&mut aniso_index, i, *label)
                                                .changed()
                                            {
                                                self.anisotropy_clamp = ANISO_VALUES[aniso_index];
                                                settings_changed = true;
                                            }
                                        }
                                    });
                            });

                            // --- Apply settings if anything changed ---
                            if settings_changed {
                                renderer.update_settings(RendererSettings {
                                    path: self.render_path,
                                    vsync: self.vsync,
                                    anisotropy_clamp: self.anisotropy_clamp,
                                });
                            }

                            ui.separator();

                            // --- Anti-Aliasing mode selector (render-path aware) ---
                            {
                                let is_basic_forward = !is_hf;

                                // Safety fallback: if we're on BasicForward but have a
                                // post-process mode selected, downgrade.
                                if is_basic_forward
                                    && self.aa_cache.current.requires_post_processing()
                                {
                                    self.aa_cache.current = AaModeType::MSAA;
                                }

                                let mut changed = false;

                                ui.horizontal(|ui| {
                                    ui.label("AA:");
                                    egui::ComboBox::from_id_salt("aa_selector")
                                        .width(90.0)
                                        .selected_text(self.aa_cache.current.label())
                                        .show_ui(ui, |ui| {
                                            for &mode in AaModeType::ALL {
                                                if is_basic_forward
                                                    && mode.requires_post_processing()
                                                {
                                                    continue;
                                                }
                                                if ui
                                                    .selectable_value(
                                                        &mut self.aa_cache.current,
                                                        mode,
                                                        mode.label(),
                                                    )
                                                    .changed()
                                                {
                                                    changed = true;
                                                }
                                            }
                                        });
                                });

                                // Detail panels for the selected mode.
                                match self.aa_cache.current {
                                    AaModeType::FXAA
                                    | AaModeType::MSAA_FXAA
                                    | AaModeType::TAA_FXAA => {
                                        ui.horizontal(|ui| {
                                            ui.label("FXAA Quality:");
                                            let cur = self.aa_cache.fxaa.quality();
                                            egui::ComboBox::from_id_salt("fxaa_quality")
                                                .width(100.0)
                                                .selected_text(cur.name())
                                                .show_ui(ui, |ui| {
                                                    for quality in FxaaQuality::all() {
                                                        if ui
                                                            .selectable_label(
                                                                cur == *quality,
                                                                quality.name(),
                                                            )
                                                            .clicked()
                                                        {
                                                            self.aa_cache
                                                                .fxaa
                                                                .set_quality(*quality);
                                                            changed = true;
                                                        }
                                                    }
                                                });
                                        });
                                    }
                                    _ => {}
                                }

                                match self.aa_cache.current {
                                    AaModeType::TAA | AaModeType::TAA_FXAA => {
                                        ui.horizontal(|ui| {
                                            ui.label("Feedback Weight:");
                                            if ui
                                                .add(
                                                    egui::Slider::new(
                                                        &mut self.aa_cache.taa.feedback_weight,
                                                        0.0..=1.0,
                                                    )
                                                    .step_by(0.01),
                                                )
                                                .changed()
                                            {
                                                changed = true;
                                            }
                                        });
                                    }
                                    _ => {}
                                }

                                // Synchronise the assembled mode to the active camera.
                                if changed {
                                    let mode = self.aa_cache.build_mode();
                                    if let Some(cam_handle) = scene.active_camera {
                                        if let Some(cam) = scene.cameras.get_mut(cam_handle) {
                                            cam.set_aa_mode(mode.clone());
                                        }
                                    }
                                }
                            }

                            ui.separator();

                            if is_hf {
                                // --- Tone Mapping 设置 (仅在 HighFidelity 模式下可用) ---
                                ui.add_enabled_ui(true, |ui| {
                                    ui.label("Tone Mapping:");

                                    // 模式选择
                                    ui.horizontal(|ui| {
                                        ui.label("Mode:");
                                        let current_mode = scene.tone_mapping.mode;
                                        egui::ComboBox::from_id_salt("tone_mapping_mode")
                                            .width(120.0)
                                            .selected_text(current_mode.name())
                                            .show_ui(ui, |ui| {
                                                for mode in myth::ToneMappingMode::all() {
                                                    if ui
                                                        .selectable_label(
                                                            current_mode == *mode,
                                                            mode.name(),
                                                        )
                                                        .clicked()
                                                    {
                                                        scene.tone_mapping.set_mode(*mode);
                                                    }
                                                }
                                            });
                                    });

                                    let mut uniforms_mut = scene.tone_mapping.uniforms_mut();

                                    // 曝光度
                                    ui.horizontal(|ui| {
                                        ui.label("Exposure:");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut uniforms_mut.exposure,
                                                0.1..=5.0,
                                            )
                                            .step_by(0.1)
                                            .logarithmic(true),
                                        )
                                    });

                                    ui.separator();
                                    // --- Color Grading (LUT) ---
                                    ui.label("Color Grading (LUT):");

                                    ui.horizontal(|ui| {
                                        ui.label("Contribution:");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut uniforms_mut.lut_contribution,
                                                0.0..=1.0,
                                            )
                                            .step_by(0.05),
                                        )
                                    });

                                    drop(uniforms_mut);

                                    if scene.tone_mapping.has_lut() {
                                        if ui.button("Remove LUT").clicked() {
                                            scene.tone_mapping.set_lut_texture(None);
                                        }
                                    } else {
                                        if ui.button("📂 Load LUT...").clicked() {
                                            let tx = self.event_tx.clone();
                                            let assets_clone = assets.clone();
                                            execute_future(async move {
                                                let file = rfd::AsyncFileDialog::new()
                                                    .add_filter("LUT Files", &["cube", "bin"])
                                                    .pick_file()
                                                    .await;

                                                if let Some(file_handle) = file {
                                                    let name = {
                                                        #[cfg(not(target_arch = "wasm32"))]
                                                        {
                                                            file_handle
                                                                .path()
                                                                .file_name()
                                                                .map(|n| {
                                                                    n.to_string_lossy().to_string()
                                                                })
                                                                .unwrap_or_else(|| {
                                                                    "Unknown.cube".to_string()
                                                                })
                                                        }
                                                        #[cfg(target_arch = "wasm32")]
                                                        {
                                                            file_handle.file_name()
                                                        }
                                                    };

                                                    // Native: fire-and-forget via path
                                                    #[cfg(not(target_arch = "wasm32"))]
                                                    {
                                                        let path_str = file_handle
                                                            .path()
                                                            .to_string_lossy()
                                                            .to_string();
                                                        let handle =
                                                            assets_clone.load_lut_texture(path_str);
                                                        let _ =
                                                            tx.send(ViewerEvent::LutTextureReady {
                                                                name,
                                                                handle,
                                                            });
                                                    }

                                                    // WASM: load from bytes (async)
                                                    #[cfg(target_arch = "wasm32")]
                                                    {
                                                        let data = file_handle.read().await;
                                                        match assets_clone
                                                            .load_lut_texture_from_bytes_async(
                                                                &name, data,
                                                            )
                                                            .await
                                                        {
                                                            Ok(handle) => {
                                                                let _ = tx.send(
                                                                    ViewerEvent::LutTextureReady {
                                                                        name,
                                                                        handle,
                                                                    },
                                                                );
                                                            }
                                                            Err(e) => {
                                                                log::error!(
                                                                    "Failed to load LUT: {}",
                                                                    e
                                                                );
                                                            }
                                                        }
                                                    }
                                                }
                                            });
                                        }
                                    }

                                    // --- Vignette ---
                                    let mut uniforms_mut = scene.tone_mapping.uniforms_mut(); // Get mutable guard for vignette settings
                                    ui.separator();
                                    ui.label("Vignette:");

                                    ui.checkbox(&mut self.vignette_breathing, "Breathing");

                                    ui.horizontal(|ui| {
                                        ui.label("Intensity:");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut uniforms_mut.vignette_intensity,
                                                0.0..=2.0,
                                            )
                                            .step_by(0.01),
                                        )
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Smoothness:");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut uniforms_mut.vignette_smoothness,
                                                0.1..=1.0,
                                            )
                                            .step_by(0.01),
                                        )
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Color:");
                                        let mut color_arr = uniforms_mut.vignette_color.to_array();
                                        if ui
                                            .color_edit_button_rgb(&mut color_arr)
                                            .changed()
                                        {
                                            uniforms_mut.vignette_color =
                                                Vec3::from_array(color_arr);
                                        }
                                    });

                                    // --- Chromatic Aberration、 Contrast & Saturation、 Film Grain ---
                                    ui.separator();
                                    ui.horizontal(|ui| {
                                        ui.label("Chromatic Aberration:");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut uniforms_mut.chromatic_aberration,
                                                0.0..=5.0,
                                            )
                                            .step_by(0.01),
                                        )
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Contrast:");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut uniforms_mut.contrast,
                                                0.5..=2.0,
                                            )
                                            .step_by(0.01),
                                        )
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Saturation:");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut uniforms_mut.saturation,
                                                0.0..=2.0,
                                            )
                                            .step_by(0.01),
                                        )
                                    });

                                    ui.horizontal(|ui| {
                                        ui.label("Film Grain:");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut uniforms_mut.film_grain,
                                                0.0..=1.0,
                                            )
                                            .step_by(0.01),
                                        )
                                    });
                                });

                                ui.separator();

                                // ===== Bloom 后处理 =====

                                // 开关 (always available when HighFidelity is on)
                                ui.add_enabled_ui(true, |ui| {
                                    let mut bloom_enabled = scene.bloom.enabled;
                                    if ui.checkbox(&mut bloom_enabled, "Enable Bloom").changed() {
                                        scene.bloom.set_enabled(bloom_enabled);
                                    }
                                });

                                let bloom_enabled = scene.bloom.enabled;

                                ui.add_enabled_ui(bloom_enabled, |ui| {
                                    // Strength
                                    ui.horizontal(|ui| {
                                        ui.label("Strength:");
                                        let mut strength = scene.bloom.strength();
                                        if ui
                                            .add(
                                                egui::Slider::new(&mut strength, 0.0..=1.0)
                                                    .step_by(0.005)
                                                    .fixed_decimals(3),
                                            )
                                            .changed()
                                        {
                                            scene.bloom.set_strength(strength);
                                        }
                                    });

                                    // Radius
                                    ui.horizontal(|ui| {
                                        ui.label("Radius:");
                                        let mut radius = scene.bloom.radius();
                                        if ui
                                            .add(
                                                egui::Slider::new(&mut radius, 0.001..=0.05)
                                                    .step_by(0.001)
                                                    .fixed_decimals(3),
                                            )
                                            .changed()
                                        {
                                            scene.bloom.set_radius(radius);
                                        }
                                    });

                                    // Mip Levels
                                    ui.horizontal(|ui| {
                                        ui.label("Mip Levels:");
                                        let mut mip_levels = scene.bloom.max_mip_levels();
                                        if ui
                                            .add(egui::Slider::new(&mut mip_levels, 1..=10))
                                            .changed()
                                        {
                                            scene.bloom.set_max_mip_levels(mip_levels);
                                        }
                                    });

                                    // Karis Average
                                    let mut karis = scene.bloom.karis_average;
                                    if ui
                                        .checkbox(&mut karis, "Karis Average (anti-firefly)")
                                        .changed()
                                    {
                                        scene.bloom.set_karis_average(karis);
                                    }
                                });

                                ui.separator();
                                // ===== SSAO 设置 =====
                                ui.add_enabled_ui(true, |ui| {
                                    let mut ssao_enabled = scene.ssao.enabled;
                                    if ui.checkbox(&mut ssao_enabled, "Enable SSAO").changed() {
                                        scene.ssao.set_enabled(ssao_enabled);
                                    }
                                });

                                let ssao_enabled = scene.ssao.enabled;

                                ui.add_enabled_ui(ssao_enabled, |ui| {
                                    // Radius
                                    ui.horizontal(|ui| {
                                        ui.label("Radius:");
                                        let mut radius = scene.ssao.radius();
                                        if ui
                                            .add(
                                                egui::Slider::new(&mut radius, 0.1..=5.0)
                                                    .step_by(0.1)
                                                    .fixed_decimals(2),
                                            )
                                            .changed()
                                        {
                                            scene.ssao.set_radius(radius);
                                        }
                                    });

                                    // Intensity
                                    ui.horizontal(|ui| {
                                        ui.label("Intensity:");
                                        let mut intensity = scene.ssao.intensity();
                                        if ui
                                            .add(
                                                egui::Slider::new(&mut intensity, 0.1..=5.0)
                                                    .step_by(0.1)
                                                    .fixed_decimals(2),
                                            )
                                            .changed()
                                        {
                                            scene.ssao.set_intensity(intensity);
                                        }
                                    });

                                    // Sample Count
                                    ui.horizontal(|ui| {
                                        ui.label("Sample Count:");
                                        let mut sample_count = scene.ssao.sample_count();
                                        if ui
                                            .add(
                                                egui::Slider::new(&mut sample_count, 1..=64)
                                                    .step_by(1.0),
                                            )
                                            .changed()
                                        {
                                            scene.ssao.set_sample_count(sample_count);
                                        }
                                    });
                                });

                                ui.separator();
                                ui.label("ScreenSpace Profle");
                                // ScreenSpace Subsurface Scattering (SSSS)
                                ui.horizontal(|ui| {
                                    ui.checkbox(
                                        &mut scene.screen_space.enable_sss,
                                        "Screen-Space Subsurface Scattering (SSSS)",
                                    )
                                });

                                // ===== Debug View (compile-time gated) =====
                                #[cfg(feature = "debug_view")]
                                {
                                    use myth::prelude::DebugViewMode;
                                    ui.separator();

                                    ui.horizontal(|ui| {
                                        ui.label("Debug View:");

                                        let mut targets = vec![
                                            DebugViewMode::None,
                                            DebugViewMode::Albedo,
                                            DebugViewMode::Roughness,
                                            DebugViewMode::Metalness,
                                        ];

                                        if is_hf {
                                            targets.push(DebugViewMode::Depth);
                                            targets.push(DebugViewMode::Normal);
                                            targets.push(DebugViewMode::Velocity);

                                            if scene.ssao.enabled {
                                                targets.push(DebugViewMode::SSAO);
                                            }
                                        }

                                        if !targets.contains(&self.debug_view.mode) {
                                            self.debug_view.mode = DebugViewMode::None;
                                        }

                                        egui::ComboBox::from_id_salt("debug_view_selector")
                                            .width(160.0)
                                            .selected_text(self.debug_view.mode.label())
                                            .show_ui(ui, |ui| {
                                                for &m in &targets {
                                                    ui.selectable_value(
                                                        &mut self.debug_view.mode,
                                                        m,
                                                        m.label(),
                                                    );
                                                }
                                            });
                                    });

                                    // Show scale slider for velocity mode.
                                    if self.debug_view.mode == DebugViewMode::Velocity {
                                        ui.horizontal(|ui| {
                                            ui.label("Velocity Scale:");
                                            ui.add(
                                                egui::Slider::new(
                                                    &mut self.debug_view.custom_scale,
                                                    1.0..=500.0,
                                                )
                                                .logarithmic(true),
                                            );
                                        });
                                    }

                                    // Apply debug view settings to camera.
                                    if let Some(cam_node) = scene.active_camera {
                                        if let Some(cam) = scene.get_camera_mut(cam_node) {
                                            cam.debug_view = self.debug_view;
                                        }
                                    }
                                }
                            }
                        });

                        ui.separator();

                        // ===== Inspector 开关 =====
                        if self.gltf_node.is_some()
                            && ui
                                .button(if self.show_inspector {
                                    "🔍 Hide Inspector"
                                } else {
                                    "🔍 Show Inspector"
                                })
                                .clicked()
                        {
                            self.show_inspector = !self.show_inspector;
                        }

                        ui.separator();

                        if ui.button("Dump Render Graph").clicked() {
                            if let Some(graph_dump) = renderer.dump_graph_mermaid() {
                                #[cfg(target_arch = "wasm32")]
                                {
                                    showRenderGraph(&graph_dump);
                                }

                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    println!("Render Graph:\n{}", graph_dump);
                                }
                            } else {
                                println!("Failed to dump render graph");
                            }
                        }

                        ui.separator();

                        // ===== 信息显示 =====
                        ui.label(format!("FPS: {:.1}", self.current_fps));
                    });
            });
    }

    /// 渲染 Inspector 面板
    fn render_inspector(&mut self, ctx: &egui::Context, scene: &mut Scene, assets: &AssetServer) {
        let Some(gltf_node) = self.gltf_node else {
            return;
        };

        egui::Window::new("🔍 Inspector")
            .resizable(true)
            .default_width(600.0)
            .default_height(500.0)
            .vscroll(false)
            .show(ctx, |ui| {
                ui.columns(2, |columns| {
                    columns[0].push_id("inspector_tree", |ui| {
                        let available_height = ui.available_height();

                        egui::ScrollArea::both()
                            .id_salt("inspector_tree")
                            .min_scrolled_height(available_height)
                            .show(ui, |ui| {
                                ui.set_min_width(ui.available_width());

                                egui::CollapsingHeader::new("📦 Nodes")
                                    .id_salt("nodes_tree")
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        self.render_node_tree(ui, scene, gltf_node, 0);
                                    });

                                egui::CollapsingHeader::new("🎨 Materials")
                                    .id_salt("materials_list")
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        for mat_info in &self.inspector_materials {
                                            let is_selected = self.inspector_target
                                                == Some(InspectorTarget::Material(mat_info.handle));
                                            if ui
                                                .selectable_label(is_selected, &mat_info.name)
                                                .clicked()
                                            {
                                                self.inspector_target = Some(
                                                    InspectorTarget::Material(mat_info.handle),
                                                );
                                            }
                                        }
                                    });

                                egui::CollapsingHeader::new("🖼 Textures")
                                    .id_salt("textures_list")
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        for tex_info in &self.inspector_textures {
                                            let is_selected = self.inspector_target
                                                == Some(InspectorTarget::Texture(tex_info.handle));
                                            if ui
                                                .selectable_label(is_selected, &tex_info.name)
                                                .clicked()
                                            {
                                                self.inspector_target =
                                                    Some(InspectorTarget::Texture(tex_info.handle));
                                            }
                                        }
                                    });
                            });
                    });

                    // === 右侧：详情面板 ===
                    columns[1].push_id("inspector_details", |ui| {
                        let available_height = ui.available_height();

                        egui::ScrollArea::vertical()
                            .id_salt("inspector_details")
                            .min_scrolled_height(available_height)
                            .show(ui, |ui| {
                                ui.set_min_width(ui.available_width());

                                if let Some(target) = &self.inspector_target {
                                    match target {
                                        InspectorTarget::Node(handle) => {
                                            self.render_node_details(ui, scene, *handle, assets);
                                        }
                                        InspectorTarget::Material(handle) => {
                                            self.render_material_details(ui, assets, *handle);
                                        }
                                        InspectorTarget::Texture(handle) => {
                                            self.render_texture_details(ui, assets, *handle);
                                        }
                                    }
                                } else {
                                    ui.label("Select an item from the tree to see details.");
                                }
                            });
                    });
                });
            });
    }

    /// 递归渲染节点树
    fn render_node_tree(
        &mut self,
        ui: &mut egui::Ui,
        scene: &Scene,
        node: NodeHandle,
        depth: usize,
    ) {
        let Some(node_data) = scene.get_node(node) else {
            return;
        };

        let name = scene
            .get_name(node)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Node_{:?}", node));

        // 确定节点图标
        let icon = if scene.get_mesh(node).is_some() {
            "🧊"
        } else if scene.get_camera(node).is_some() {
            "📷"
        } else if scene.get_light(node).is_some() {
            "💡"
        } else {
            "📁"
        };

        let label = format!("{} {}", icon, name);
        let is_selected = self.inspector_target == Some(InspectorTarget::Node(node));

        if node_data.children().is_empty() {
            // 叶子节点
            if ui.selectable_label(is_selected, &label).clicked() {
                self.inspector_target = Some(InspectorTarget::Node(node));
            }
        } else {
            // 有子节点，使用折叠
            let header = egui::CollapsingHeader::new(&label)
                .default_open(depth < 2)
                .show(ui, |ui| {
                    for child in &node_data.children().to_vec() {
                        self.render_node_tree(ui, scene, *child, depth + 1);
                    }
                });

            if header.header_response.clicked() {
                self.inspector_target = Some(InspectorTarget::Node(node));
            }
        }
    }

    /// 渲染节点详情
    fn render_node_details(
        &self,
        ui: &mut egui::Ui,
        scene: &mut Scene,
        node: NodeHandle,
        assets: &AssetServer,
    ) {
        let Some(node_data) = scene.get_node(node) else {
            ui.label("Node not found");
            return;
        };

        let name = scene.get_name(node).unwrap_or("Unnamed");
        ui.heading(format!("📦 {}", name));
        ui.separator();

        // Transform 信息
        ui.label("Transform:");
        egui::Grid::new("transform_grid")
            .num_columns(2)
            .spacing([20.0, 4.0])
            .show(ui, |ui| {
                ui.label("Position:");
                ui.label(format!(
                    "{:.3}, {:.3}, {:.3}",
                    node_data.transform.position.x,
                    node_data.transform.position.y,
                    node_data.transform.position.z
                ));
                ui.end_row();

                ui.label("Rotation:");
                let euler = node_data.transform.rotation.to_euler(glam::EulerRot::XYZ);
                ui.label(format!(
                    "{:.1}°, {:.1}°, {:.1}°",
                    euler.0.to_degrees(),
                    euler.1.to_degrees(),
                    euler.2.to_degrees()
                ));
                ui.end_row();

                ui.label("Scale:");
                ui.label(format!(
                    "{:.3}, {:.3}, {:.3}",
                    node_data.transform.scale.x,
                    node_data.transform.scale.y,
                    node_data.transform.scale.z
                ));
                ui.end_row();

                ui.label("Visible:");
                ui.label(if node_data.visible { "Yes" } else { "No" });
                ui.end_row();
            });

        // Mesh 信息
        if let Some(mesh) = scene.get_mesh(node) {
            ui.separator();
            ui.label("Mesh:");

            egui::Grid::new("mesh_grid")
                .num_columns(2)
                .spacing([20.0, 4.0])
                .show(ui, |ui| {
                    if let Some(geo) = assets.geometries.get(mesh.geometry) {
                        // 获取顶点数（从 position 属性）
                        if let Some(pos_attr) = geo.get_attribute("position") {
                            ui.label("Vertices:");
                            ui.label(format!("{}", pos_attr.count));
                            ui.end_row();
                        }

                        if let Some(index_attr) = geo.index_attribute() {
                            ui.label("Indices:");
                            ui.label(format!("{}", index_attr.count));
                            ui.end_row();
                        }
                    }

                    ui.label("Material:");
                    let mat_name = assets
                        .materials
                        .get(mesh.material)
                        .and_then(|m| m.name.clone())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "Unknown".to_string());
                    ui.label(mat_name);
                    ui.end_row();
                });
        }
    }

    /// 渲染材质详情
    fn render_material_details(
        &mut self,
        ui: &mut egui::Ui,
        assets: &AssetServer,
        handle: MaterialHandle,
    ) {
        let Some(material) = assets.materials.get(handle) else {
            ui.label("Material not found");
            return;
        };

        // let mut material = (*material).clone();

        let name = material
            .name
            .clone()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "Unnamed Material".to_string());
        ui.heading(format!("🎨 {}", name));
        ui.separator();

        // let settings = material.settings();

        egui::Grid::new("material_grid")
            .num_columns(2)
            .spacing([20.0, 4.0])
            .show(ui, |ui| {
                // 透明度模式
                ui.label("Alpha Mode:");
                ui.label(format!("{:?}", material.alpha_mode()));
                ui.end_row();

                ui.label("Side:");
                ui.label(format!("{:?}", material.side()));
                ui.end_row();

                // 只处理 Physical 材质
                if let MaterialType::Physical(m) = &material.data {
                    {
                        // uniforms
                        // let mut uniform_mut = m.uniforms_mut();
                        let mut uniform_mut = m.uniforms_mut();

                        ui.label("Type:");
                        ui.label("PhysicalMaterial");
                        ui.end_row();

                        ui.label("Color:");
                        let mut color_arr = uniform_mut.color.to_array();
                        if ui
                            .color_edit_button_rgba_unmultiplied(&mut color_arr)
                            .changed()
                        {
                            uniform_mut.color = glam::Vec4::from_array(color_arr);
                        }
                        ui.end_row();

                        ui.label("Metalness:");
                        // ui.add(egui::DragValue::new(&mut uniform_mut.metalness).speed(0.01));
                        ui.add(egui::DragValue::new(&mut uniform_mut.metalness).speed(0.01));
                        ui.end_row();

                        ui.label("Roughness:");
                        ui.add(egui::DragValue::new(&mut uniform_mut.roughness).speed(0.01));
                        ui.end_row();

                        ui.label("Specular Intensity:");
                        ui.add(
                            egui::DragValue::new(&mut uniform_mut.specular_intensity).speed(0.01),
                        );
                        ui.end_row();

                        ui.label("Specular Color:");
                        let mut spec_arr = uniform_mut.specular_color.to_array();
                        if ui.color_edit_button_rgb(&mut spec_arr).changed() {
                            uniform_mut.specular_color = glam::Vec3::from_array(spec_arr);
                        }
                        ui.end_row();

                        ui.label("Clearcoat:");
                        ui.add(egui::DragValue::new(&mut uniform_mut.clearcoat).speed(0.01));
                        ui.end_row();

                        ui.label("Clearcoat Roughness:");
                        ui.add(
                            egui::DragValue::new(&mut uniform_mut.clearcoat_roughness).speed(0.01),
                        );
                        ui.end_row();

                        ui.label("IOR:");
                        ui.add(egui::DragValue::new(&mut uniform_mut.ior).speed(0.01));
                        ui.end_row();
                    }

                    ui.separator();
                    ui.end_row();

                    // --- SSS Profile ---
                    ui.label("SSS Profile:");
                    ui.horizontal(|ui| {
                        let mut current_sss_id_opt = m.sss_id();

                        let selected_text = if let Some(id) = current_sss_id_opt {
                            self.sss_profiles
                                .iter()
                                .find(|(pid, _, _)| *pid == id)
                                .map(|(_, name, _)| name.clone())
                                .unwrap_or_else(|| format!("Profile {}", id.0.get()))
                        } else {
                            "None".to_string()
                        };

                        egui::ComboBox::from_id_salt("sss_profile_combo")
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_value(&mut current_sss_id_opt, None, "None")
                                    .changed()
                                {
                                    m.set_sss_id(None);
                                }
                                for (id, name, _) in &self.sss_profiles {
                                    if ui
                                        .selectable_value(&mut current_sss_id_opt, Some(*id), name)
                                        .changed()
                                    {
                                        m.set_sss_id(Some(*id));
                                    }
                                }
                            });

                        if ui.button("New Profile").clicked() {
                            let new_profile = myth::resources::screen_space::SssProfile::new(
                                Vec3::new(0.85, 0.25, 0.15),
                                0.15,
                            );
                            if let Some(new_id) = assets.sss_registry.write().add(&new_profile) {
                                self.sss_profiles.push((
                                    new_id,
                                    format!("Profile {}", new_id.0.get()),
                                    new_profile,
                                ));
                                m.set_sss_id(Some(new_id));
                            }
                        }
                    });
                    ui.separator();
                    ui.end_row();

                    let current_sss_id_opt = m.sss_id();

                    if let Some(current_id) = current_sss_id_opt {
                        ui.label("SSS Settings:");
                        ui.vertical(|ui| {
                            let mut profile_to_update = None;
                            let mut profile_to_remove = None;

                            if let Some((id, name, profile)) = self
                                .sss_profiles
                                .iter_mut()
                                .find(|(pid, _, _)| *pid == current_id)
                            {
                                ui.horizontal(|ui| {
                                    ui.label("Name:");
                                    ui.text_edit_singleline(name);
                                });

                                let mut color_arr = profile.scatter_color.to_array();
                                if ui.color_edit_button_rgb(&mut color_arr).changed() {
                                    profile.scatter_color = glam::Vec3::from_array(color_arr);
                                    profile_to_update = Some((*id, profile.clone()));
                                }

                                if ui
                                    .add(
                                        egui::DragValue::new(&mut profile.scatter_radius)
                                            .speed(0.01)
                                            .prefix("Radius: "),
                                    )
                                    .changed()
                                {
                                    profile_to_update = Some((*id, profile.clone()));
                                }

                                if ui.button("Delete Profile").clicked() {
                                    profile_to_remove = Some(*id);
                                }
                            }

                            if let Some((id, profile)) = profile_to_update {
                                assets.sss_registry.write().update(id, &profile);
                            }

                            if let Some(id) = profile_to_remove {
                                assets.sss_registry.write().remove(id);
                                self.sss_profiles.retain(|(pid, _, _)| *pid != id);
                                m.set_sss_id(None);
                            }
                        });
                        ui.end_row();
                    }

                    ui.separator();
                    ui.end_row();
                    {
                        // settings
                        let mut settings = m.settings_mut();
                        ui.label("Side");
                        egui::ComboBox::from_id_salt("side_combo")
                            .selected_text(format!("{:?}", settings.side))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut settings.side, Side::Front, "Front");
                                ui.selectable_value(&mut settings.side, Side::Back, "Back");
                                ui.selectable_value(&mut settings.side, Side::Double, "Double");
                            });
                        ui.end_row();

                        // 透明度模式
                        ui.label("Alpha Mode:");
                        egui::ComboBox::from_id_salt("alpha_mode_combo")
                            .selected_text(match settings.alpha_mode {
                                AlphaMode::Opaque => "Opaque",
                                AlphaMode::Mask => "Mask",
                                AlphaMode::Blend => "Blend",
                                AlphaMode::BlendMask => "Blend Mask",
                            })
                            .show_ui(ui, |ui| {
                                // 切换模式时，如果是 Mask 需要保留默认阈值
                                if ui
                                    .selectable_label(
                                        matches!(settings.alpha_mode, AlphaMode::Opaque),
                                        "Opaque",
                                    )
                                    .clicked()
                                {
                                    settings.alpha_mode = AlphaMode::Opaque;
                                }
                                if ui
                                    .selectable_label(
                                        matches!(settings.alpha_mode, AlphaMode::Mask),
                                        "Mask",
                                    )
                                    .clicked()
                                {
                                    // 如果之前不是 Mask，设为默认 0.5，否则保持
                                    if !matches!(settings.alpha_mode, AlphaMode::Mask) {
                                        settings.alpha_mode = AlphaMode::Mask;
                                    }
                                }
                                if ui
                                    .selectable_label(
                                        matches!(settings.alpha_mode, AlphaMode::Blend),
                                        "Blend",
                                    )
                                    .clicked()
                                {
                                    settings.alpha_mode = AlphaMode::Blend;
                                }
                                if ui
                                    .selectable_label(
                                        matches!(settings.alpha_mode, AlphaMode::BlendMask),
                                        "Blend Mask",
                                    )
                                    .clicked()
                                {
                                    if !matches!(settings.alpha_mode, AlphaMode::BlendMask) {
                                        settings.alpha_mode = AlphaMode::BlendMask;
                                    }
                                }
                            });

                        // 如果是 Mask 或 BlendMask 模式，额外显示阈值滑块
                        if let AlphaMode::Mask | AlphaMode::BlendMask = &mut settings.alpha_mode {
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::DragValue::new(&mut m.uniforms_mut().alpha_test)
                                        .speed(0.01)
                                        .range(0.0..=1.0)
                                        .prefix("Cutoff: "),
                                );
                                ui.checkbox(&mut settings.alpha_to_coverage, "A2C");
                            });
                        }
                        ui.end_row();

                        // --- Depth ---
                        ui.label("Depth:");
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut settings.depth_test, "Test");
                            ui.checkbox(&mut settings.depth_write, "Write");
                        });
                        ui.end_row();
                    }
                    // 纹理绑定
                    ui.separator();
                    ui.end_row();

                    ui.label("Textures:");
                    ui.end_row();
                    let builder = &mut ResourceBuilder::new();
                    m.define_bindings(builder);
                    for b in builder.bindings.iter() {
                        if let BindingResource::Texture(source) = &b.resource {
                            let name = b.name;
                            ui.label(format!("{}:", name));

                            if let Some(s) = source {
                                match s {
                                    TextureSource::Asset(tex_handle) => {
                                        if ui.label(name).clicked() {
                                            self.inspector_target =
                                                Some(InspectorTarget::Texture(*tex_handle));
                                        }

                                        if let Some(tex) = assets.textures.get(*tex_handle) {
                                            let tex_name = tex
                                                .name()
                                                .or_else(|| {
                                                    self.inspector_textures
                                                        .iter()
                                                        .find(|t| t.handle == *tex_handle)
                                                        .map(|t| t.name.as_str())
                                                })
                                                .unwrap_or("Unnamed");

                                            if ui.link(tex_name).clicked() {
                                                self.inspector_target =
                                                    Some(InspectorTarget::Texture(*tex_handle));
                                            }
                                        } else {
                                            ui.label("None");
                                        }
                                    }
                                    _ => {
                                        ui.label("Non-asset texture");
                                    }
                                }
                            }

                            // });
                            ui.end_row();
                        };
                    }
                }
            });
    }

    /// 渲染纹理详情
    fn render_texture_details(
        &mut self,
        ui: &mut egui::Ui,
        assets: &AssetServer,
        handle: TextureHandle,
    ) {
        let Some(texture) = assets.textures.get(handle) else {
            ui.label("Texture not found");
            return;
        };

        let name = texture
            .name()
            .or_else(|| {
                self.inspector_textures
                    .iter()
                    .find(|t| t.handle == handle)
                    .map(|t| t.name.as_str())
            })
            .unwrap_or("Unnamed");

        ui.heading(format!("🖼 {}", name));
        ui.separator();

        let image_info = assets.images.get(texture.image);

        egui::Grid::new("texture_grid")
            .num_columns(2)
            .spacing([20.0, 4.0])
            .show(ui, |ui| {
                ui.label("Dimensions:");
                if let Some(ref img) = image_info {
                    ui.label(format!("{}x{}", img.width, img.height));
                } else {
                    ui.label("N/A");
                }
                ui.end_row();

                ui.label("Format:");
                if let Some(ref img) = image_info {
                    ui.label(format!("{:?}", img.format));
                } else {
                    ui.label("N/A");
                }
                ui.end_row();

                ui.label("Mip Levels:");
                ui.label(if texture.generate_mipmaps {
                    "Auto-generated"
                } else {
                    "1"
                });
                ui.end_row();

                ui.label("Address Mode U:");
                ui.label(format!("{:?}", texture.sampler.address_mode_u));
                ui.end_row();

                ui.label("Address Mode V:");
                ui.label(format!("{:?}", texture.sampler.address_mode_v));
                ui.end_row();

                ui.label("Mag Filter:");
                ui.label(format!("{:?}", texture.sampler.mag_filter));
                ui.end_row();

                ui.label("Min Filter:");
                ui.label(format!("{:?}", texture.sampler.min_filter));
                ui.end_row();
            });

        ui.separator();
        // 预览纹理
        ui.label("Preview:");
        if let Some(tex_id) = self.ui_pass.request_texture(handle) {
            let size = if let Some(ref img) = image_info {
                egui::vec2(img.width as f32, img.height as f32)
            } else {
                egui::vec2(256.0, 256.0)
            };

            // 自适应缩放
            let available_width = ui.available_width();
            let display_size = if size.x > available_width {
                let scale = available_width / size.x;
                egui::vec2(available_width, size.y * scale)
            } else {
                size
            };

            ui.image(egui::load::SizedTexture::new(tex_id, display_size));
        } else {
            // 如果返回 None，说明还在注册中或等待 GPU 上传
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(" Loading GPU Texture...");
            });

            // 强制触发重绘，以便一旦纹理就绪能立刻显示出来，不用等鼠标动
            ui.ctx().request_repaint();
        }
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// Fetch the remote Khronos glTF model list.
async fn fetch_remote_model_list() -> Result<Vec<ModelInfo>, String> {
    let request = ehttp::Request::get(MODEL_LIST_URL);
    let response = ehttp::fetch_async(request)
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !response.ok {
        return Err(format!(
            "HTTP error: {} {}",
            response.status, response.status_text
        ));
    }

    let text = response
        .text()
        .ok_or_else(|| "Response body is not valid UTF-8".to_string())?;

    let models: Vec<ModelInfo> =
        serde_json::from_str(text).map_err(|e| format!("Failed to parse JSON: {e}"))?;

    Ok(models)
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_future<F: std::future::Future<Output = ()> + Send + 'static>(f: F) {
    tokio::spawn(f);
}

#[cfg(target_arch = "wasm32")]
fn execute_future<F: std::future::Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}

#[myth::main]
fn main() -> myth::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Tokio Runtime");

    #[cfg(not(target_arch = "wasm32"))]
    let _enter = rt.enter();

    #[cfg(target_arch = "wasm32")]
    log::info!("Starting glTF Viewer (WASM)...");

    App::new()
        .with_title("glTF Viewer")
        .with_settings(RendererSettings {
            vsync: cfg!(target_arch = "wasm32"),
            ..Default::default()
        })
        .run::<GltfViewer>()
}

// ============================================================================
// WASM Entry Point
// ============================================================================

// 绑定全局 window 对象上的 JS 方法
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window)]
    fn showRenderGraph(graph: &str);
}
