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
//! Run: cargo run --example gltf_viewer --release
//! 
//! # Architecture Notes
//! This example demonstrates the "UI as a Plugin" pattern:
//! - `UiPass` implements `RenderNode` trait, can be injected into RenderGraph
//! - Inject UI Pass into UI stage via `configure_render_pipeline()` method
//! - Engine core does not depend on egui at all

mod ui_pass;

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

use glam::Vec3;
use three::engine::FrameState;
use three::renderer::core::{BindingResource, ResourceBuilder};
use three::resources::texture::TextureSource;
use winit::event::WindowEvent;

use three::app::winit::{App, AppHandler};
use three::assets::{GltfLoader, MaterialHandle, SharedPrefab, TextureHandle};
use three::scene::{Camera, NodeHandle, light};
use three::renderer::graph::RenderStage;
use three::renderer::settings::{RenderSettings};
use three::ToneMappingMode;
use three::{AssetServer, OrbitControls, RenderableMaterialTrait, Scene, ThreeEngine};
use three::utils::fps_counter::FpsCounter;

use ui_pass::UiPass;
use winit::keyboard::PhysicalKey;
use winit::window::Window;

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

    // === File Dialog Related ===
    /// File dialog receiver
    #[cfg(not(target_arch = "wasm32"))]
    file_dialog_rx: Receiver<PathBuf>,
    #[cfg(target_arch = "wasm32")]
    file_dialog_rx: Receiver<(String, Vec<u8>)>,

    /// File dialog sender
    #[cfg(not(target_arch = "wasm32"))]
    file_dialog_tx: Sender<PathBuf>,
    #[cfg(target_arch = "wasm32")]
    file_dialog_tx: Sender<(String, Vec<u8>)>,
    
    // === Remote Model Related ===
    /// Remote model list
    model_list: Vec<ModelInfo>,
    /// Currently selected remote model index
    selected_model_index: usize,
    /// Loading state
    loading_state: LoadingState,
    /// Async load result receiver
    load_receiver: Option<Receiver<LoadResult>>,
    /// Async load request sender
    load_sender: Sender<LoadResult>,
    /// Preferred glTF variants (by priority)
    preferred_variants: Vec<&'static str>,
    
    // === Async Prefab Loading ===
    /// Prefab load result receiver
    prefab_receiver: Receiver<PrefabLoadResult>,
    /// Prefab load sender
    prefab_sender: Sender<PrefabLoadResult>,
    
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
    /// HDR rendering toggle (cached from renderer)
    hdr_enabled: bool,
    /// MSAA sample count (cached from renderer)
    msaa_samples: u32,
    
    hdr_receiver: Option<Receiver<TextureHandle>>,

    show_ui: bool,
}

/// Async Prefab load result
struct PrefabLoadResult {
    prefab: SharedPrefab,
    display_name: String,
}

/// Async load result
enum LoadResult {
    ModelList(Result<Vec<ModelInfo>, String>),
}

#[cfg(not(target_arch = "wasm32"))]
const ASSET_PATH: &str = "examples/assets/";
#[cfg(target_arch = "wasm32")]
const ASSET_PATH: &str = "assets/";

impl AppHandler for GltfViewer {
    fn init(engine: &mut ThreeEngine, window: &Arc<Window>) -> Self {
        // 1. Create UI Pass
        let wgpu_ctx = engine.renderer.wgpu_ctx().expect("Renderer not initialized");
        let ui_pass = UiPass::new(
            &wgpu_ctx.device,
            wgpu_ctx.surface_view_format,
            window,
        );

        let scene = engine.scene_manager.create_active();

        let (hdr_tx, hdr_rx) = channel();

        let asset_server = engine.assets.clone();
        execute_future(async move {
            // AssetSource 会自动处理路径/URL
            // let env_map_path = [
            //     format!("{}{}", ASSET_PATH, "Park2/posx.jpg"),
            //     format!("{}{}", ASSET_PATH, "Park2/negx.jpg"),
            //     format!("{}{}", ASSET_PATH, "Park2/posy.jpg"),
            //     format!("{}{}", ASSET_PATH, "Park2/negy.jpg"),
            //     format!("{}{}", ASSET_PATH, "Park2/posz.jpg"),
            //     format!("{}{}", ASSET_PATH, "Park2/negz.jpg"),
            // ];
                
            let map_path = "royal_esplanade_2k.hdr.jpg";
            let env_map_path = format!("{}{}", ASSET_PATH, map_path);

            // match asset_server.load_cube_texture_async(env_map_path, three::ColorSpace::Srgb, true).await {
            match asset_server.load_hdr_texture_async(env_map_path).await {
                Ok(handle) => {
                    log::info!("HDR loaded");
                    let _ = hdr_tx.send(handle); // 发送 Handle 回主线程
                }
                Err(e) => log::error!("HDR load failed: {}", e),
            }
        });        
        
        scene.environment.set_ambient_color(Vec3::splat(0.3));

        // 3. 添加灯光
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 3.0);

        let light_node = scene.add_light(light);
        if let Some(node) = scene.get_node_mut(light_node) {
            node.transform.position = Vec3::new(1.0, 1.0, 1.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        // 4. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);
        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 1.0, 5.0);
            node.transform.look_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        }
        scene.active_camera = Some(cam_node_id);

        // 5. 创建异步通道
        let (tx, rx) = channel();
        let (file_dialog_tx, file_dialog_rx) = channel();
        let (prefab_tx, prefab_rx) = channel();

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

            // === 文件对话框相关 ===
            file_dialog_rx,
            file_dialog_tx,

            // 远程模型
            model_list: Vec::new(),
            selected_model_index: 0,
            loading_state: LoadingState::Idle,
            load_receiver: Some(rx),
            load_sender: tx,
            preferred_variants: vec!["glTF-Binary", "glTF-Embedded", "glTF"],
            
            // Prefab 异步加载
            prefab_receiver: prefab_rx,
            prefab_sender: prefab_tx,
            
            // Inspector
            show_inspector: false,
            inspector_target: None,
            inspector_materials: Vec::new(),
            inspector_textures: Vec::new(),
            
            // 渲染设置
            ibl_enabled: true,
            hdr_enabled: true,  // Match RenderSettings in main()
            msaa_samples: 4,     // Match RenderSettings in main()

            hdr_receiver: Some(hdr_rx),

            show_ui: true,
        };

        // 6. 启动加载远程模型列表
        viewer.fetch_model_list();

        viewer
    }

    fn on_event(&mut self, _engine: &mut ThreeEngine, window: &Arc<Window>, event: &WindowEvent) -> bool {


        // Tab 键切换 UI 显示
        if let WindowEvent::KeyboardInput { event, .. } = event {
            let PhysicalKey::Code(code) = event.physical_key else {
                return false;
            };
            if code == winit::keyboard::KeyCode::Tab && event.state == winit::event::ElementState::Pressed {
                self.show_ui = !self.show_ui;
                return true;
            }

        }

        // UI 优先处理事件
        if self.ui_pass.handle_input(window, event) {
            return true;
        }
        
        // 处理窗口大小调整
        if let WindowEvent::Resized(size) = event {
            let scale_factor = window.scale_factor() as f32;
            self.ui_pass.resize(size.width, size.height, scale_factor);
        }
        
        false
    }

    fn update(&mut self, engine: &mut ThreeEngine, window: &Arc<Window>, frame: &FrameState) {

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // 0. 处理异步加载结果
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
        if let Some(gltf_node) = self.gltf_node {
            if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
                mixer.time_scale = self.playback_speed;
            }
        }

        // 3. 相机控制
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls.update(transform, &engine.input, camera.fov, frame.dt);
            // camera.near = self.controls.spherical.radius * 0.01;
            // camera.update_projection_matrix();
        }

        // 4. 构建 UI
        if self.show_ui {
            self.ui_pass.begin_frame(window);
            let egui_ctx = self.ui_pass.context().clone();
            self.handle_drag_and_drop(&egui_ctx, engine.assets.clone());
            self.render_ui(engine);
            self.ui_pass.end_frame(window);
        }

    }

    fn compose_frame<'a>(&'a mut self, composer: three::renderer::graph::FrameComposer<'a>) {
        if self.show_ui {
            composer
                .add_node(RenderStage::UI, &mut self.ui_pass)
                .render();
        }else{
            composer.render();
        }
    }
}

impl GltfViewer {
    // ========================================================================
    // 模型加载
    // ========================================================================

    /// 异步获取远程模型列表
    fn fetch_model_list(&mut self) {
        self.loading_state = LoadingState::LoadingList;
        let tx = self.load_sender.clone();
        
        execute_future(
            async move {
                let result = fetch_remote_model_list().await;
                let _ = tx.send(LoadResult::ModelList(result));
            }
        );

    }

    /// 处理异步加载结果
    fn process_load_results(&mut self, scene: &mut Scene, assets: &AssetServer) {
        // 处理模型列表加载结果
        if let Some(rx) = &self.load_receiver {
            while let Ok(result) = rx.try_recv() {
                match result {
                    LoadResult::ModelList(Ok(list)) => {
                        log::info!("Loaded {} models from remote", list.len());
                        self.model_list = list;
                        self.loading_state = LoadingState::Idle;
                    }
                    LoadResult::ModelList(Err(e)) => {
                        log::error!("Failed to load model list: {}", e);
                        self.loading_state = LoadingState::Error(e);
                    }
                }
            }
        }
        
        // 处理 HDR 环境贴图加载结果

        if let Some(rx) = &self.hdr_receiver {
            if let Ok(texture) = rx.try_recv() {
                log::info!("Applying HDR environment map");
                scene.environment.set_env_map(Some(texture));
                scene.environment.set_intensity(1.0);
            }
        }
        

        // 处理 Prefab 加载结果 - 实例化到场景中
        while let Ok(result) = self.prefab_receiver.try_recv() {
            // 实例化新模型
            self.instantiate_prefab(scene, assets , result);
        }

        // Native: 处理文件对话框结果
        #[cfg(not(target_arch = "wasm32"))]
        while let Ok(path) = self.file_dialog_rx.try_recv() {
            self.load_model(ModelSource::Local(path), assets.clone());
        }

        // WASM: 处理浏览器文件选择结果
        #[cfg(target_arch = "wasm32")]
        while let Ok((name, data)) = self.file_dialog_rx.try_recv() {
            self.load_model(ModelSource::Local(name, data), assets.clone());
        }
    }

    /// 将加载完成的 Prefab 实例化到场景
    fn instantiate_prefab(&mut self, scene: &mut Scene, assets: &AssetServer, result: PrefabLoadResult) {

        // 清理旧模型
        if let Some(gltf_node) = self.gltf_node {
            scene.remove_node(gltf_node);
        }
        self.gltf_node = None;
        self.animations.clear();
        self.inspector_materials.clear();
        self.inspector_textures.clear();
        self.inspector_target = None;

        // 实例化新模型
        let gltf_node = scene.instantiate(&result.prefab);
        
        self.gltf_node = Some(gltf_node);
        self.model_name = Some(result.display_name.clone());
        self.current_animation = 0;

        // 获取动画列表并自动播放
        if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
            self.animations = mixer.list_animations();
            if let Some(clip_name) = self.animations.first() {
                mixer.play(clip_name);
            }
        }

        // 更新子树变换
        scene.update_subtree(gltf_node);
        
        // 调整相机以适应模型
        if let Some(bbox) = scene.get_bbox_of_node(gltf_node, assets) {
            let center = bbox.center();
            let radius = bbox.size().length() * 0.5;
            if let Some((_transform, camera)) = scene.query_main_camera_bundle() {
                camera.near = radius / 100.0;
                camera.update_projection_matrix();
                self.controls.set_target(center);
                // let distance = radius / (camera.fov / 2.0).tan();
                // self.controls.set_position(center + Vec3::new(0.0, radius, distance * 1.25));
                self.controls.set_position(center + Vec3::new(0.0, radius, radius * 2.5));
            }
        }

        // 收集 Inspector 数据
        self.collect_inspector_targets(scene, assets, gltf_node);
        
        self.loading_state = LoadingState::Idle;
        log::info!("Instantiated model: {}", result.display_name);
    }

    /// 加载模型（本地或远程） - 真正的异步加载
    fn load_model(&mut self, source: ModelSource, assets: AssetServer) {

        let prefab_tx = self.prefab_sender.clone();

        // 处理不同的加载源
        match source {
            ModelSource::Remote(url) => {
                let display_name = url.rsplit('/').next()
                    .unwrap_or("Remote Model")
                    .to_string();
                
                self.loading_state = LoadingState::LoadingModel(display_name.clone());

                execute_future(
                    async move {
                        match GltfLoader::load_async(url, assets).await {
                            Ok(prefab) => {
                                let _ = prefab_tx.send(PrefabLoadResult { prefab, display_name });
                            }
                            Err(e) => {
                                log::error!("Failed to load model: {}", e);
                            }
                        }
                    }
                );
            }

            #[cfg(not(target_arch = "wasm32"))]
            ModelSource::Local(path) => {
                let display_name = path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "Unknown".to_string());
                
                self.loading_state = LoadingState::LoadingModel(display_name.clone());
                
                let load_path = path.to_string_lossy().to_string();
                
                execute_future(
                    async move {
                        match GltfLoader::load_async(load_path, assets).await {
                            Ok(prefab) => {
                                let _ = prefab_tx.send(PrefabLoadResult { prefab, display_name });
                            }
                            Err(e) => {
                                log::error!("Failed to load model: {}", e);
                            }
                        }
                    }
                );
            }
            
            #[cfg(target_arch = "wasm32")]
            ModelSource::Local(name, data) => {
                self.loading_state = LoadingState::LoadingModel(name.clone());
                
                execute_future(async move {
                    match GltfLoader::load_from_bytes(data, assets).await {
                        Ok(prefab) => {
                            let _ = prefab_tx.send(PrefabLoadResult { prefab, display_name: name });
                        }
                        Err(e) => {
                            log::error!("Failed to load model from bytes: {}", e);
                        }
                    }
                });
            }
        }
    }

    /// 从选中的远程模型构建 URL
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
    
    /// 处理全局拖拽事件 (Native & WASM)
    fn handle_drag_and_drop(&mut self, ctx: &egui::Context, assets: AssetServer) {
        // 1. 检查是否有文件正在悬停 (可选：显示视觉提示)
        if !ctx.input(|i| i.raw.hovered_files.is_empty()) {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("file_drop_overlay")));
            let screen_rect = ctx.content_rect();
            
            // 绘制半透明覆盖层提示用户松手
            painter.rect_filled(
                screen_rect,
                0.0,
                egui::Color32::from_black_alpha(100),
            );
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
    
    /// 将 egui 的 DroppedFile 转换为 ModelSource
    fn process_dropped_file(&mut self, file: &egui::DroppedFile, assets: AssetServer) {
        // Native 平台：使用文件路径
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = &file.path {
            log::info!("File dropped (Native): {:?}", path);
            self.load_model(ModelSource::Local(path.clone()), assets);
        }

        // WASM 平台：使用文件字节数据
        // egui 在 Web 上会自动读取数据到 file.bytes (需要 features = ["persistence"] 或默认开启)
        #[cfg(target_arch = "wasm32")]
        if let Some(bytes) = &file.bytes {
            log::info!("File dropped (WASM): {}, {} bytes", file.name, bytes.len());
            self.load_model(ModelSource::Local(file.name.clone(), bytes.to_vec()), assets);
        } else {
            // 如果在 Native 拖拽但没拿到 path，或者 WASM 没拿到 bytes
            log::warn!("Dropped file has no data. Native path: {:?}, Bytes present: {}", file.path, file.bytes.is_some());
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
                stack.extend(node.children.iter().cloned());
            }
            
            // 收集 Mesh 的材质
            if let Some(mesh) = scene.get_mesh(node_handle) {
                let mat_handle = mesh.material;
                
                if !visited_materials.contains(&mat_handle) {
                    visited_materials.insert(mat_handle);
                    
                    let mat_name = assets.materials.get(mat_handle)
                        .and_then(|m| m.name.clone())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("Material_{:?}", mat_handle));
                    
                    self.inspector_materials.push(MaterialInfo {
                        handle: mat_handle,
                        name: mat_name.clone(),
                    });
                    
                    // 收集材质使用的纹理
                    if let Some(material) = assets.materials.get(mat_handle) {
                        self.collect_textures_from_material(&material, &mat_name, assets, &mut visited_textures);
                    }
                }
            }
        }
    }

    /// 从材质中收集纹理信息
    fn collect_textures_from_material(
        &mut self, 
        material: &three::Material, 
        mat_name: &str,
        assets: &AssetServer,
        visited: &mut std::collections::HashSet<TextureHandle>
    ) {
        // 使用通用方式收集纹理：通过 visit_textures trait 方法
        let mut collected = Vec::new();
        material.as_renderable().visit_textures(&mut |tex_source| {
            if let three::resources::texture::TextureSource::Asset(handle) = tex_source {
                if !visited.contains(handle) {
                    visited.insert(*handle);
                    collected.push(*handle);
                }
            }
        });
        
        for (i, tex_handle) in collected.into_iter().enumerate() {
            let texture_name = assets.textures.get(tex_handle)
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

    fn render_ui(&mut self, engine: &mut ThreeEngine) {
        let egui_ctx = self.ui_pass.context().clone();

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };
        
        // 主控制面板
        self.render_control_panel(&egui_ctx, scene, &engine.assets, &mut engine.renderer);
        
        // Inspector 面板
        if self.show_inspector {
            self.render_inspector(&egui_ctx,scene, &engine.assets);
        }
    }

    /// 渲染主控制面板
    fn render_control_panel(&mut self, ctx: &egui::Context, scene: &mut Scene, assets: &AssetServer, renderer: &mut three::Renderer) {

        egui::Window::new("Control Panel")
            .default_pos([10.0, 10.0])
            .default_width(320.0)
            .show(ctx, |ui| {
                // ===== 远程模型加载 =====
                ui.collapsing("🌐 Remote Models", |ui| {
                    let is_loading = matches!(self.loading_state, LoadingState::LoadingList | LoadingState::LoadingModel(_));
                    
                    ui.add_enabled_ui(!is_loading, |ui| {

                        ui.horizontal(|ui| {
                            let model_names: Vec<_> = self.model_list.iter()
                                .map(|m| m.name.as_str())
                                .collect();
                            ui.label("Model:");

                            let combo = egui::ComboBox::from_id_salt("remote_model_selector")
                                .width(180.0)
                                .selected_text(
                                    model_names.get(self.selected_model_index)
                                        .copied()
                                        .unwrap_or("Select a model...")
                                );
                            
                            combo.show_ui(ui, |ui| {
                                ui.set_min_width(250.0);
                                for (i, name) in model_names.iter().enumerate() {
                                    ui.selectable_value(&mut self.selected_model_index, i, *name);
                                }
                            });

                            if ui.button("Load").clicked() {
                                if let Some(url) = self.build_remote_url(self.selected_model_index) {
                                    self.load_model(ModelSource::Remote(url), assets.clone());
                                }
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
                            ui.colored_label(egui::Color32::RED, format!("⚠ Error: {}", e));
                        }
                        LoadingState::Idle => {}
                    }
                    
                    ui.label(format!("{} models available", self.model_list.len()));
                });

                ui.separator();

                // ===== 本地文件加载 =====
                ui.collapsing("📁 Local File", |ui| {
                    if ui.button("Open glTF/glb File...").clicked() {
                        // 克隆发送端，移动到异步块中
                        let sender = self.file_dialog_tx.clone();

                        // 生成异步任务
                        execute_future(async move {
                            let file = rfd::AsyncFileDialog::new()
                                .add_filter("glTF", &["gltf", "glb"])
                                .pick_file()
                                .await; // 这里 await 不会卡死 UI

                            if let Some(file_handle) = file {
                                // 获取路径并发送回主线程
                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    let path = file_handle.path().to_path_buf();
                                    let _ = sender.send(path);
                                }

                                #[cfg(target_arch = "wasm32")]
                                {
                                    let data = file_handle.read().await;
                                    let file_name = file_handle.file_name();
                                    let _ = sender.send((file_name, data));
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

                // ===== 动画控制 =====
                ui.collapsing("🎬 Animation", |ui| {
                    if self.animations.is_empty() {
                        ui.label("No animations available");
                    } else {
                        // 动画选择
                        let anim_name = self.animations.get(self.current_animation)
                            .cloned()
                            .unwrap_or_else(|| "Select Animation".to_string());
                        
                        ui.horizontal(|ui| {
                            ui.label("Clip:");
                            egui::ComboBox::from_id_salt("animation_selector")
                                .width(150.0)
                                .selected_text(&anim_name)
                                .show_ui(ui, |ui| {
                                    for (i, clip) in self.animations.iter().enumerate() {
                                        if ui.selectable_value(&mut self.current_animation, i, clip).changed() {
                                            if let Some(gltf_node) = self.gltf_node {
                                                if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
                                                    mixer.stop_all();
                                                    mixer.play(clip);
                                                }
                                            }
                                        }
                                    }
                                });
                        });

                        // 播放控制
                        ui.horizontal(|ui| {
                            if ui.button(if self.is_playing { "⏸ Pause" } else { "▶ Play" }).clicked() {
                                self.is_playing = !self.is_playing;
                                if let Some(gltf_node) = self.gltf_node {
                                    if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
                                        if self.is_playing {
                                            if let Some(anim) = self.animations.get(self.current_animation) {
                                                mixer.play(anim);
                                            }
                                        } else {
                                            mixer.stop_all();
                                        }
                                    }
                                }
                            }
                        });

                        // 播放速度
                        ui.horizontal(|ui| {
                            ui.label("Speed:");
                            ui.add(egui::Slider::new(&mut self.playback_speed, 0.0..=2.0)
                                .step_by(0.1)
                                .suffix("x"));
                        });
                    }
                });

                ui.separator();

                // ===== 渲染设置 =====
                ui.collapsing("⚙ Rendering", |ui| {
                    // --- HDR 渲染 ---
                    if ui.checkbox(&mut self.hdr_enabled, "HDR Rendering").changed() {
                        renderer.set_hdr_enabled(self.hdr_enabled);
                    }
                    
                    // --- MSAA 抗锯齿 ---
                    ui.horizontal(|ui| {
                        ui.label("MSAA:");
                        let msaa_options = [1u32, 4];
                        egui::ComboBox::from_id_salt("msaa_selector")
                            .width(60.0)
                            .selected_text(if self.msaa_samples == 1 { 
                                "Off".to_string() 
                            } else { 
                                format!("{}x", self.msaa_samples) 
                            })
                            .show_ui(ui, |ui| {
                                for &samples in &msaa_options {
                                    let label = if samples == 1 { "Off".to_string() } else { format!("{}x", samples) };
                                    if ui.selectable_value(&mut self.msaa_samples, samples, label).changed() {
                                        renderer.set_msaa_samples(self.msaa_samples);
                                    }
                                }
                            });
                    });
                    
                    ui.separator();
                    
                    // --- IBL 环境贴图 ---
                    if ui.checkbox(&mut self.ibl_enabled, "IBL (Environment Map)").changed() {
                        scene.environment.set_intensity(if self.ibl_enabled { 1.0 } else { 0.0 });
                    }
                    
                    ui.separator();
                    
                    // --- Tone Mapping 设置 (仅在 HDR 模式下可用) ---
                    ui.add_enabled_ui(self.hdr_enabled, |ui| {
                        ui.label("Tone Mapping:");
                        
                        // 曝光度
                        ui.horizontal(|ui| {
                            ui.label("Exposure:");
                            let mut exposure = scene.tone_mapping.exposure;
                            if ui.add(egui::Slider::new(&mut exposure, 0.1..=5.0)
                                .step_by(0.1)
                                .logarithmic(true))
                                .changed() 
                            {
                                scene.tone_mapping.set_exposure(exposure);
                            }
                        });
                        
                        // 模式选择
                        ui.horizontal(|ui| {
                            ui.label("Mode:");
                            let current_mode = scene.tone_mapping.mode;
                            egui::ComboBox::from_id_salt("tone_mapping_mode")
                                .width(120.0)
                                .selected_text(current_mode.name())
                                .show_ui(ui, |ui| {
                                    for mode in ToneMappingMode::all() {
                                        if ui.selectable_label(current_mode == *mode, mode.name()).clicked() {
                                            scene.tone_mapping.set_mode(*mode);
                                        }
                                    }
                                });
                        });
                    });
                    
                    if !self.hdr_enabled {
                        ui.label("ℹ Enable HDR to configure tone mapping");
                    }
                });

                ui.separator();

                // ===== Inspector 开关 =====
                if self.gltf_node.is_some() {
                    if ui.button(if self.show_inspector { "🔍 Hide Inspector" } else { "🔍 Show Inspector" }).clicked() {
                        self.show_inspector = !self.show_inspector;
                    }
                }

                ui.separator();

                // ===== 信息显示 =====
                ui.label(format!("FPS: {:.1}", self.current_fps));
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

                // 使用 columns(2) 将窗口分为左右两栏，它们会自动填充窗口宽度
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
                                        let is_selected = self.inspector_target == Some(InspectorTarget::Material(mat_info.handle));
                                        if ui.selectable_label(is_selected, &mat_info.name).clicked() {
                                            self.inspector_target = Some(InspectorTarget::Material(mat_info.handle));
                                        }
                                    }
                                });

                                egui::CollapsingHeader::new("🖼 Textures")
                                .id_salt("textures_list")
                                .default_open(true)
                                .show(ui, |ui| {
                                    for tex_info in &self.inspector_textures {
                                        let is_selected = self.inspector_target == Some(InspectorTarget::Texture(tex_info.handle));
                                        if ui.selectable_label(is_selected, &tex_info.name).clicked() {
                                            self.inspector_target = Some(InspectorTarget::Texture(tex_info.handle));
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
    fn render_node_tree(&mut self, ui: &mut egui::Ui, scene: &three::Scene, node: NodeHandle, depth: usize) {
        let Some(node_data) = scene.get_node(node) else {
            return;
        };
        
        let name = scene.get_name(node)
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
        
        if node_data.children.is_empty() {
            // 叶子节点
            if ui.selectable_label(is_selected, &label).clicked() {
                self.inspector_target = Some(InspectorTarget::Node(node));
            }
        } else {
            // 有子节点，使用折叠
            let header = egui::CollapsingHeader::new(&label)
                .default_open(depth < 2)
                .show(ui, |ui| {
                    for child in &node_data.children.clone() {
                        self.render_node_tree(ui, scene, *child, depth + 1);
                    }
                });
            
            if header.header_response.clicked() {
                self.inspector_target = Some(InspectorTarget::Node(node));
            }
        }
    }

    /// 渲染节点详情
    fn render_node_details(&self, ui: &mut egui::Ui, scene: &mut three::Scene, node: NodeHandle, assets: &AssetServer) {
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
                ui.label(format!("{:.3}, {:.3}, {:.3}", 
                    node_data.transform.position.x,
                    node_data.transform.position.y,
                    node_data.transform.position.z));
                ui.end_row();

                ui.label("Rotation:");
                let euler = node_data.transform.rotation.to_euler(glam::EulerRot::XYZ);
                ui.label(format!("{:.1}°, {:.1}°, {:.1}°", 
                    euler.0.to_degrees(),
                    euler.1.to_degrees(),
                    euler.2.to_degrees()));
                ui.end_row();

                ui.label("Scale:");
                ui.label(format!("{:.3}, {:.3}, {:.3}", 
                    node_data.transform.scale.x,
                    node_data.transform.scale.y,
                    node_data.transform.scale.z));
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
                    let mat_name = assets.materials.get(mesh.material)
                        .and_then(|m| m.name.clone())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "Unknown".to_string());
                    ui.label(mat_name);
                    ui.end_row();
                });
        }
    }

    /// 渲染材质详情
    fn render_material_details(&mut self, ui: &mut egui::Ui, assets: &AssetServer, handle: MaterialHandle) {
        let Some(material) = assets.materials.get(handle) else {
            ui.label("Material not found");
            return;
        };

        // let mut material = (*material).clone();

        let name = material.name.clone()
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
                match &material.data {
                    three::MaterialType::Physical(m) => {
                        {   // uniforms
                            // let mut uniform_mut = m.uniforms_mut();
                            let mut uniform_mut = m.uniforms_mut();

                            ui.label("Type:");
                            ui.label("MeshPhysicalMaterial");
                            ui.end_row();

                            
                            ui.label("Color:");
                            let mut color_arr = uniform_mut.color.to_array();
                            if ui.color_edit_button_rgba_unmultiplied(&mut color_arr).changed() {
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
                            ui.add(egui::DragValue::new(&mut uniform_mut.specular_intensity).speed(0.01));
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
                            ui.add(egui::DragValue::new(&mut uniform_mut.clearcoat_roughness).speed(0.01));
                            ui.end_row();

                            ui.label("IOR:");
                            ui.add(egui::DragValue::new(&mut uniform_mut.ior).speed(0.01));
                            ui.end_row();
                        }

                            ui.separator();
                            ui.end_row();

                        {   // settings
                            let mut settings = m.settings_mut();
                            ui.label("Side");
                            egui::ComboBox::from_id_salt("side_combo")
                                .selected_text(format!("{:?}", settings.side))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut settings.side, three::Side::Front, "Front");
                                    ui.selectable_value(&mut settings.side, three::Side::Back, "Back");
                                    ui.selectable_value(&mut settings.side, three::Side::Double, "Double");
                                });
                            ui.end_row();
                            
                            // 透明度模式
                            ui.label("Alpha Mode:");
                            egui::ComboBox::from_id_salt("alpha_mode_combo")
                                .selected_text(match settings.alpha_mode {
                                    three::AlphaMode::Opaque => "Opaque",
                                    three::AlphaMode::Mask(..) => "Mask",
                                    three::AlphaMode::Blend => "Blend",
                                })
                                .show_ui(ui, |ui| {
                                    // 切换模式时，如果是 Mask 需要保留默认阈值
                                    if ui.selectable_label(matches!(settings.alpha_mode, three::AlphaMode::Opaque), "Opaque").clicked() {
                                        settings.alpha_mode = three::AlphaMode::Opaque;
                                    }
                                    if ui.selectable_label(matches!(settings.alpha_mode, three::AlphaMode::Mask(..)), "Mask").clicked() {
                                        // 如果之前不是 Mask，设为默认 0.5，否则保持
                                        if !matches!(settings.alpha_mode, three::AlphaMode::Mask(..)) {
                                            settings.alpha_mode = three::AlphaMode::Mask(0.5, false);
                                        }
                                    }
                                    if ui.selectable_label(matches!(settings.alpha_mode, three::AlphaMode::Blend), "Blend").clicked() {
                                        settings.alpha_mode = three::AlphaMode::Blend;
                                    }
                                });
                            
                            // 如果是 Mask 模式，额外显示阈值滑块
                            if let three::AlphaMode::Mask(cutoff, a2c) = &mut settings.alpha_mode {
                                ui.horizontal(|ui| {
                                    // ui[1].add(egui::DragValue::new(cutoff).speed(0.01).range(0.0..=1.0).prefix(""));
                                    ui.add(egui::DragValue::new(cutoff).speed(0.01).range(0.0..=1.0).prefix("Cutoff: "));
                                    ui.checkbox(a2c, "A2C");
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
                        for (binding, name) in builder.resources.iter().zip(builder.names.iter()) {
                            match binding {
                                BindingResource::Texture(source) => {
                                    // ui.horizontal(|ui| {
                                        ui.label(format!("{}:", name));

                                        if let Some(s) = source{
                                            match s {
                                                TextureSource::Asset(tex_handle) => {
                                                    if ui.label(name).clicked() {
                                                        self.inspector_target = Some(InspectorTarget::Texture(*tex_handle));
                                                    }

                                                    if let Some(tex) = assets.textures.get(*tex_handle) {
                                                        let tex_name = tex.name()
                                                            .or_else(|| {
                                                                    self.inspector_textures.iter()
                                                                        .find(|t| t.handle == *tex_handle)
                                                                        .map(|t| t.name.as_str())
                                                                })
                                                                .unwrap_or("Unnamed");


                                                        if ui.link(tex_name).clicked() {
                                                            self.inspector_target = Some(InspectorTarget::Texture(*tex_handle));
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
                                }
                                _ => {}
                            };
                        }
                    }
                    _ => {}
                }
            });
    }

    /// 渲染纹理详情
    fn render_texture_details(&self, ui: &mut egui::Ui, assets: &AssetServer, handle: TextureHandle) {
        let Some(texture) = assets.textures.get(handle) else {
            ui.label("Texture not found");
            return;
        };

        let name = texture.name() 
            .or_else(|| {
                self.inspector_textures.iter()
                    .find(|t| t.handle == handle)
                    .map(|t| t.name.as_str())
            })
            .unwrap_or("Unnamed");

        ui.heading(format!("🖼 {}", name));
        ui.separator();

        egui::Grid::new("texture_grid")
            .num_columns(2)
            .spacing([20.0, 4.0])
            .show(ui, |ui| {
                ui.label("Dimensions:");
                ui.label(format!("{}x{}", texture.image.width(), texture.image.height()));
                ui.end_row();

                ui.label("Format:");
                ui.label(format!("{:?}", texture.image.format()));
                ui.end_row();

                ui.label("Mip Levels:");
                ui.label(if texture.generate_mipmaps { "Auto-generated" } else { "1" });
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
            let size = egui::vec2(texture.image.width() as f32, texture.image.height() as f32);
            
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

/// 同步获取远程模型列表 (Native)
async fn fetch_remote_model_list() -> Result<Vec<ModelInfo>, String> {
    let client = reqwest::Client::builder();

    // 仅在 Native 平台设置超时
    #[cfg(not(target_arch = "wasm32"))]
    let client = client.timeout(std::time::Duration::from_secs(30));

    let client = client.build().map_err(|e| format!("Failed to create HTTP client: {}", e))?;
    
    let response = client.get(MODEL_LIST_URL)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()));
    }
    
    let text = response.text().await
        .map_err(|e| format!("Failed to read response: {}", e))?;
    
    let models: Vec<ModelInfo> = serde_json::from_str(&text)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;
    
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

// ============================================================================
// Native Main Entry Point
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    env_logger::init();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("无法创建 Tokio Runtime");

    let _enter = rt.enter();
    
    App::new()
        .with_title("glTF Viewer")
        .with_settings(RenderSettings { 
            vsync: false, 
            clear_color: wgpu::Color { r: 0.03, g: 0.03, b: 0.03, a: 1.0 },
            enable_hdr: true,    // 启用 HDR 渲染
            msaa_samples: 4,    // 4x MSAA 抗锯齿
            ..Default::default() 
        })
        .run::<GltfViewer>()
}

// ============================================================================
// WASM Entry Point
// ============================================================================

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    // Set up panic hook for better error messages
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    
    // Initialize logging
    console_log::init_with_level(log::Level::Info)
        .expect("Failed to initialize logger");
    
    log::info!("Starting glTF Viewer (WASM)...");
    
    // Run the application
    if let Err(e) = App::new()
        .with_title("glTF Viewer")
        .with_settings(RenderSettings { vsync: true, ..Default::default() })
        .run::<GltfViewer>()
    {
        log::error!("Application error: {}", e);
    }
}

// WASM 需要一个空的 main 函数
#[cfg(target_arch = "wasm32")]
fn main() {}
