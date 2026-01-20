//! glTF Viewer 示例 (基于 App 模块)
//!
//! 一个交互式的 glTF/glb 文件查看器，演示如何将 egui 作为外部插件集成。
//! 
//! 功能：
//! - 通过文件对话框加载本地 glTF/glb 文件
//! - 动画播放控制（播放/暂停、速度调节）
//! - 相机轨道控制
//! - FPS 显示
//!
//! 运行：cargo run --example gltf_viewer --release
//! 
//! # 架构说明
//! 这个示例展示了 "UI as a Plugin" 模式：
//! - `UiPass` 实现了 `RenderNode` trait，可以注入到 RenderGraph
//! - 通过 `extra_render_nodes()` 方法将 UI Pass 注入渲染流程
//! - 引擎核心完全不依赖 egui

mod ui_pass;

use std::sync::Arc;
use std::path::PathBuf;
use glam::Vec3;
use winit::event::WindowEvent;

use three::app::{App, AppContext, AppHandler};
use three::assets::GltfLoader;
use three::scene::{Camera, NodeIndex, light};
use three::renderer::graph::RenderNode;
use three::renderer::settings::RenderSettings;
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::{AnimationMixer, AnimationAction, Binder};
use three::animation::clip::AnimationClip;

use ui_pass::UiPass;

/// glTF Viewer 应用状态
struct GltfViewer {
    /// UI Pass (egui 渲染)
    ui_pass: UiPass,
    
    /// 当前加载的模型根节点
    loaded_nodes: Vec<NodeIndex>,
    /// 动画混合器
    mixer: AnimationMixer,
    /// 可用的动画列表
    animations: Vec<Arc<AnimationClip>>,
    /// 当前选中的动画索引
    current_animation: usize,
    /// 是否正在播放动画
    is_playing: bool,
    /// 动画播放速度
    playback_speed: f32,
    /// 轨道控制器
    controls: OrbitControls,
    /// FPS 计数器
    fps_counter: FpsCounter,
    /// 当前 FPS
    current_fps: f32,
    /// 模型文件路径
    model_path: Option<PathBuf>,
    /// 是否需要重新加载模型
    pending_load: Option<PathBuf>,
}

impl AppHandler for GltfViewer {
    fn init(ctx: &mut AppContext) -> Self {
        // 1. 创建 UI Pass
        let wgpu_ctx = ctx.renderer.wgpu_ctx().expect("Renderer not initialized");
        let ui_pass = UiPass::new(
            &wgpu_ctx.device,
            wgpu_ctx.config.format,
            ctx.window,
        );

        // 2. 加载环境贴图
        let env_texture_handle = ctx.assets.load_cube_texture_from_files(
            [
                "examples/assets/Park2/posx.jpg",
                "examples/assets/Park2/negx.jpg",
                "examples/assets/Park2/posy.jpg",
                "examples/assets/Park2/negy.jpg",
                "examples/assets/Park2/posz.jpg",
                "examples/assets/Park2/negz.jpg",
            ],
            three::ColorSpace::Srgb
        ).expect("Failed to load environment map");

        let env_texture = ctx.assets.get_texture_mut(env_texture_handle).unwrap();
        env_texture.generate_mipmaps = true;
        ctx.scene.environment.set_env_map(Some((env_texture_handle, &env_texture)));

        // 3. 添加灯光
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        ctx.scene.add_light(light);

        // 4. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1, 1000.0);
        let cam_node_id = ctx.scene.add_camera(camera);
        if let Some(node) = ctx.scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 1.0, 5.0);
            node.transform.look_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        }
        ctx.scene.active_camera = Some(cam_node_id);

        Self {
            ui_pass,
            loaded_nodes: Vec::new(),
            mixer: AnimationMixer::new(),
            animations: Vec::new(),
            current_animation: 0,
            is_playing: true,
            playback_speed: 1.0,
            controls: OrbitControls::new(Vec3::new(0.0, 1.0, 5.0), Vec3::new(0.0, 1.0, 0.0)),
            fps_counter: FpsCounter::new(),
            current_fps: 0.0,
            model_path: None,
            pending_load: None,
        }
    }

    fn on_event(&mut self, ctx: &mut AppContext, event: &WindowEvent) -> bool {
        // UI 优先处理事件
        if self.ui_pass.handle_input(ctx.window, event) {
            return true;
        }
        
        // 处理窗口大小调整
        if let WindowEvent::Resized(size) = event {
            let scale_factor = ctx.window.scale_factor() as f32;
            self.ui_pass.resize(size.width, size.height, scale_factor);
        }
        
        false
    }

    fn update(&mut self, ctx: &mut AppContext) {
        // 1. 更新 FPS
        if let Some(fps) = self.fps_counter.update() {
            self.current_fps = fps;

            let title = if let Some(path) = &self.model_path {
                format!("glTF Viewer - {} | FPS: {:.0}", 
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    self.current_fps)
            } else {
                format!("glTF Viewer | FPS: {:.0}", self.current_fps)
            };
            ctx.window.set_title(&title);
        }

        // 2. 更新动画
        if self.is_playing {
            self.mixer.update(ctx.dt * self.playback_speed, ctx.scene);
        }

        // 3. 相机控制
        if let Some((transform, camera)) = ctx.scene.query_main_camera_bundle() {
            self.controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
        }

        // 4. 构建 UI
        self.ui_pass.begin_frame(ctx.window);
        self.render_ui(ctx);
        self.ui_pass.end_frame(ctx.window);

        // 5. 处理待加载的模型
        if let Some(path) = self.pending_load.take() {
            self.load_model(&path, ctx);
        }


    }

    fn extra_render_nodes(&self) -> Vec<&dyn RenderNode> {
        vec![&self.ui_pass]
    }
}

impl GltfViewer {
    fn load_model(&mut self, path: &PathBuf, ctx: &mut AppContext) {
        // 清理旧模型
        for node_id in &self.loaded_nodes {
            ctx.scene.remove_node(*node_id);
        }
        self.loaded_nodes.clear();
        self.animations.clear();
        self.mixer = AnimationMixer::new();

        // 加载新模型
        match GltfLoader::load(path, ctx.assets, ctx.scene) {
            Ok((nodes, animations)) => {
                self.loaded_nodes = nodes.clone();
                self.animations = animations.iter().map(|c| Arc::new(c.clone())).collect();
                self.model_path = Some(path.clone());
                self.current_animation = 0;

                // 自动播放第一个动画
                if !self.animations.is_empty() {
                    let clip = self.animations[0].clone();
                    let root_node = nodes.first().copied().unwrap();
                    let bindings = Binder::bind(ctx.scene, root_node, &clip);
                    let mut action = AnimationAction::new(clip);
                    action.bindings = bindings;
                    self.mixer.add_action(action);
                }

                if let Some(root_node) = nodes.first() {
                    ctx.scene.update_subtree(*root_node);
                    if let Some(bbox) = ctx.scene.get_bbox_of_node(*root_node, ctx.assets) {
                        let center = bbox.center();
                        let radius = bbox.size().length() * 0.5;
                        if let Some((_transform, camera)) = ctx.scene.query_main_camera_bundle() {
                            // self.controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
                            camera.near = radius * 0.01;
                            camera.far = radius * 10.0;
                            camera.update_projection_matrix();
                            self.controls.set_target(center);
                            self.controls.set_position(center + Vec3::new(0.0, radius, radius * 2.5));
                        }
                    }
                }

                log::info!("Loaded model: {:?}", path);
            }
            Err(e) => {
                log::error!("Failed to load model: {}", e);
            }
        }
    }

    fn render_ui(&mut self, ctx: &mut AppContext) {
        let egui_ctx = self.ui_pass.context().clone();

        // 主控制面板
        egui::Window::new("Control Panel")
            .default_pos([10.0, 10.0])
            .default_width(280.0)
            .show(&egui_ctx, |ui| {
                // 文件加载部分
                ui.heading("📁 File");
                ui.horizontal(|ui| {
                    if ui.button("Open glTF/glb File...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("glTF", &["gltf", "glb"])
                            .pick_file()
                        {
                            self.pending_load = Some(path);
                        }
                    }
                });

                if let Some(path) = &self.model_path {
                    ui.label(format!("Current File: {}", 
                        path.file_name().unwrap_or_default().to_string_lossy()));
                } else {
                    ui.label("No model loaded");
                }

                ui.separator();

                // Animation Control Section
                ui.heading("🎬 Animation");
                
                if self.animations.is_empty() {
                    ui.label("No animations available");
                } else {
                    // Animation selection
                    let current_anim = self.current_animation;
                    let anim_name = if current_anim < self.animations.len() {
                        self.animations[current_anim].name.clone()
                    } else {
                        "Select Animation".to_string()
                    };
                    
                    ui.horizontal(|ui| {
                        ui.label("Animation:");
                        egui::ComboBox::from_id_salt("animation_selector")
                            .selected_text(&anim_name)
                            .show_ui(ui, |ui| {
                                for (i, clip) in self.animations.iter().enumerate() {
                                    if ui.selectable_value(&mut self.current_animation, i, &clip.name).changed() {
                                        // 切换动画
                                        self.mixer = AnimationMixer::new();
                                        let root_node = self.loaded_nodes.first().copied().unwrap();
                                        let bindings = Binder::bind(ctx.scene, root_node, clip);
                                        let mut action = AnimationAction::new(clip.clone());
                                        action.bindings = bindings;
                                        self.mixer.add_action(action);
                                    }
                                }
                            });
                    });

                    // 播放控制
                    ui.horizontal(|ui| {
                        if ui.button(if self.is_playing { "⏸ Pause" } else { "▶ Play" }).clicked() {
                            self.is_playing = !self.is_playing;
                        }
                        
                        if ui.button("⏹ Stop").clicked() {
                            self.is_playing = false;
                            self.mixer = AnimationMixer::new();
                        }
                    });

                    // 播放速度
                    ui.horizontal(|ui| {
                        ui.label("Speed:");
                        ui.add(egui::Slider::new(&mut self.playback_speed, 0.0..=2.0)
                            .step_by(0.1)
                            .suffix("x"));
                    });

                    // 显示动画信息
                    if current_anim < self.animations.len() {
                        let clip = &self.animations[current_anim];
                        ui.label(format!("Duration: {:.2}s | Tracks: {}", clip.duration, clip.tracks.len()));
                    }
                }

                ui.separator();

                // 信息显示
                ui.heading("ℹ️ Information");
                ui.label(format!("FPS: {:.1}", self.current_fps));
                ui.label(format!("Nodes: {}", self.loaded_nodes.len()));
            });

        // Help Window
        egui::Window::new("Help")
            .default_pos([10.0, 400.0])
            .default_width(200.0)
            .collapsible(true)
            .default_open(false)
            .show(&egui_ctx, |ui| {
                ui.label("🖱️ Mouse Controls:");
                ui.label("  Left Drag: Rotate View");
                ui.label("  Right Drag: Pan");
                ui.label("  Scroll: Zoom");
                ui.separator();
                ui.label("⌨️ Keyboard Shortcuts:");
                ui.label("  Space: Play/Pause");
            });
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    App::new()
        .with_title("glTF Viewer")
        .with_settings(RenderSettings { vsync: false, ..Default::default() })
        .run::<GltfViewer>()
}
