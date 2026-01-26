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
use three::engine::FrameState;
use winit::event::WindowEvent;

use three::app::winit::{App, AppHandler};
use three::assets::GltfLoader;
use three::scene::{Camera, NodeHandle, light};
use three::renderer::graph::RenderNode;
use three::renderer::settings::RenderSettings;
use three::{OrbitControls, ThreeEngine};
use three::utils::fps_counter::FpsCounter;

use ui_pass::UiPass;
use winit::window::Window;

/// glTF Viewer 应用状态
struct GltfViewer {
    /// UI Pass (egui 渲染)
    ui_pass: UiPass,
    
    /// 当前加载的模型根节点
    gltf_node: Option<NodeHandle>,
    // /// 可用的动画列表
    animations: Vec<String>,
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
    fn init(engine: &mut ThreeEngine, window: &Arc<Window>) -> Self {
        // 1. 创建 UI Pass
        let wgpu_ctx = engine.renderer.wgpu_ctx().expect("Renderer not initialized");
        let ui_pass = UiPass::new(
            &wgpu_ctx.device,
            wgpu_ctx.config.format,
            window,
        );

        // 2. 加载环境贴图
        let env_texture_handle = engine.assets.load_cube_texture_from_files(
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

        let scene = engine.scene_manager.create_active();

        let env_texture = engine.assets.get_texture_mut(env_texture_handle).unwrap();
        env_texture.generate_mipmaps = true;

        scene.environment.set_env_map(Some((env_texture_handle.into(), &env_texture)));
        scene.environment.set_intensity(1.0);

        scene.environment.set_ambient_color(Vec3::splat(0.6));

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

        Self {
            ui_pass,
            gltf_node: None,
            // model_root_node: None,
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

    fn on_event(&mut self, _engine: &mut ThreeEngine, window: &Arc<Window>, event: &WindowEvent) -> bool {
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
        let Some(scene) = engine.scene_manager.active_scene_mut() else{
            return;
        };
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
            window.set_title(&title);
        }

        // 2. 更新动画播放速度（通过 action.time_scale 控制）
        if let Some(gltf_node) = self.gltf_node {
            if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
                mixer.time_scale = self.playback_speed;
            }
        }

        // 3. 相机控制
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls.update(transform, &engine.input, camera.fov.to_degrees(), frame.dt);
        }

        // 4. 构建 UI
        self.ui_pass.begin_frame(window);
        self.render_ui(engine);
        self.ui_pass.end_frame(window);

        // 5. 处理待加载的模型
        if let Some(path) = self.pending_load.take() {
            self.load_model(&path, engine);
        }
    }

    fn extra_render_nodes(&self) -> Vec<&dyn RenderNode> {
        vec![&self.ui_pass]
    }
}

impl GltfViewer {
    fn load_model(&mut self, path: &PathBuf, engine: &mut ThreeEngine) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else{
            return;
        };
        // 清理旧模型
        if let Some(gltf_node) = self.gltf_node {
            scene.remove_node(gltf_node);
        }
        self.gltf_node = None;
        // self.animations.clear();
        // self.model_root_node = None;

        // 加载新模型
        match GltfLoader::load(path, &mut engine.assets, scene) {
            Ok(gltf_node) => {
                self.gltf_node = Some(gltf_node);
                // self.animations = animations.iter().map(|c| Arc::new(c.clone())).collect();
                self.model_path = Some(path.clone());
                self.current_animation = 0;

                // 自动播放第一个动画
                if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
                    self.animations = mixer.list_animations();

                    if let Some(clip_name) = self.animations.first() {
                        println!("Auto-playing animation: {}", clip_name);
                        mixer.play(clip_name);
                    }
                }
        

                scene.update_subtree(gltf_node);
                if let Some(bbox) = scene.get_bbox_of_node(gltf_node, &engine.assets) {
                    let center = bbox.center();
                    let radius = bbox.size().length() * 0.5;
                    if let Some((_transform, camera)) = scene.query_main_camera_bundle() {
                        camera.near = radius / 100.0;
                        camera.update_projection_matrix();
                        self.controls.set_target(center);
                        self.controls.set_position(center + Vec3::new(0.0, radius, radius * 2.5));
                    }
                }
                log::info!("Loaded model: {:?}", path);
            }
            Err(e) => {
                log::error!("Failed to load model: {}", e);
            }
        }
    }

    fn render_ui(&mut self, engine: &mut ThreeEngine) {
        let egui_ctx = self.ui_pass.context().clone();
        let Some(scene) = engine.scene_manager.active_scene_mut() else{
            return;
        };
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
                        self.animations[current_anim].clone()
                    } else {
                        "Select Animation".to_string()
                    };
                    
                    ui.horizontal(|ui| {
                        ui.label("Animation:");
                        egui::ComboBox::from_id_salt("animation_selector")
                            .selected_text(&anim_name)
                            .show_ui(ui, |ui| {
                                for (i, clip) in self.animations.iter().enumerate() {
                                    if ui.selectable_value(&mut self.current_animation, i, clip).changed() {
                                        // 切换动画
                                        if let Some(gltf_node) = self.gltf_node {
                                            println!("click to animation: {}", clip);
                                            if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
                                                mixer.stop_all();
                                                println!("Switching to animation: {}", clip);
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
                                        mixer.play(&self.animations[self.current_animation]);
                                    } else {
                                        mixer.stop_all();
                                    }
                                }
                            }
                        }
                        
                        // if ui.button("⏹ Stop").clicked() {
                        //     self.is_playing = false;
                        //     if let Some(gltf_node) = self.gltf_node {
                        //         if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
                        //             mixer.stop_all();
                        //         }
                        //     }
                        // }
                    });

                    // 播放速度
                    ui.horizontal(|ui| {
                        ui.label("Speed:");
                        ui.add(egui::Slider::new(&mut self.playback_speed, 0.0..=2.0)
                            .step_by(0.1)
                            .suffix("x"));
                    });

                }

                ui.separator();

                // 信息显示
                ui.heading("Information");
                ui.label(format!("FPS: {:.1}", self.current_fps));
            });

        // Help Window
        // egui::Window::new("Help")
        //     .default_pos([10.0, 400.0])
        //     .default_width(200.0)
        //     .collapsible(true)
        //     .default_open(false)
        //     .show(&egui_ctx, |ui| {
        //         ui.label("🖱️ Mouse Controls:");
        //         ui.label("  Left Drag: Rotate View");
        //         ui.label("  Right Drag: Pan");
        //         ui.label("  Scroll: Zoom");
        //         ui.separator();
        //         ui.label("⌨️ Keyboard Shortcuts:");
        //         ui.label("  Space: Play/Pause");
        //     });
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    App::new()
        .with_title("glTF Viewer")
        .with_settings(RenderSettings { vsync: false, ..Default::default() })
        .run::<GltfViewer>()
}
