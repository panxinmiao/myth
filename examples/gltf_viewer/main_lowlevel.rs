//! glTF Viewer 示例
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
//! - `UiPass` 是外部代码，实现了 `RenderNode` trait
//! - 通过 `render_with_nodes(&[(RenderStage::UI, &ui_pass)])` 注入渲染流程
//! - 引擎核心完全不依赖 egui

mod ui_pass;

use std::sync::Arc;
use std::path::PathBuf;
use glam::Vec3;

use three::resources::Input;
use three::app::winit::input_adapter;
use three::renderer::graph::RenderStage;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use three::assets::{AssetServer, GltfLoader};
use three::scene::{Scene, Camera, light, NodeHandle};
use three::renderer::{Renderer, settings::RenderSettings};
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;

use ui_pass::UiPass;

/// 应用状态
struct ViewerState {
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

impl Default for ViewerState {
    fn default() -> Self {
        Self {
            gltf_node: None,
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
}

/// glTF Viewer 应用
struct GltfViewer {
    window: Option<Arc<Window>>,
    renderer: Renderer,
    assets: AssetServer,
    scene: Scene,
    input: Input,
    
    // UI Pass (外部插件)
    ui_pass: Option<UiPass>,
    
    // 应用状态
    state: ViewerState,
    
    // 时间
    start_time: std::time::Instant,
    last_loop_time: std::time::Instant,
}

impl GltfViewer {
    fn new() -> anyhow::Result<Self> {
        let mut assets = AssetServer::new();
        let mut scene = Scene::new();
        let renderer = Renderer::new(RenderSettings {
            vsync: false,
            ..Default::default()
        });

        // 加载环境贴图
        let env_texture_handle = assets.load_cube_texture_from_files(
            [
                "examples/assets/Park2/posx.jpg",
                "examples/assets/Park2/negx.jpg",
                "examples/assets/Park2/posy.jpg",
                "examples/assets/Park2/negy.jpg",
                "examples/assets/Park2/posz.jpg",
                "examples/assets/Park2/negz.jpg",
            ],
            three::ColorSpace::Srgb
        )?;

        let env_texture = assets.get_texture_mut(env_texture_handle).unwrap();
        env_texture.generate_mipmaps = true;
        scene.environment.set_env_map(Some((env_texture_handle.into(), env_texture)));

        // 添加灯光
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        scene.add_light(light);

        // 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);
        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 1.0, 5.0);
            node.transform.look_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        }
        scene.active_camera = Some(cam_node_id);

        let now = std::time::Instant::now();
        Ok(Self {
            window: None,
            renderer,
            assets,
            scene,
            input: Input::new(),
            ui_pass: None,
            state: ViewerState::default(),
            start_time: now,
            last_loop_time: now,
        })
    }

    fn update(&mut self, window: &Window) {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_loop_time).as_secs_f32();
        self.last_loop_time = now;

        // 更新 FPS
        if let Some(fps) = self.state.fps_counter.update() {
            self.state.current_fps = fps;
        }

        // 相机控制
        if let Some((transform, camera)) = self.scene.query_main_camera_bundle() {
            self.state.controls.update(transform, &self.input, camera.fov.to_degrees(), dt);
        }

        // UI 逻辑 - 先处理 UI 帧，再渲染 UI
        if let Some(ui_pass) = &self.ui_pass {
            ui_pass.begin_frame(window);
        }
        
        // 渲染 UI（在单独的作用域中获取 egui_ctx）
        if let Some(ui_pass) = &self.ui_pass {
            let egui_ctx = ui_pass.context().clone();
            self.render_ui(&egui_ctx);
        }
        
        if let Some(ui_pass) = &self.ui_pass {
            ui_pass.end_frame(window);
        }

        // 处理待加载的模型
        if let Some(path) = self.state.pending_load.take() {
            self.load_model(&path);
        }

        // 更新窗口标题
        let title = if let Some(path) = &self.state.model_path {
            format!("glTF Viewer - {} | FPS: {:.0}", 
                path.file_name().unwrap_or_default().to_string_lossy(),
                self.state.current_fps)
        } else {
            format!("glTF Viewer | FPS: {:.0}", self.state.current_fps)
        };
        window.set_title(&title);

        self.input.start_frame();
        self.scene.update(&self.input, dt);
    }

    fn render(&mut self) {
        if let Some(cam_id) = self.scene.active_camera {
            if let Some(camera) = self.scene.cameras.get(cam_id) {
                let time_seconds = self.last_loop_time
                    .duration_since(self.start_time)
                    .as_secs_f32();
                let render_camera = camera.extract_render_camera();
                
                // 使用新的 FrameBuilder API
                if let Some(prepared_frame) = self.renderer.begin_frame(
                    &mut self.scene,
                    &render_camera,
                    &self.assets,
                    time_seconds,
                ) {
                    // 注入 UI Pass
                    if let Some(ui_pass) = &self.ui_pass {
                        prepared_frame.render_with_nodes(&[(RenderStage::UI, ui_pass as &dyn three::renderer::graph::RenderNode)]);
                    } else {
                        prepared_frame.render_default();
                    }
                }
                
                // 定期清理资源
                self.renderer.maybe_prune();
            }
        }
    }

    fn load_model(&mut self, path: &PathBuf) {
        // 清理旧模型
        if let Some(gltf_node) = self.state.gltf_node {
            self.scene.remove_node(gltf_node);
        }
        self.state.gltf_node = None;
        self.state.animations.clear();
        // 加载新模型
        match GltfLoader::load(path, &mut self.assets, &mut self.scene) {
            Ok(gltf_node) => {
                self.state.gltf_node = Some(gltf_node);
                self.state.model_path = Some(path.clone());
                self.state.current_animation = 0;

                // 自动播放第一个动画
                if let Some(mixer) = self.scene.animation_mixers.get_mut(gltf_node) {
                    self.state.animations = mixer.list_animations();

                    if let Some(clip_name) = self.state.animations.first() {
                        println!("Auto-playing animation: {}", clip_name);
                        mixer.play(clip_name);
                    }
                }

                log::info!("Loaded model: {:?}", path);
            }
            Err(e) => {
                log::error!("Failed to load model: {}", e);
            }
        }
    }

    fn render_ui(&mut self, egui_ctx: &egui::Context) {
        // 主控制面板
        egui::Window::new("控制面板")
            .default_pos([10.0, 10.0])
            .default_width(280.0)
            .show(egui_ctx, |ui| {
                // 文件加载部分
                ui.heading("📁 文件");
                ui.horizontal(|ui| {
                    if ui.button("打开 glTF/glb 文件...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("glTF", &["gltf", "glb"])
                            .pick_file()
                        {
                            self.state.pending_load = Some(path);
                        }
                    }
                });

                if let Some(path) = &self.state.model_path {
                    ui.label(format!("当前文件: {}", 
                        path.file_name().unwrap_or_default().to_string_lossy()));
                } else {
                    ui.label("未加载模型");
                }

                ui.separator();

                // 动画控制部分
                ui.heading("🎬 动画");
                
                if self.state.animations.is_empty() {
                    ui.label("无可用动画");
                } else {
                    // 动画选择
                    let current_anim = self.state.current_animation;
                    let anim_name = if current_anim < self.state.animations.len() {
                        self.state.animations[current_anim].clone()
                    } else {
                        "Select Animation".to_string()
                    };
                    
                    ui.horizontal(|ui| {
                        ui.label("Animation:");
                        egui::ComboBox::from_id_salt("animation_selector")
                            .selected_text(&anim_name)
                            .show_ui(ui, |ui| {
                                for (i, clip) in self.state.animations.iter().enumerate() {
                                    if ui.selectable_value(&mut self.state.current_animation, i, clip).changed() {
                                        // 切换动画
                                        if let Some(gltf_node) = self.state.gltf_node {
                                            println!("click to animation: {}", clip);
                                            if let Some(mixer) = self.scene.animation_mixers.get_mut(gltf_node) {
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
                        if ui.button(if self.state.is_playing { "⏸ 暂停" } else { "▶ 播放" }).clicked() {
                            self.state.is_playing = !self.state.is_playing;
                        }
                        
                        if let Some(gltf_node) = self.state.gltf_node {
                            if let Some(mixer) = self.scene.animation_mixers.get_mut(gltf_node) {
                                if self.state.is_playing {
                                    mixer.play(&self.state.animations[self.state.current_animation]);
                                } else {
                                    mixer.stop_all();
                                }
                            }
                        }
                    });

                    // 播放速度
                    ui.horizontal(|ui| {
                        ui.label("速度:");
                        ui.add(egui::Slider::new(&mut self.state.playback_speed, 0.0..=2.0)
                            .step_by(0.1)
                            .suffix("x"));
                    });
                }

                ui.separator();

                // 信息显示
                ui.heading("ℹ️ 信息");
                ui.label(format!("FPS: {:.1}", self.state.current_fps));
            });

        // 帮助提示
        egui::Window::new("帮助")
            .default_pos([10.0, 400.0])
            .default_width(200.0)
            .collapsible(true)
            .default_open(false)
            .show(egui_ctx, |ui| {
                ui.label("🖱️ 鼠标控制:");
                ui.label("  左键拖动: 旋转视角");
                ui.label("  右键拖动: 平移");
                ui.label("  滚轮: 缩放");
                ui.separator();
                ui.label("⌨️ 快捷键:");
                ui.label("  空格: 播放/暂停");
            });
    }
}

impl ApplicationHandler for GltfViewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let window_attributes = Window::default_attributes()
            .with_title("glTF Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = event_loop.create_window(window_attributes).expect("Failed to create window");
        let window = Arc::new(window);
        self.window = Some(window.clone());

        // 初始化 Renderer
        log::info!("Initializing Renderer Backend...");
        let size = window.inner_size();
        if let Err(e) = pollster::block_on(self.renderer.init(window.clone(), size.width, size.height)) {
            log::error!("Fatal Renderer Error: {}", e);
            event_loop.exit();
            return;
        }

        // 初始化 UI Pass (在 Renderer 初始化后)
        if let (Some(device), Some(format)) = (self.renderer.device(), self.renderer.surface_format()) {
            self.ui_pass = Some(UiPass::new(device, format, &window));
            log::info!("UI Pass initialized");
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // UI 输入处理
        if let Some(ui_pass) = &self.ui_pass {
            if let Some(window) = &self.window {
                if ui_pass.handle_input(window, &event) {
                    return; // 事件被 UI 消耗
                }
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                let scale_factor = self.window.as_ref().map(|w| w.scale_factor() as f32).unwrap_or(1.0);
                self.renderer.resize(physical_size.width, physical_size.height, scale_factor);
                
                if let Some(ui_pass) = &self.ui_pass {
                    ui_pass.resize(physical_size.width, physical_size.height, scale_factor);
                }

                self.input.inject_resize(physical_size.width, physical_size.height);

                if physical_size.height > 0 {
                    let new_aspect = physical_size.width as f32 / physical_size.height as f32;
                    if let Some(node_handle) = self.scene.active_camera {
                        if let Some(camera) = self.scene.cameras.get_mut(node_handle) {
                            camera.aspect = new_aspect;
                            camera.update_projection_matrix();
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // 克隆 window 引用以避免借用冲突
                let window = self.window.clone();
                if let Some(window) = &window {
                    self.update(window);
                }
                self.render();
                if let Some(window) = &window {
                    window.request_redraw();
                }
            }
            _ => {
                // 使用 input_adapter 将 winit 事件翻译到引擎 Input
                input_adapter::process_window_event(&mut self.input, &event);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    let mut app = GltfViewer::new()?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;
    
    Ok(())
}
