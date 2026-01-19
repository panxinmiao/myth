pub mod input;

use self::input::Input;

use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::assets::AssetServer;
use crate::scene::Scene;
use crate::renderer::{Renderer, settings::RenderSettings};

/// 应用程序更新上下文
/// 包含了这一帧逻辑更新所需的所有数据
pub struct AppUpdateContext<'a> {
    // === 基础系统 ===
    pub window: &'a winit::window::Window,
    pub input: &'a Input,
    pub assets: &'a AssetServer,
    
    // === 可变数据 (用户主要修改这里) ===
    pub scene: &'a mut Scene,
    
    // === 时间信息 ===
    pub time: f32,      // 从启动开始的总秒数
    pub dt: f32,        // 上一帧的间隔秒数
    pub frame_count: u64, // 帧数计数器
}

// 新的回调定义：只接受一个参数
pub type UpdateFn = Box<dyn FnMut(&mut AppUpdateContext)>;

pub struct App {
    window: Option<Arc<Window>>,
    pub title: String,
    pub renderer: Renderer,
    pub assets: AssetServer,
    pub scene: Scene,

    update_fn: Option<UpdateFn>,
    start_time: std::time::Instant,
    last_loop_time: std::time::Instant,
    frame_count: u64,

    input: Input,
}

impl App {
    pub fn new() -> Self {
        let assets = AssetServer::new();
        let scene = Scene::new();
        // 创建默认配置的 Renderer
        let renderer = Renderer::new(RenderSettings::default());

        let now = std::time::Instant::now();
        Self {
            window: None,
            title: "Three-rs Engine".into(),
            renderer,
            assets,
            scene,
            update_fn: None,
            start_time: now,
            last_loop_time: now,
            frame_count: 0,
            input: Input::new(),
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// 允许用户替换 Renderer (配置阶段)
    pub fn with_renderer(mut self, renderer: Renderer) -> Self {
        self.renderer = renderer;
        self
    }

    pub fn with_settings(mut self, settings: RenderSettings) -> Self {
        self.renderer = Renderer::new(settings);
        self
    }

    pub fn set_update_fn<F>(&mut self, f: F) -> &mut Self 
    where
        F: FnMut(&mut AppUpdateContext) + 'static,
    {
        self.update_fn = Some(Box::new(f));
        self
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(&mut self).map_err(Into::into)
    }

    fn update(&mut self) {

        let now = std::time::Instant::now();
        let total_time = now.duration_since(self.start_time).as_secs_f32();
        let dt = now.duration_since(self.last_loop_time).as_secs_f32();
        self.last_loop_time = now;
        self.frame_count += 1;

        if let Some(ref mut update_fn) = self.update_fn {
            if let Some(window) = &self.window {
                // 组装 Context
                let mut ctx = AppUpdateContext {
                    window,
                    input: &self.input,
                    assets: &self.assets,
                    scene: &mut self.scene,
                    time: total_time,
                    dt,
                    frame_count: self.frame_count,
                };

                // 调用用户回调
                update_fn(&mut ctx);
            }
        }

        self.input.end_frame();
        self.scene.update();
    }

    fn render(&mut self) {
        if self.window.is_some()
            && let Some(cam_id) = self.scene.active_camera {
                // 需要先获取 camera 的引用，然后用 &mut self.scene
                if let Some(node) = self.scene.get_node(cam_id)
                    && let Some(camera_idx) = node.camera
                        && let Some(camera) = self.scene.cameras.get(camera_idx) {
                            let time_seconds = self.last_loop_time
                                .duration_since(self.start_time)
                                .as_secs_f32();
                            // 克隆 camera 以避免借用冲突
                            let camera_clone = camera.clone();
                            self.renderer.render(
                                &mut self.scene, 
                                &camera_clone, 
                                &self.assets, 
                                time_seconds,
                                &[], // 无额外渲染节点
                            );
                        }
            }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let window_attributes = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = event_loop.create_window(window_attributes).expect("Failed to create window");
        let window = Arc::new(window);
        self.window = Some(window.clone());

        // 延迟初始化 Renderer
        log::info!("Initializing Renderer Backend...");

        if let Err(e) = pollster::block_on(self.renderer.init(window)) {
            log::error!("Fatal Renderer Error: {}", e);
            event_loop.exit(); 
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                let scale_factor = self.window.as_ref().map(|w| w.scale_factor() as f32).unwrap_or(1.0);
                self.renderer.resize(physical_size.width, physical_size.height, scale_factor);

                self.input.handle_resize(physical_size.width, physical_size.height);

                if physical_size.height > 0 {
                    let new_aspect = physical_size.width as f32 / physical_size.height as f32;
                    let camera_idx = self.scene.active_camera
                        .and_then(|node_id| self.scene.get_node(node_id))
                        .and_then(|node| node.camera);

                    if let Some(idx) = camera_idx
                        && let Some(camera) = self.scene.cameras.get_mut(idx) {
                            camera.aspect = new_aspect;
                            camera.update_projection_matrix();
                        }
                }
            }
            WindowEvent::RedrawRequested => {
                self.update();
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.handle_cursor_move(position.x, position.y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input.handle_mouse_input(state, button);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.handle_mouse_wheel(delta);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}