pub mod input;

use self::input::Input;

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::assets::AssetServer;
use crate::scene::Scene;
use crate::renderer::{Renderer, settings::RenderSettings};
use crate::renderer::graph::RenderNode;

/// 应用上下文：提供给用户的所有引擎资源
/// 在 init, update, on_event 中传递给用户
pub struct AppContext<'a> {
    pub window: &'a Window,
    pub renderer: &'a mut Renderer, // 允许用户 resize 或访问 device
    pub scene: &'a mut Scene,
    pub assets: &'a mut AssetServer,
    pub input: &'a Input,
    
    // 时间状态
    pub time: f32,       // 启动以来的秒数
    pub dt: f32,         // 上一帧间隔秒数
    pub frame_count: u64,
}

/// 用户程序必须实现的 Trait (替代旧的闭包回调)
pub trait AppHandler: Sized + 'static {
    /// 初始化：引擎资源（Window, Renderer）准备好后调用
    /// 用户应在此处创建自己的状态（如 UiPass, GameState）
    fn init(ctx: &mut AppContext) -> Self;

    /// 窗口事件：返回 true 表示事件被用户消耗了，引擎不应再处理（例如 UI 捕获了鼠标）
    fn on_event(&mut self, _ctx: &mut AppContext, _event: &WindowEvent) -> bool {
        false 
    }

    /// 逻辑更新：每一帧调用
    fn update(&mut self, ctx: &mut AppContext);

    /// 渲染扩展：返回需要注入到 RenderGraph 的额外节点（例如 UI Pass）
    fn extra_render_nodes(&self) -> Vec<&dyn RenderNode> {
        Vec::new() 
    }
}

/// 默认的空 Handler，用于简单的示例
pub struct DefaultHandler;
impl AppHandler for DefaultHandler {
    fn init(_ctx: &mut AppContext) -> Self { Self }
    fn update(&mut self, _ctx: &mut AppContext) {}
}

/// App 构建器
/// 负责收集配置，并不持有运行时状态
pub struct App {
    title: String,
    render_settings: RenderSettings,
}

impl App {
    pub fn new() -> Self {
        Self {
            title: "Three-rs Engine".into(),
            render_settings: RenderSettings::default(),
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn with_settings(mut self, settings: RenderSettings) -> Self {
        self.render_settings = settings;
        self
    }

    /// 启动引擎，接管主循环
    /// H: 用户自定义的状态结构体
    pub fn run<H: AppHandler>(self) -> anyhow::Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut runner = AppRunner::<H>::new(self.title, self.render_settings);
        event_loop.run_app(&mut runner).map_err(Into::into)
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// 实际的应用运行器 (实现了 winit ApplicationHandler)
/// 它持有所有的引擎子系统以及用户的状态 H
struct AppRunner<H: AppHandler> {
    // === 配置 ===
    title: String,
    render_settings: RenderSettings,

    // === 系统 ===
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>, // 使用 Option 因为要在 resumed 中初始化
    scene: Scene,
    assets: AssetServer,
    input: Input,

    // === 运行时状态 ===
    user_state: Option<H>, // 在 init 后才存在
    start_time: Instant,
    last_loop_time: Instant,
    frame_count: u64,
}

impl<H: AppHandler> AppRunner<H> {
    fn new(title: String, render_settings: RenderSettings) -> Self {
        let now = Instant::now();
        Self {
            title,
            render_settings,
            window: None,
            renderer: None,
            scene: Scene::new(),
            assets: AssetServer::new(),
            input: Input::new(),
            user_state: None,
            start_time: now,
            last_loop_time: now,
            frame_count: 0,
        }
    }

    // 内部辅助：更新逻辑
    fn update_logic(&mut self) {
        let now = Instant::now();
        let total_time = now.duration_since(self.start_time).as_secs_f32();
        let dt = now.duration_since(self.last_loop_time).as_secs_f32();
        self.last_loop_time = now;
        self.frame_count += 1;

        if let (Some(window), Some(renderer), Some(user_state)) = (
            &self.window, 
            &mut self.renderer, 
            &mut self.user_state
        ) {
            // 1. 构造 Context
            let mut ctx = AppContext {
                window,
                renderer,
                scene: &mut self.scene,
                assets: &mut self.assets,
                input: &self.input,
                time: total_time,
                dt,
                frame_count: self.frame_count,
            };

            // 2. 用户更新
            user_state.update(&mut ctx);
        }

        // 3. 引擎内部系统更新
        self.input.end_frame();
        self.scene.update();
    }

    // 内部辅助：渲染逻辑
    fn render_frame(&mut self) {
        if let (Some(renderer), Some(user_state)) = (&mut self.renderer, &self.user_state) {
            
            // 获取当前激活相机
            let Some(node_handle) = self.scene.active_camera else {
                return;
            };

            // 直接获取相机组件的 clone
            let camera = if let Some(cam) = self.scene.cameras.get(node_handle) {
                cam.clone()
            } else {
                return;
            };

            let time_seconds = self.last_loop_time.duration_since(self.start_time).as_secs_f32();
            
            // 获取用户注入的 Render Nodes (例如 UI Pass)
            let extra_nodes = user_state.extra_render_nodes();

            renderer.render(
                &mut self.scene, 
                &camera, 
                &self.assets, 
                time_seconds,
                &extra_nodes, // <--- 关键：注入用户 Pass
            );
        }
    }
}

impl<H: AppHandler> ApplicationHandler for AppRunner<H> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        // 1. 创建窗口
        let window_attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = event_loop.create_window(window_attributes).expect("Failed to create window");
        let window = Arc::new(window);
        self.window = Some(window.clone());

        // 2. 初始化 Renderer
        log::info!("Initializing Renderer Backend...");
        let mut renderer = Renderer::new(self.render_settings.clone());
        if let Err(e) = pollster::block_on(renderer.init(window.clone())) {
            log::error!("Fatal Renderer Error: {}", e);
            event_loop.exit();
            return;
        }
        self.renderer = Some(renderer);

        // 3. 初始化用户状态 (H::init)
        // 构造一个临时的 Context 用于初始化
        let now = Instant::now();
        let mut ctx = AppContext {
            window: &window,
            renderer: self.renderer.as_mut().unwrap(),
            scene: &mut self.scene,
            assets: &mut self.assets,
            input: &self.input,
            time: 0.0,
            dt: 0.0,
            frame_count: 0,
        };
        
        self.start_time = now;
        self.last_loop_time = now;
        
        // 调用用户 Init
        self.user_state = Some(H::init(&mut ctx));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // 确保系统已完全初始化
        if self.window.is_none() || self.renderer.is_none() || self.user_state.is_none() {
            return;
        }

        let window = self.window.as_ref().unwrap();

        // 构造 Context 用于事件处理
        let consumed = {
            let renderer = self.renderer.as_mut().unwrap();
            let user_state = self.user_state.as_mut().unwrap();
            
            let mut ctx = AppContext {
                window,
                renderer,
                scene: &mut self.scene,
                assets: &mut self.assets,
                input: &self.input, // 借用 self.input
                time: self.last_loop_time.duration_since(self.start_time).as_secs_f32(),
                dt: 0.0,
                frame_count: self.frame_count,
            };

            user_state.on_event(&mut ctx, &event)
        };

        // B. 如果未被用户消耗，引擎默认处理
        if !consumed {
            self.input.process_event(&event);
            
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(physical_size) => {
                    let scale_factor = window.scale_factor() as f32;
                    // 更新 Renderer
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(physical_size.width, physical_size.height, scale_factor);
                    }
                    // 更新 Input
                    self.input.handle_resize(physical_size.width, physical_size.height);
                    
                    // 更新相机长宽比
                    if physical_size.height > 0 {
                        let new_aspect = physical_size.width as f32 / physical_size.height as f32;
                        // 查找 active camera 并更新
                        if let Some(node_handle) = self.scene.active_camera {
                            if let Some(camera) = self.scene.cameras.get_mut(node_handle) {
                                camera.aspect = new_aspect;
                                camera.update_projection_matrix();
                            }
                        }
                    }
                },
                WindowEvent::RedrawRequested => {
                    self.update_logic();
                    self.render_frame();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                },
                _ => {}
            }
        } else {
            // 即使被 consumed，如果是 Resize 或 Close，通常系统也需要响应
            // 这里根据需求决定。为了安全，Resize 事件通常建议广播。
            if let WindowEvent::Resized(ps) = event {
                 let scale_factor = window.scale_factor() as f32;
                 if let Some(renderer) = &mut self.renderer {
                    renderer.resize(ps.width, ps.height, scale_factor);
                 }
            }
            if let WindowEvent::RedrawRequested = event {
                 self.update_logic();
                 self.render_frame();
                 if let Some(window) = &self.window {
                    window.request_redraw();
                 }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}