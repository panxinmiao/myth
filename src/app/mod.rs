use std::sync::Arc;
use std::time::Instant;

use slotmap::{new_key_type, SlotMap};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::assets::AssetServer;
use crate::scene::Scene;
use crate::renderer::{Renderer, settings::RenderSettings};
use crate::renderer::graph::RenderNode;
use crate::resources::input::Input;

new_key_type! {
    pub struct SceneHandle;
}

/// 负责管理场景生命周期的子系统
pub struct SceneManager {
    scenes: SlotMap<SceneHandle, Scene>,
    active_scene: Option<SceneHandle>,
}

impl SceneManager {
    pub fn new() -> Self {
        Self {
            scenes: SlotMap::with_key(),
            active_scene: None,
        }
    }

    /// 创建一个新场景，返回其句柄
    pub fn create_scene(&mut self) -> SceneHandle {
        self.scenes.insert(Scene::new())
    }

    /// 删除场景（带安全检查）
    pub fn remove_scene(&mut self, handle: SceneHandle) {
        if self.active_scene == Some(handle) {
            self.active_scene = None;
            log::warn!("Active scene was removed! Screen will be empty.");
        }
        self.scenes.remove(handle);
    }

    /// [Helper] 设置当前激活场景
    pub fn set_active(&mut self, handle: SceneHandle) {
        if self.scenes.contains_key(handle) {
            self.active_scene = Some(handle);
        } else {
            log::error!("Attempted to set invalid SceneHandle as active.");
        }
    }

    /// 创建并设置一个新的激活场景，返回其可变引用
    pub fn create_active(&mut self) -> &mut Scene {
        let handle = self.create_scene();
        self.set_active(handle);
        self.get_scene_mut(handle).unwrap()
    }

    /// 获取当前激活场景的句柄
    pub fn active_handle(&self) -> Option<SceneHandle> {
        self.active_scene
    }

    /// 获取任意场景的引用
    pub fn get_scene(&self, handle: SceneHandle) -> Option<&Scene> {
        self.scenes.get(handle)
    }

    /// 获取任意场景的可变引用
    pub fn get_scene_mut(&mut self, handle: SceneHandle) -> Option<&mut Scene> {
        self.scenes.get_mut(handle)
    }

    /// 快捷方式：获取当前激活场景
    pub fn active_scene(&self) -> Option<&Scene> {
        self.active_scene.and_then(|h| self.scenes.get(h))
    }

    /// 获取当前激活的场景（可变引用）
    pub fn active_scene_mut(&mut self) -> Option<&mut Scene> {
        self.active_scene.and_then(|h| self.scenes.get_mut(h))
    }
}

/// 应用上下文：提供给用户的所有引擎资源
/// 在 init, update, on_event 中传递给用户
pub struct AppContext<'a> {
    pub window: &'a Window,
    pub renderer: &'a mut Renderer,

    pub scenes: &'a mut SceneManager,
    pub assets: &'a mut AssetServer,
    pub input: &'a Input,
    
    // 时间状态
    pub time: f32,       // 启动以来的秒数
    pub dt: f32,         // 上一帧间隔秒数
    pub frame_count: u64,
}


/// 用户程序必须实现的 Trait
pub trait AppHandler: Sized + 'static {
    fn init(ctx: &mut AppContext) -> Self;

    fn on_event(&mut self, _ctx: &mut AppContext, _event: &WindowEvent) -> bool {
        false 
    }

    fn update(&mut self, _ctx: &mut AppContext){}

    fn extra_render_nodes(&self) -> Vec<&dyn RenderNode> {
        Vec::new() 
    }
}

/// 默认的空 Handler
pub struct DefaultHandler;
impl AppHandler for DefaultHandler {
    fn init(ctx: &mut AppContext) -> Self {
        // 示例：默认创建一个场景，防止黑屏
        ctx.scenes.create_active();
        Self 
    }
    fn update(&mut self, _ctx: &mut AppContext) {}
}

/// App 构建器
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

/// 实际的应用运行器
struct AppRunner<H: AppHandler> {
    // === 配置 ===
    title: String,
    render_settings: RenderSettings,

    // === 系统 ===
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    
    // === 场景管理 ===
    scene_manager: SceneManager,

    assets: AssetServer,
    input: Input,

    // === 运行时状态 ===
    user_state: Option<H>,
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
            scene_manager: SceneManager::new(),
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
                scenes: &mut self.scene_manager,
                assets: &mut self.assets,
                input: &self.input,
                time: total_time,
                dt,
                frame_count: self.frame_count,
            };

            // 2. 用户更新
            user_state.update(&mut ctx);
        }
        
        // 只更新当前激活的场景
        if let Some(handle) = self.scene_manager.active_handle() {
            if let Some(scene) = self.scene_manager.scenes.get_mut(handle) {
                scene.update(&self.input, dt);
            }
        }

        // 3. 引擎内部系统更新
        self.input.end_frame();
    }

    // 内部辅助：渲染逻辑
    fn render_frame(&mut self) {
        if let (Some(renderer), Some(user_state)) = (&mut self.renderer, &self.user_state) {
            
            // 获取当前激活场景
            let Some(scene_handle) = self.scene_manager.active_handle() else {
                return;
            };
            let Some(scene) = self.scene_manager.scenes.get_mut(scene_handle) else { return; };

            // 获取当前激活相机
            let Some(camera_node) = scene.active_camera else {
                return;
            };

            // 直接获取相机组件的 clone
            let render_camera = if let Some(cam) = scene.cameras.get(camera_node) {
                cam.extract_render_camera()
            } else {
                return;
            };

            let time_seconds = self.last_loop_time.duration_since(self.start_time).as_secs_f32();
            
            // 获取用户注入的 Render Nodes
            let extra_nodes = user_state.extra_render_nodes();

            renderer.render(
                scene,  // 传入 active scene
                render_camera, 
                &self.assets, 
                time_seconds,
                &extra_nodes,
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
        let now = Instant::now();
        let mut ctx = AppContext {
            window: &window,
            renderer: self.renderer.as_mut().unwrap(),
            scenes: &mut self.scene_manager,
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
                scenes: &mut self.scene_manager,
                assets: &mut self.assets,
                input: &self.input, 
                time: self.last_loop_time.duration_since(self.start_time).as_secs_f32(),
                dt: 0.0,
                frame_count: self.frame_count,
            };

            user_state.on_event(&mut ctx, &event)
        };

        if !consumed {
            self.input.process_event(&event);
            
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(physical_size) => {
                    let scale_factor = window.scale_factor() as f32;
                    
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(physical_size.width, physical_size.height, scale_factor);
                    }
                    self.input.handle_resize(physical_size.width, physical_size.height);
                    
                    // 更新当前激活场景相机的长宽比
                    if physical_size.height > 0 {
                        let new_aspect = physical_size.width as f32 / physical_size.height as f32;
                        
                        // 先检查是否有 active_scene
                        if let Some(scene_handle) = self.scene_manager.active_handle() {
                             if let Some(scene) = self.scene_manager.scenes.get_mut(scene_handle) {
                                // 再检查 active_camera
                                if let Some(node_handle) = scene.active_camera {
                                    if let Some(camera) = scene.cameras.get_mut(node_handle) {
                                        camera.aspect = new_aspect;
                                        camera.update_projection_matrix();
                                    }
                                }
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
             // 即使被 consumed，如果是 Resize 或 Redraw，引擎仍需响应
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