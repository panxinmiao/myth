//! 应用程序抽象
//! 
//! 提供顶层的 App 抽象，管理窗口、渲染器、场景和资产

use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::assets::AssetServer;
use crate::scene::Scene;
use crate::render::Renderer;
use thunderdome::Index;

pub type NodeId = Index;

/// 更新回调函数类型
pub type UpdateFn = Box<dyn FnMut(&mut Scene, &AssetServer, f32)>;

/// 主应用程序结构
pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,

    // 核心逻辑组件
    pub assets: AssetServer,
    pub scene: Scene,

    // 当前激活的相机节点 ID
    pub active_camera: Option<NodeId>,

    // 自定义更新回调
    update_fn: Option<UpdateFn>,

    // 时间追踪
    start_time: Option<std::time::Instant>,
    last_frame_time: f32,
}

impl App {
    /// 创建新的应用程序实例
    pub fn new() -> Self {
        let assets = AssetServer::new();
        let scene = Scene::new();

        Self {
            window: None,
            renderer: None,
            assets,
            scene,
            active_camera: None,
            update_fn: None,
            start_time: None,
            last_frame_time: 0.0,
        }
    }

    /// 设置自定义更新回调函数
    /// 
    /// 回调函数会在每帧渲染前被调用，接收：
    /// - scene: 场景的可变引用
    /// - assets: 资产服务器的不可变引用
    /// - delta_time: 距离应用启动的时间（秒）
    pub fn set_update_fn<F>(&mut self, f: F) -> &mut Self 
    where
        F: FnMut(&mut Scene, &AssetServer, f32) + 'static,
    {
        self.update_fn = Some(Box::new(f));
        self
    }

    /// 运行应用程序事件循环
    pub fn run(mut self) -> anyhow::Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(&mut self)?;
        Ok(())
    }

    /// 更新场景逻辑
    fn update(&mut self) {
        // 初始化计时器
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        // 计算时间
        if let Some(start) = self.start_time {
            self.last_frame_time = start.elapsed().as_secs_f32();
        }

        // 调用用户自定义更新函数
        if let Some(ref mut update_fn) = self.update_fn {
            update_fn(&mut self.scene, &self.assets, self.last_frame_time);
        }

        // 更新场景变换
        self.scene.update_matrix_world();
        self.scene.update_cameras();
    }

    /// 渲染场景
    fn render(&mut self) {
        if let (Some(renderer), Some(_window)) = (&mut self.renderer, &self.window) {
            // 获取激活的相机
            if let Some(cam_id) = self.active_camera {
                // 先更新场景
                self.scene.update_matrix_world();
                
                // 克隆scene用于更新相机（避免借用冲突）
                let scene_ref = &self.scene;
                
                if let Some(node) = scene_ref.get_node(cam_id) {
                    if let Some(camera_idx) = node.camera {
                        if let Some(camera) = self.scene.cameras.get(camera_idx) {
                            renderer.render(&self.scene, camera, &self.assets);
                        }
                    } else {
                        log::warn!("Active camera node has no Camera component!");
                    }
                }
            }
        }
    }
}

// Winit 0.30+ 核心：实现 ApplicationHandler
impl ApplicationHandler for App {
    // 1. 应用恢复/启动时调用（创建窗口的地方）
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // 如果窗口已经存在，直接返回
        if self.window.is_some() {
            return;
        }

        // 创建窗口
        let window_attributes = Window::default_attributes()
            .with_title("Three-rs Engine")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = event_loop.create_window(window_attributes).unwrap();
        let window = Arc::new(window);
        self.window = Some(window.clone());

        // 初始化渲染器
        let renderer = pollster::block_on(Renderer::new(window.clone()));
        self.renderer = Some(renderer);
        
        log::info!("App Resumed: Window and Renderer created");
    }

    // 2. 窗口事件处理
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, exiting...");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(physical_size.width, physical_size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                // 每一帧的核心循环
                self.update();
                self.render();
                
                // 请求下一帧重绘
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    // 3. 处理空闲时间
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
