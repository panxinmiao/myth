use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::engine::{FrameState, ThreeEngine};
use crate::renderer::graph::RenderNode;
use crate::renderer::settings::RenderSettings;

pub mod input_adapter;


/// 用户程序必须实现的 Trait
pub trait AppHandler: Sized + 'static {
    fn init(_ctx: &mut ThreeEngine, _window: &Arc<Window>) -> Self;

    fn on_event(&mut self, _engine: &mut ThreeEngine, _window: &Arc<Window>, _event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self, _engine: &mut ThreeEngine, _window: &Arc<Window>, _frame: &FrameState) {}

    fn extra_render_nodes(&self) -> Vec<&dyn RenderNode> {
        Vec::new()
    }
}

/// 默认的空 Handler
pub struct DefaultHandler;

impl AppHandler for DefaultHandler {
    fn init(_ctx: &mut ThreeEngine, _window: &Arc<Window>) -> Self {
        Self
    }
    fn update(&mut self, _engine: &mut ThreeEngine, _window: &Arc<Window>,  _frame: &FrameState) {}

}

/// App 构建器
pub struct App {
    title: String,
    render_settings: RenderSettings,
}

impl App {
    #[must_use]
    pub fn new() -> Self {
        Self {
            title: "Three-rs Engine".into(),
            render_settings: RenderSettings::default(),
        }
    }

    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    #[must_use]
    pub fn with_settings(mut self, settings: RenderSettings) -> Self {
        self.render_settings = settings;
        self
    }

    /// 运行应用
    ///
    /// # Errors
    /// 如果事件循环创建或运行失败则返回错误。
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

/// 实际的应用运行器（Winit 适配器）
struct AppRunner<H: AppHandler> {
    title: String,
    render_settings: RenderSettings,

    window: Option<Arc<Window>>,
    engine: Option<ThreeEngine>,
    user_state: Option<H>,

    start_time: Instant,
    last_loop_time: Instant,
}

impl<H: AppHandler> AppRunner<H> {
    fn new(title: String, render_settings: RenderSettings) -> Self {
        let now = Instant::now();
        Self {
            title,
            render_settings,
            window: None,
            engine: None,
            user_state: None,
            start_time: now,
            last_loop_time: now,
        }
    }

    fn update_logic(&mut self) {
        let now = Instant::now();
        let total_time = now.duration_since(self.start_time).as_secs_f32();
        let dt = now.duration_since(self.last_loop_time).as_secs_f32();
        self.last_loop_time = now;

        let (Some(window), Some(engine), Some(user_state)) =
            (&self.window, &mut self.engine, &mut self.user_state)
        else {
            return;
        };

        let frame_state = FrameState {
            time: total_time,
            dt,
            frame_count: engine.frame_count,
        };

        user_state.update(engine, window, &frame_state);
        engine.update(dt);
    }

    fn render_frame(&mut self) {
        let (Some(engine), Some(user_state)) = (&mut self.engine, &self.user_state) else {
            return;
        };

        let Some(scene_handle) = engine.scene_manager.active_handle() else {
            return;
        };
        let Some(scene) = engine.scene_manager.get_scene_mut(scene_handle) else {
            return;
        };
        let Some(camera_node) = scene.active_camera else {
            return;
        };
        let Some(cam) = scene.cameras.get(camera_node) else {
            return;
        };

        let render_camera = cam.extract_render_camera();
        let extra_nodes = user_state.extra_render_nodes();

        engine
            .renderer
            .render(scene, render_camera, &engine.assets, engine.time, &extra_nodes);
    }
}

impl<H: AppHandler> ApplicationHandler for AppRunner<H> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        let window = Arc::new(window);
        self.window = Some(window.clone());

        log::info!("Initializing Renderer Backend...");

        let mut engine = ThreeEngine::new(self.render_settings.clone());
        let size = window.inner_size();

        if let Err(e) = pollster::block_on(engine.init(window.clone(), size.width, size.height)) {
            log::error!("Fatal Renderer Error: {}", e);
            event_loop.exit();
            return;
        }

        self.user_state = Some(H::init(&mut engine, &window));

        self.engine = Some(engine);

        let now = Instant::now();
        self.start_time = now;
        self.last_loop_time = now;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let (Some(window), Some(engine), Some(user_state)) =
            (&self.window, &mut self.engine, &mut self.user_state)
        else {
            return;
        };

        let consumed = {
            user_state.on_event(engine, window, &event)
        };

        if !consumed {
            // 使用 adapter 将 winit 事件翻译为引擎 Input
            input_adapter::process_window_event(&mut engine.input, &event);

            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(physical_size) => {
                    let scale_factor = window.scale_factor() as f32;
                    engine.resize(physical_size.width, physical_size.height, scale_factor);
                }
                WindowEvent::RedrawRequested => {
                    self.update_logic();
                    self.render_frame();
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
                _ => {}
            }
        } else {
            if let WindowEvent::Resized(ps) = event {
                let scale_factor = window.scale_factor() as f32;
                engine.resize(ps.width, ps.height, scale_factor);
            }
            if let WindowEvent::RedrawRequested = event {
                self.update_logic();
                self.render_frame();
                if let Some(w) = &self.window {
                    w.request_redraw();
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