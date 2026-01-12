
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::assets::AssetServer;
use crate::scene::Scene;
use crate::render::{Renderer, settings::RendererSettings};
use crate::scene::NodeIndex;

pub type UpdateFn = Box<dyn FnMut(&mut Scene, &AssetServer, f32)>;

pub struct App {
    window: Option<Arc<Window>>,
    pub renderer: Renderer, // 始终存在

    title: String,
    pub assets: AssetServer,
    pub scene: Scene,
    pub active_camera: Option<NodeIndex>,
    update_fn: Option<UpdateFn>,
    start_time: Option<std::time::Instant>,
    last_frame_time: f32,
}

impl App {
    pub fn new() -> Self {
        let assets = AssetServer::new();
        let scene = Scene::new();
        // 创建默认配置的 Renderer
        let renderer = Renderer::new(RendererSettings::default());

        Self {
            window: None,
            renderer,
            title: "Three-rs Engine".to_string(),
            assets,
            scene,
            active_camera: None,
            update_fn: None,
            start_time: None,
            last_frame_time: 0.0,
        }
    }

    /// 允许用户替换 Renderer (配置阶段)
    pub fn with_renderer(mut self, renderer: Renderer) -> Self {
        self.renderer = renderer;
        self
    }

    pub fn set_update_fn<F>(&mut self, f: F) -> &mut Self 
    where
        F: FnMut(&mut Scene, &AssetServer, f32) + 'static,
    {
        self.update_fn = Some(Box::new(f));
        self
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(&mut self)?;
        Ok(())
    }

    fn update(&mut self) {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }
        if let Some(start) = self.start_time {
            self.last_frame_time = start.elapsed().as_secs_f32();
        }
        if let Some(ref mut update_fn) = self.update_fn {
            update_fn(&mut self.scene, &self.assets, self.last_frame_time);
        }
        self.scene.update();
    }

    fn render(&mut self) {
        if self.window.is_some() {
            if let Some(cam_id) = self.active_camera {
                let scene_ref = &self.scene;
                if let Some(node) = scene_ref.get_node(cam_id) {
                    if let Some(camera_idx) = node.camera {
                        if let Some(camera) = self.scene.cameras.get(camera_idx) {
                            self.renderer.render(&self.scene, camera, &self.assets);
                        }
                    }
                }
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

        let window = event_loop.create_window(window_attributes).unwrap();
        let window = Arc::new(window);
        self.window = Some(window.clone());

        // 延迟初始化 Renderer
        log::info!("Initializing Renderer Backend...");
        pollster::block_on(self.renderer.init(window.clone()));
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
                self.renderer.resize(physical_size.width, physical_size.height);

                if physical_size.height > 0 {
                    let new_aspect = physical_size.width as f32 / physical_size.height as f32;
                    let camera_idx = self.active_camera
                        .and_then(|node_id| self.scene.get_node(node_id))
                        .and_then(|node| node.camera);

                    if let Some(idx) = camera_idx {
                        if let Some(camera) = self.scene.cameras.get_mut(idx) {
                            camera.aspect = new_aspect;
                            camera.update_projection_matrix();
                        }
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