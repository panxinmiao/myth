//! Winit-based Application Framework
//!
//! This module provides a complete application framework built on top of the
//! [winit](https://crates.io/crates/winit) cross-platform windowing library.
//!
//! # Overview
//!
//! The framework consists of:
//!
//! - [`App`]: Builder for configuring and launching applications
//! - [`AppHandler`]: Trait that users implement to define application behavior
//! - [`AppRunner`]: Internal event loop handler (not exposed publicly)

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use glam::Vec2;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
pub use winit::window::Window as WinitWindow;
use winit::window::{Window, WindowId};

use crate::app::AppHandler;
use crate::engine::{Engine, FrameState};
use crate::window::Window as WindowTrait;
use myth_core::{Error, PlatformError};
use myth_render::settings::{RendererInitConfig, RendererSettings};

pub mod input_adapter;

// ============================================================================
// Device Detection (WASM / Native)
// ============================================================================

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(
    inline_js = "export function is_mobile_device() { return /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent); }"
)]
extern "C" {
    pub fn is_mobile_device() -> bool;
}

// ============================================================================
// Window Trait Implementation for winit::Window
// ============================================================================

impl WindowTrait for Window {
    fn set_title(&self, title: &str) {
        Window::set_title(self, title);
    }

    fn inner_size(&self) -> Vec2 {
        let size = Window::inner_size(self);
        Vec2::new(size.width as f32, size.height as f32)
    }

    fn scale_factor(&self) -> f32 {
        Window::scale_factor(self) as f32
    }

    fn request_redraw(&self) {
        Window::request_redraw(self);
    }

    fn set_cursor_visible(&self, visible: bool) {
        Window::set_cursor_visible(self, visible);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// App Builder
// ============================================================================

/// Application builder for configuring and launching the engine.
pub struct App {
    title: String,
    init_config: RendererInitConfig,
    render_settings: RendererSettings,
    #[cfg(not(target_arch = "wasm32"))]
    window_size: Option<(u32, u32)>,
    #[cfg(target_arch = "wasm32")]
    canvas_id: Option<String>,
}

impl App {
    /// Creates a new application builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            title: "Myth Engine".into(),
            init_config: RendererInitConfig::default(),
            render_settings: RendererSettings::default(),
            #[cfg(not(target_arch = "wasm32"))]
            window_size: None,
            #[cfg(target_arch = "wasm32")]
            canvas_id: None,
        }
    }

    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Sets the static GPU initialization configuration.
    #[must_use]
    pub fn with_init_config(mut self, config: RendererInitConfig) -> Self {
        self.init_config = config;
        self
    }

    /// Sets the runtime rendering settings.
    #[must_use]
    pub fn with_settings(mut self, settings: RendererSettings) -> Self {
        self.render_settings = settings;
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Sets the initial logical size of the window (only effective for native, WASM is controlled by CSS).
    #[must_use]
    pub fn with_inner_size(mut self, width: u32, height: u32) -> Self {
        self.window_size = Some((width, height));
        self
    }

    #[cfg(target_arch = "wasm32")]
    /// Sets the HTML canvas element ID to use for rendering (WASM only).
    #[must_use]
    pub fn with_canvas_id(mut self, id: impl Into<String>) -> Self {
        self.canvas_id = Some(id.into());
        self
    }

    /// Runs the application with the specified handler.
    ///
    /// This method blocks until the application exits. The event loop
    /// takes ownership of the current thread.
    ///
    /// # Type Parameters
    ///
    /// * `H` - The application handler type implementing [`AppHandler`]
    ///
    /// # Errors
    ///
    /// Returns an error if event loop creation or execution fails.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run<H: AppHandler>(self) -> myth_core::Result<()> {
        let event_loop = EventLoop::new()
            .map_err(|e| Error::Platform(PlatformError::EventLoop(e.to_string())))?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut runner = AppRunner::<H>::new(
            self.title,
            self.init_config,
            self.render_settings,
            self.window_size,
        );
        event_loop
            .run_app(&mut runner)
            .map_err(|e| Error::Platform(PlatformError::EventLoop(e.to_string())))
    }

    /// Runs the application with the specified handler (WASM version).
    ///
    /// On WASM, this spawns an async task and returns immediately.
    /// The event loop runs via requestAnimationFrame.
    #[cfg(target_arch = "wasm32")]
    pub fn run<H: AppHandler>(self) -> myth_core::Result<()> {
        use winit::platform::web::EventLoopExtWebSys;

        let event_loop = EventLoop::new()
            .map_err(|e| Error::Platform(PlatformError::EventLoop(e.to_string())))?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let runner = AppRunner::<H>::new(
            self.title,
            self.init_config,
            self.render_settings,
            self.canvas_id,
        );
        event_loop.spawn_app(runner);

        Ok(())
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Internal AppRunner
// ============================================================================

/// Internal application runner that implements winit's `ApplicationHandler`.
struct AppRunner<H: AppHandler> {
    title: String,
    init_config: RendererInitConfig,
    render_settings: RendererSettings,

    #[cfg(not(target_arch = "wasm32"))]
    window_size: Option<(u32, u32)>,

    #[cfg(target_arch = "wasm32")]
    canvas_id: Option<String>,

    window: Option<Arc<Window>>,
    engine: Option<Engine>,
    user_state: Option<H>,

    start_time: Instant,
    last_loop_time: Instant,

    /// WASM async initialization state
    #[cfg(target_arch = "wasm32")]
    init_state: std::rc::Rc<std::cell::RefCell<WasmInitState<H>>>,
}

/// State for WASM async initialization
#[cfg(target_arch = "wasm32")]
struct WasmInitState<H: AppHandler> {
    pending: bool,
    result: Option<(Engine, H)>,
}

#[cfg(target_arch = "wasm32")]
impl<H: AppHandler> Default for WasmInitState<H> {
    fn default() -> Self {
        Self {
            pending: false,
            result: None,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl<H: AppHandler> WasmInitState<H> {
    fn try_take_result(&mut self) -> Option<(Engine, H)> {
        self.result.take()
    }
}

impl<H: AppHandler> AppRunner<H> {
    fn new(
        title: String,
        init_config: RendererInitConfig,
        render_settings: RendererSettings,
        #[cfg(not(target_arch = "wasm32"))] window_size: Option<(u32, u32)>,
        #[cfg(target_arch = "wasm32")] canvas_id: Option<String>,
    ) -> Self {
        let now = Instant::now();
        Self {
            title,
            init_config,
            render_settings,
            #[cfg(not(target_arch = "wasm32"))]
            window_size,
            #[cfg(target_arch = "wasm32")]
            canvas_id,

            window: None,
            engine: None,
            user_state: None,
            start_time: now,
            last_loop_time: now,
            #[cfg(target_arch = "wasm32")]
            init_state: std::rc::Rc::new(std::cell::RefCell::new(WasmInitState::default())),
        }
    }

    fn update_logic(&mut self) {
        let now = Instant::now();
        let total_time = now.duration_since(self.start_time).as_secs_f32();

        // Limiting max dt to 100ms to prevent death spiral from window dragging or tab switching
        let raw_dt = now.duration_since(self.last_loop_time).as_secs_f32();
        let dt = raw_dt.min(0.1);

        self.last_loop_time = now;

        let (Some(window), Some(engine), Some(user_state)) =
            (&self.window, &mut self.engine, &mut self.user_state)
        else {
            return;
        };

        let frame_state = FrameState {
            time: total_time,
            dt,
            frame_count: engine.frame_count(),
        };

        // Pass &dyn WindowTrait (winit::Window implements our Window trait)
        user_state.update(engine, window.as_ref(), &frame_state);
        engine.update(dt);
    }

    fn render_frame(&mut self) {
        let (Some(engine), Some(user_state)) = (&mut self.engine, &mut self.user_state) else {
            return;
        };

        let Some(scene_handle) = engine.scene_manager.active_handle() else {
            return;
        };
        let time = engine.frame_time();
        let Some(scene) = engine.scene_manager.get_scene_mut(scene_handle) else {
            return;
        };
        let Some(camera_node) = scene.active_camera else {
            return;
        };

        // Sync the camera's AA mode to the renderer and advance TAA jitter.
        if let Some(cam) = scene.cameras.get_mut(camera_node)
            && engine.renderer.render_path().supports_post_processing()
        {
            cam.step_frame();
        }

        let Some(cam) = scene.cameras.get(camera_node) else {
            return;
        };
        let render_camera = cam.extract_render_camera();

        if let Some(composer) =
            engine
                .renderer
                .begin_frame(scene, &render_camera, &engine.assets, time)
        {
            user_state.compose_frame(composer);
        }

        engine.renderer.maybe_prune();
    }
}

impl<H: AppHandler> ApplicationHandler for AppRunner<H> {
    #[cfg(not(target_arch = "wasm32"))]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let (width, height) = self.window_size.unwrap_or((1280, 720));
        let window_attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height));

        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        let window = Arc::new(window);
        self.window = Some(window.clone());

        log::info!("Initializing Renderer Backend...");

        let mut engine = Engine::new(self.init_config.clone(), self.render_settings.clone());
        let size = window.inner_size();

        if let Err(e) = pollster::block_on(engine.init(window.clone(), size.width, size.height)) {
            log::error!("Fatal Renderer Error: {e}");
            event_loop.exit();
            return;
        }

        self.user_state = Some(H::init(&mut engine, window.as_ref()));
        self.engine = Some(engine);

        let now = Instant::now();
        self.start_time = now;
        self.last_loop_time = now;
    }

    #[cfg(target_arch = "wasm32")]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowAttributesExtWebSys;

        if self.window.is_some() {
            return;
        }

        let web_window = web_sys::window().expect("No window found");
        let document = web_window.document().expect("No document found");

        let canvas_id = self.canvas_id.as_deref().unwrap_or("myth-canvas");

        let canvas = document
            .get_element_by_id(canvas_id)
            .expect(&format!("Canvas element '{}' not found", canvas_id))
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("Element is not a canvas");

        canvas.set_attribute("tabindex", "0").ok();
        canvas.focus().ok();

        let mut dpr = web_window.device_pixel_ratio();
        if is_mobile_device() {
            dpr = dpr.min(1.5); // Limit the maximum device pixel ratio on mobile devices
            log::info!(
                "📱 Mobile device detected. Clamping initial DPR to {:.2}",
                dpr
            );
        }

        let width = (canvas.client_width() as f64 * dpr) as u32;
        let height = (canvas.client_height() as f64 * dpr) as u32;
        canvas.set_width(width.max(1));
        canvas.set_height(height.max(1));

        let window_attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_canvas(Some(canvas.clone()));

        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        let window = Arc::new(window);
        self.window = Some(window.clone());

        log::info!("Initializing WebGPU Renderer Backend...");

        let render_settings = self.render_settings.clone();
        let init_config = self.init_config.clone();
        let init_state = self.init_state.clone();
        let window_clone = window.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let mut engine = Engine::new(init_config, render_settings);
            let size = window_clone.inner_size();
            let w = size.width.max(1);
            let h = size.height.max(1);

            match engine.init(window_clone.clone(), w, h).await {
                Ok(_) => {
                    log::info!("WebGPU initialization successful");
                    let user_state = H::init(&mut engine, window_clone.as_ref());
                    init_state.borrow_mut().result = Some((engine, user_state));

                    window_clone.request_redraw();
                }
                Err(e) => {
                    log::error!("Fatal Renderer Error: {}", e);
                    panic!("Failed to initialize engine: {}", e);
                }
            }
        });

        self.init_state.borrow_mut().pending = true;

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
        #[cfg(target_arch = "wasm32")]
        {
            if self.engine.is_none() {
                let result = {
                    match self.init_state.try_borrow_mut() {
                        Ok(mut state) => state.try_take_result(),
                        Err(_) => return,
                    }
                };

                if let Some((mut engine, user_state)) = result {
                    if let Some(window) = &self.window {
                        let size = window.inner_size();
                        let w = size.width.max(1);
                        let h = size.height.max(1);
                        engine.resize(w, h);
                        log::trace!("Resized to {}x{} after init", w, h);
                    }

                    self.engine = Some(engine);
                    self.user_state = Some(user_state);
                    log::trace!("Engine initialization completed, starting render loop");
                } else {
                    return;
                }
            }
        }

        let (Some(window), Some(engine), Some(user_state)) =
            (&self.window, &mut self.engine, &mut self.user_state)
        else {
            return;
        };

        // Pass raw event to user via &dyn Any (platform-independent signature)
        let consumed = user_state.on_event(engine, window.as_ref(), &event);

        if !consumed {
            input_adapter::process_window_event(&mut engine.input, &event);
        }

        // If the event was a resize, we need to update the renderer immediately to avoid rendering issues.
        // We also want to update the input system's screen size so that input coordinates remain correct.
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(physical_size) => {
                // Handle DPR changes on browser resize to prevent rendering issues.
                // On native platforms, the window size is already in physical pixels, so we can use it directly.
                // On WASM, we need to recalculate the DPR and adjust the canvas size accordingly.
                #[cfg(target_arch = "wasm32")]
                let (w, h) = {
                    let scale_factor = window.scale_factor();

                    let logical_w = physical_size.width as f64 / scale_factor;
                    let logical_h = physical_size.height as f64 / scale_factor;

                    let mut target_dpr = scale_factor;
                    if is_mobile_device() {
                        target_dpr = target_dpr.min(1.5);
                    }

                    let new_w = (logical_w * target_dpr).round() as u32;
                    let new_h = (logical_h * target_dpr).round() as u32;

                    // Update the HTML canvas element's width and height attributes to match the new physical size. This is crucial for WebGPU to recognize the correct surface dimensions and avoid rendering issues.
                    use wasm_bindgen::JsCast;
                    if let Some(canvas) = web_sys::window()
                        .and_then(|win| win.document())
                        .and_then(|doc| doc.query_selector("canvas").ok().flatten())
                        .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                    {
                        canvas.set_width(new_w.max(1));
                        canvas.set_height(new_h.max(1));
                    }

                    (new_w, new_h)
                };

                #[cfg(not(target_arch = "wasm32"))]
                let (w, h) = (physical_size.width, physical_size.height);

                engine.resize(w, h);
            }

            WindowEvent::RedrawRequested => {
                self.update_logic();
                self.render_frame();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.engine.is_some()
            && let Some(window) = &self.window
        {
            window.request_redraw();
        }
    }
}
