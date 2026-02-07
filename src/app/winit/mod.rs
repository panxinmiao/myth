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
//!
//! # Usage
//!
//! 1. Implement [`AppHandler`] for your application struct
//! 2. Use [`App`] builder to configure window settings
//! 3. Call [`App::run`] to start the event loop
//!
//! # Example
//!
//! ```rust,ignore
//! use myth::app::winit::{App, AppHandler};
//! use myth::engine::{Engine, FrameState};
//! use std::sync::Arc;
//! use winit::window::Window;
//!
//! struct GameApp {
//!     // Your game state here
//! }
//!
//! impl AppHandler for GameApp {
//!     fn init(engine: &mut Engine, window: &Arc<Window>) -> Self {
//!         // Initialize scene, load assets, etc.
//!         GameApp {}
//!     }
//!
//!     fn update(&mut self, engine: &mut Engine, window: &Arc<Window>, frame: &FrameState) {
//!         // Update game logic
//!     }
//!
//!     fn compose_frame<'a>(&'a self, composer: FrameComposer<'a>) {
//!         // Add custom render passes if needed
//!         composer.render();
//!     }
//! }
//!
//! fn main() -> myth::errors::Result<()> {
//!     App::new()
//!         .with_title("My Game")
//!         .run::<GameApp>()
//! }
//! ```

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
pub use winit::window::{Window, WindowId};

use crate::engine::{Engine, FrameState};
use crate::renderer::graph::FrameComposer;
use crate::renderer::settings::RenderSettings;

pub mod input_adapter;

/// Trait for defining application behavior.
///
/// Implement this trait to create your application. The framework will call
/// these methods at appropriate times during the application lifecycle.
///
/// # Lifecycle
///
/// 1. [`init`](Self::init) - Called once when the window is created
/// 2. [`on_event`](Self::on_event) - Called for each window event
/// 3. [`update`](Self::update) - Called each frame before rendering
/// 4. [`compose_frame`](Self::compose_frame) - Called to configure the render pipeline
///
/// # Example
///
/// ```rust,ignore
/// impl AppHandler for MyApp {
///     fn init(engine: &mut Engine, window: &Arc<Window>) -> Self {
///         // Load assets, create scene
///         MyApp { /* ... */ }
///     }
///
///     fn update(&mut self, engine: &mut Engine, window: &Arc<Window>, frame: &FrameState) {
///         // Update animations, physics, etc.
///     }
/// }
/// ```
pub trait AppHandler: Sized + 'static {
    /// Initializes the application.
    ///
    /// Called once after the window is created and the renderer is initialized.
    /// Use this to set up your scene, load assets, and prepare the initial state.
    ///
    /// # Arguments
    ///
    /// * `engine` - Mutable reference to the engine instance
    /// * `window` - Reference to the window (for querying size, etc.)
    fn init(engine: &mut Engine, window: &Arc<Window>) -> Self;

    /// Handles window events.
    ///
    /// Called for each window event before the engine processes it.
    /// Return `true` to consume the event (prevent default handling),
    /// or `false` to allow normal processing.
    ///
    /// # Arguments
    ///
    /// * `engine` - Mutable reference to the engine
    /// * `window` - Reference to the window
    /// * `event` - The window event to handle
    ///
    /// # Returns
    ///
    /// `true` if the event was consumed, `false` otherwise.
    #[allow(unused_variables)]
    fn on_event(&mut self, engine: &mut Engine, window: &Arc<Window>, event: &WindowEvent) -> bool {
        false
    }

    /// Updates application state.
    ///
    /// Called once per frame before rendering. Use this for game logic,
    /// animations, physics updates, etc.
    ///
    /// # Arguments
    ///
    /// * `engine` - Mutable reference to the engine
    /// * `window` - Reference to the window
    /// * `frame` - Frame timing information
    #[allow(unused_variables)]
    fn update(&mut self, engine: &mut Engine, window: &Arc<Window>, frame: &FrameState) {}

    /// Configures the render pipeline for this frame.
    ///
    /// Override this method to add custom render passes (UI, post-processing, etc.).
    /// The default implementation only renders the built-in forward pass.
    ///
    /// # Arguments
    ///
    /// * `composer` - The frame composer for adding render nodes
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// fn compose_frame<'a>(&'a mut self, composer: FrameComposer<'a>) {
    ///     composer
    ///         .add_node(RenderStage::UI, &mut self.ui_pass)
    ///         .add_node(RenderStage::PostProcess, &mut self.bloom_pass)
    ///         .render();
    /// }
    /// ```
    fn compose_frame<'a>(&'a mut self, composer: FrameComposer<'a>) {
        composer.render();
    }
}

/// A minimal no-op handler for testing or as a template.
///
/// This handler does nothing but can be used to verify that
/// the engine initializes and runs correctly.
pub struct DefaultHandler;

impl AppHandler for DefaultHandler {
    fn init(_ctx: &mut Engine, _window: &Arc<Window>) -> Self {
        Self
    }
    fn update(&mut self, _engine: &mut Engine, _window: &Arc<Window>, _frame: &FrameState) {}
}

/// Application builder for configuring and launching the engine.
///
/// Use the builder pattern to configure window settings, then call
/// [`run`](Self::run) to start the application.
///
/// # Example
///
/// ```rust,ignore
/// App::new()
///     .with_title("My 3D Application")
///     .with_settings(RenderSettings {
///         vsync: true,
///         ..Default::default()
///     })
///     .run::<MyHandler>()?;
/// ```
pub struct App {
    title: String,
    render_settings: RenderSettings,
    #[cfg(target_arch = "wasm32")]
    canvas_id: Option<String>,
}

impl App {
    /// Creates a new application builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            title: "Myth Engine".into(),
            render_settings: RenderSettings::default(),
            #[cfg(target_arch = "wasm32")]
            canvas_id: None,
        }
    }

    /// Sets the window title.
    ///
    /// # Arguments
    ///
    /// * `title` - The window title to display
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Sets the render settings.
    ///
    /// # Arguments
    ///
    /// * `settings` - Custom render configuration
    #[must_use]
    pub fn with_settings(mut self, settings: RenderSettings) -> Self {
        self.render_settings = settings;
        self
    }

    #[cfg(target_arch = "wasm32")]
    /// Sets the HTML canvas element ID to use for rendering (WASM only).
    ///
    /// # Arguments
    ///
    /// * `id` - The ID of the canvas element in the DOM
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
    pub fn run<H: AppHandler>(self) -> crate::errors::Result<()> {
        use crate::Error;

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut runner = AppRunner::<H>::new(self.title, self.render_settings);
        event_loop.run_app(&mut runner).map_err(Error::from)
    }

    /// Runs the application with the specified handler (WASM version).
    ///
    /// On WASM, this spawns an async task and returns immediately.
    /// The event loop runs via requestAnimationFrame.
    #[cfg(target_arch = "wasm32")]
    pub fn run<H: AppHandler>(self) -> crate::errors::Result<()> {
        use winit::platform::web::EventLoopExtWebSys;

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let runner = AppRunner::<H>::new(self.title, self.render_settings, self.canvas_id);
        event_loop.spawn_app(runner);

        Ok(())
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal application runner that implements winit's `ApplicationHandler`.
///
/// This struct manages the application lifecycle including window creation,
/// event handling, and frame rendering.
struct AppRunner<H: AppHandler> {
    title: String,
    render_settings: RenderSettings,

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
    /// Try to take the result if available, returns None if not ready or already taken
    fn try_take_result(&mut self) -> Option<(Engine, H)> {
        self.result.take()
    }

    // Check if initialization is complete (result is ready)
    // fn is_complete(&self) -> bool {
    //     self.result.is_some()
    // }
}

impl<H: AppHandler> AppRunner<H> {
    fn new(
        title: String,
        render_settings: RenderSettings,
        #[cfg(target_arch = "wasm32")] canvas_id: Option<String>,
    ) -> Self {
        let now = Instant::now();
        Self {
            title,
            render_settings,
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
        let (Some(engine), Some(user_state)) = (&mut self.engine, &mut self.user_state) else {
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

        // Use new chained FrameComposer API
        if let Some(composer) =
            engine
                .renderer
                .begin_frame(scene, &render_camera, &engine.assets, engine.time)
        {
            // User adds nodes via compose_frame chaining
            user_state.compose_frame(composer);
        }

        // Periodically clean up resources
        engine.renderer.maybe_prune();
    }
}

impl<H: AppHandler> ApplicationHandler for AppRunner<H> {
    #[cfg(not(target_arch = "wasm32"))]
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

        let mut engine = Engine::new(self.render_settings.clone());
        let size = window.inner_size();

        if let Err(e) = pollster::block_on(engine.init(window.clone(), size.width, size.height)) {
            log::error!("Fatal Renderer Error: {e}");
            event_loop.exit();
            return;
        }

        self.user_state = Some(H::init(&mut engine, &window));

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

        // Get canvas from DOM
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

        // Get canvas size
        let window = web_sys::window().unwrap();
        let dpr = window.device_pixel_ratio();
        let width = (canvas.client_width() as f64 * dpr) as u32;
        let height = (canvas.client_height() as f64 * dpr) as u32;
        canvas.set_width(width);
        canvas.set_height(height);

        let window_attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_canvas(Some(canvas.clone()));

        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        let window = Arc::new(window);
        self.window = Some(window.clone());

        log::info!("Initializing WebGPU Renderer Backend...");

        // On WASM, we must use true async initialization because requestAdapter is async
        let render_settings = self.render_settings.clone();
        let init_state = self.init_state.clone();
        let window_clone = window.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let mut engine = Engine::new(render_settings);
            let size = window_clone.inner_size();
            let w = size.width.max(1);
            let h = size.height.max(1);

            match engine.init(window_clone.clone(), w, h).await {
                Ok(_) => {
                    log::info!("WebGPU initialization successful");
                    let user_state = H::init(&mut engine, &window_clone);
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
        // On WASM, check if async initialization has completed
        #[cfg(target_arch = "wasm32")]
        {
            if self.engine.is_none() {
                // Try to get the result without panicking if already borrowed
                let result = {
                    match self.init_state.try_borrow_mut() {
                        Ok(mut state) => state.try_take_result(),
                        Err(_) => {
                            return;
                        }
                    }
                };

                if let Some((mut engine, user_state)) = result {
                    // Immediately resize to correct dimensions after init completes
                    if let Some(window) = &self.window {
                        let size = window.inner_size();
                        let scale_factor = window.scale_factor() as f32;
                        let w = size.width.max(1);
                        let h = size.height.max(1);
                        engine.resize(w, h, scale_factor);
                        log::info!("Resized to {}x{} after init", w, h);
                    }

                    self.engine = Some(engine);
                    self.user_state = Some(user_state);
                    log::info!("Engine initialization completed, starting render loop");
                } else {
                    // Still initializing
                    return;
                }
            }
        }

        let (Some(window), Some(engine), Some(user_state)) =
            (&self.window, &mut self.engine, &mut self.user_state)
        else {
            return;
        };

        let consumed = { user_state.on_event(engine, window, &event) };

        if consumed {
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
        } else {
            // Use adapter to translate winit events to engine Input
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
