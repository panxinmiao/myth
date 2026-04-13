//! Engine Core Module
//!
//! This module contains [`Engine`], the central coordinator of the rendering engine.
//! It is a pure engine instance without any window management logic, allowing it to be
//! driven by different frontends (Winit, Python bindings, WebAssembly, etc.).
//!
//! # Architecture
//!
//! The engine follows a clean separation of concerns:
//!
//! - **Renderer**: Handles all GPU operations and rendering pipeline
//! - **`SceneManager`**: Manages multiple scenes and their lifecycles
//! - **`AssetServer`**: Centralized asset storage and loading
//! - **Input**: Unified input state management
//!
//! # Example
//!
//! ```rust,ignore
//! use myth_app::{Engine, RendererInitConfig, RendererSettings};
//!
//! // Create engine with custom settings
//! let mut engine = Engine::new(RendererInitConfig::default(), RendererSettings::default());
//!
//! // Initialize GPU context with a window
//! engine.init(window, 1280, 720).await?;
//!
//! // Main loop
//! loop {
//!     engine.update(dt);
//!     // ... render frame ...
//! }
//! ```

use myth_core::Error;
use myth_render::core::{ReadbackFrame, ReadbackStream};
use myth_render::graph::FrameComposer;
use myth_render::renderer::FrameTime;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use myth_assets::AssetServer;
use myth_assets::manager::SceneManager;
use myth_render::Renderer;
use myth_render::settings::{RendererInitConfig, RendererSettings};
use myth_resources::input::Input;

/// The core engine instance that orchestrates all rendering subsystems.
///
/// `Engine` is a pure engine implementation without window management,
/// making it suitable for integration with various windowing systems and platforms.
///
/// # Components
///
/// - `renderer`: The rendering subsystem handling GPU operations
/// - `scene_manager`: Manages multiple scenes and active scene selection
/// - `assets`: Central asset storage for geometries, materials, textures, etc.
/// - `input`: Unified input state (keyboard, mouse, touch)
///
/// # Lifecycle
///
/// 1. Create with [`Engine::new`] or [`Engine::default`]
/// 2. Initialize GPU with [`Engine::init`]
/// 3. Update each frame with [`Engine::update`]
/// 4. Render using [`Renderer::begin_frame`]
pub struct Engine {
    pub renderer: Renderer,
    pub scene_manager: SceneManager,
    pub assets: AssetServer,
    pub input: Input,

    frame_time: FrameTime,
}

impl Engine {
    /// Creates a new engine instance with the specified configuration.
    ///
    /// This only creates the engine configuration. GPU resources are not
    /// allocated until [`init`](Self::init) is called.
    ///
    /// # Arguments
    ///
    /// * `init_config` - Static GPU initialization parameters
    /// * `settings` - Runtime rendering settings
    #[must_use]
    pub fn new(init_config: RendererInitConfig, settings: RendererSettings) -> Self {
        let assets = AssetServer::new();
        Self {
            renderer: Renderer::new(init_config, settings),
            scene_manager: SceneManager::new(assets.clone()),
            assets,
            input: Input::new(),
            frame_time: FrameTime::default(),
        }
    }

    /// Initializes GPU resources with the given window.
    ///
    /// This method must be called before any rendering can occur. It accepts
    /// any type that implements the raw window handle traits, making it
    /// compatible with various windowing libraries.
    ///
    /// # Arguments
    ///
    /// * `window` - A window that provides display and window handles
    /// * `width` - Initial surface width in pixels
    /// * `height` - Initial surface height in pixels
    ///
    /// # Errors
    ///
    /// Returns an error if GPU initialization fails due to:
    /// - No compatible GPU adapter found
    /// - Device request failed (unsupported features/limits)
    /// - Surface configuration failed
    pub async fn init<W>(&mut self, window: W, width: u32, height: u32) -> myth_core::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        self.renderer.init(window, width, height).await?;

        Ok(())
    }

    /// Initializes the GPU context in headless (offscreen) mode.
    ///
    /// No window or surface is created. An offscreen render target of the
    /// specified dimensions is allocated instead, suitable for server-side
    /// rendering, automated testing, and GPU readback.
    ///
    /// # Arguments
    ///
    /// * `width` — Render target width in pixels.
    /// * `height` — Render target height in pixels.
    /// * `format` — Desired pixel format. Pass `None` for the default
    ///   `Rgba8Unorm` (sRGB). Use `Some(Rgba16Float)` for HDR readback.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU initialization fails.
    pub async fn init_headless(
        &mut self,
        width: u32,
        height: u32,
        format: Option<myth_resources::PixelFormat>,
    ) -> myth_core::Result<()> {
        self.renderer.init_headless(width, height, format).await?;

        Ok(())
    }

    /// Reads back the current headless render target as raw pixel data.
    ///
    /// The returned `Vec<u8>` contains tightly-packed pixel data whose byte
    /// count per pixel matches the headless texture format. A staging buffer
    /// is cached internally to avoid per-frame allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the renderer is not initialised or not in headless mode.
    pub fn readback_pixels(&mut self) -> myth_core::Result<Vec<u8>> {
        self.renderer.readback_pixels()
    }

    /// Submits the current headless frame to a [`ReadbackStream`] (non-blocking).
    ///
    /// Returns [`ReadbackError::RingFull`] if all ring-buffer slots are
    /// in-flight. The caller may skip the frame or drain with
    /// [`ReadbackStream::try_recv`] first.
    pub fn submit_to_stream(&self, stream: &mut ReadbackStream) -> myth_core::Result<()> {
        let ctx = self
            .renderer
            .wgpu_ctx()
            .ok_or_else(|| Error::General("Engine is not initialized yet.".into()))?;

        let target = ctx.headless_texture.as_ref().ok_or_else(|| {
            Error::General(
                "Cannot submit to stream: Engine is not running in Headless mode.".into(),
            )
        })?;

        stream.try_submit(&ctx.device, &ctx.queue, target)?;

        Ok(())
    }

    /// Submits the current headless frame to a [`ReadbackStream`], blocking
    /// when the ring buffer is full.
    ///
    /// Completed frames are stashed internally and can be retrieved via
    /// [`ReadbackStream::try_recv`] or [`ReadbackStream::try_recv_into`].
    /// `max_stash_size` caps the number of unconsumed stashed frames to
    /// prevent unbounded memory growth.
    pub fn submit_to_stream_blocking(&self, stream: &mut ReadbackStream) -> myth_core::Result<()> {
        let ctx = self
            .renderer
            .wgpu_ctx()
            .ok_or_else(|| Error::General("Engine is not initialized yet.".into()))?;

        let target = ctx.headless_texture.as_ref().ok_or_else(|| {
            Error::General(
                "Cannot submit to stream: Engine is not running in Headless mode.".into(),
            )
        })?;

        stream.submit_blocking(&ctx.device, &ctx.queue, target)?;

        Ok(())
    }

    /// Flushes a `ReadbackStream`, blocking until all in-flight frames are returned.
    pub fn flush_stream(
        &self,
        stream: &mut ReadbackStream,
    ) -> myth_core::Result<Vec<ReadbackFrame>> {
        let ctx = self
            .renderer
            .wgpu_ctx()
            .ok_or_else(|| Error::General("Engine is not initialized yet.".into()))?;

        let frames = stream.flush(&ctx.device)?;

        Ok(frames)
    }

    /// Drives pending GPU callbacks without blocking.
    ///
    /// Call this once per frame in a readback-stream loop so that
    /// `map_async` callbacks fire and frames become available.
    pub fn poll_device(&self) {
        self.renderer.poll_device();
    }

    /// Returns the total elapsed time in seconds since the engine started.
    #[inline]
    #[must_use]
    pub fn time(&self) -> f32 {
        self.frame_time.time
    }

    #[inline]
    #[must_use]
    pub fn frame_time(&self) -> FrameTime {
        self.frame_time
    }

    /// Returns the total number of frames rendered since startup.
    #[inline]
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_time.frame_count
    }

    /// Returns the current surface/window size in pixels as `(width, height)`.
    #[inline]
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        self.renderer.size()
    }

    /// Handles window resize events.
    ///
    /// This method should be called whenever the window size changes.
    /// It updates the renderer's surface configuration and camera aspect ratios.
    ///
    /// # Arguments
    ///
    /// * `width` - New width in pixels
    /// * `height` - New height in pixels
    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
        self.input.inject_resize(width, height);

        if width > 0 && height > 0 {
            self.update_camera_viewport(width as f32, height as f32);
        }
    }

    /// Updates the engine state for the current frame.
    ///
    /// This method should be called once per frame before rendering. It:
    /// - Processes completed background asset loads
    /// - Updates the total elapsed time and frame counter
    /// - Runs scene logic and animations
    /// - Resets per-frame input state
    ///
    /// # Arguments
    ///
    /// * `dt` - Delta time since the last frame in seconds
    pub fn update(&mut self, dt: f32) {
        // Promote any assets that finished loading in the background.
        self.assets.process_loading_events();

        self.frame_time.time += dt;
        self.frame_time.frame_count += 1;
        self.frame_time.delta_time = dt;

        if let Some(scene) = self.scene_manager.active_scene_mut() {
            scene.update(&self.input, dt);
        }

        self.input.start_frame();
    }

    /// Performs periodic resource cleanup.
    ///
    /// This method should be called after each frame to release unused GPU
    /// resources and prevent memory leaks. It uses internal heuristics to
    /// avoid expensive cleanup operations on every frame.
    #[inline]
    pub fn maybe_prune(&mut self) {
        self.renderer.maybe_prune();
    }

    fn update_camera_viewport(&mut self, width: f32, height: f32) {
        let Some(scene) = self.scene_manager.active_scene_mut() else {
            return;
        };
        let Some(cam_handle) = scene.active_camera else {
            return;
        };
        if let Some(cam) = scene.cameras.get_mut(cam_handle) {
            cam.set_viewport_size(width, height);
        }
    }

    /// Syncs the active camera's AA state with the renderer and advances the
    /// temporal jitter sequence.
    fn step_camera_frame(&mut self) {
        let Some(scene) = self.scene_manager.active_scene_mut() else {
            return;
        };
        let Some(cam_handle) = scene.active_camera else {
            return;
        };
        if let Some(cam) = scene.cameras.get_mut(cam_handle)
            && self.renderer.render_path().supports_post_processing()
        {
            cam.step_frame();
        }
    }

    /// Prepares a new frame for rendering by extracting scene data and configuring the renderer.
    ///
    /// Returns a `FrameComposer` if a frame could be prepared, or `None` if no active scene or camera is available.
    pub fn compose_frame(&mut self) -> Option<FrameComposer<'_>> {
        self.step_camera_frame();

        let scene_handle = self.scene_manager.active_handle()?;
        let scene = self.scene_manager.get_scene_mut(scene_handle)?;
        let camera_node = scene.active_camera?;
        let cam = scene.cameras.get(camera_node)?;
        let render_camera = cam.extract_render_camera();

        self.renderer
            .begin_frame(scene, render_camera, &self.assets, self.frame_time)
    }

    /// Renders the active scene using the active camera.
    ///
    /// This is a convenience method that combines scene lookup, camera extraction,
    /// and frame rendering into a single call. It avoids the split-borrow issues
    /// that arise when accessing the renderer and scene manager separately.
    ///
    /// Returns `true` if a frame was successfully rendered, `false` if rendering
    /// was skipped (no active scene, no active camera, etc.).
    pub fn render_active_scene(&mut self) -> bool {
        if let Some(composer) = self.compose_frame() {
            composer.render();
            true
        } else {
            false
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(RendererInitConfig::default(), RendererSettings::default())
    }
}

/// Per-frame timing and state information.
///
/// This struct is passed to user update callbacks each frame,
/// providing essential timing information for animations and logic.
#[derive(Debug, Clone, Copy)]
pub struct FrameState {
    /// Total elapsed time since the application started (in seconds).
    pub time: f32,
    /// Delta time since the last frame (in seconds).
    pub dt: f32,
    /// Total number of frames rendered since startup.
    pub frame_count: u64,
}
