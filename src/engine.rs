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
//! use myth::{Engine, RendererSettings};
//!
//! // Create engine with custom settings
//! let mut engine = Engine::new(RendererSettings::default());
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

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::assets::AssetServer;
use crate::renderer::Renderer;
use crate::renderer::settings::RendererSettings;
use crate::resources::input::Input;
use crate::scene::manager::SceneManager;

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

    time: f32,
    frame_count: u64,
}

impl Engine {
    /// Creates a new engine instance with the specified render settings.
    ///
    /// This only creates the engine configuration. GPU resources are not
    /// allocated until [`init`](Self::init) is called.
    ///
    /// # Arguments
    ///
    /// * `settings` - Render configuration including power preference, features, etc.
    #[must_use]
    pub fn new(settings: RendererSettings) -> Self {
        let assets = AssetServer::new();
        Self {
            renderer: Renderer::new(settings),
            scene_manager: SceneManager::new(assets.clone()),
            assets,
            input: Input::new(),
            time: 0.0,
            frame_count: 0,
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
    pub async fn init<W>(&mut self, window: W, width: u32, height: u32) -> crate::errors::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        self.renderer.init(window, width, height).await?;

        Ok(())
    }

    /// Returns the total elapsed time in seconds since the engine started.
    #[inline]
    #[must_use]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Returns the total number of frames rendered since startup.
    #[inline]
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
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
    /// * `scale_factor` - Display scale factor (for `HiDPI` support)
    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {
        self.renderer.resize(width, height, scale_factor);
        self.input.inject_resize(width, height);

        if width > 0 && height > 0 {
            self.update_camera_aspect(width as f32 / height as f32);
        }
    }

    /// Updates the engine state for the current frame.
    ///
    /// This method should be called once per frame before rendering. It:
    /// - Updates the total elapsed time and frame counter
    /// - Runs scene logic and animations
    /// - Resets per-frame input state
    ///
    /// # Arguments
    ///
    /// * `dt` - Delta time since the last frame in seconds
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.frame_count += 1;

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

    fn update_camera_aspect(&mut self, aspect: f32) {
        let Some(scene) = self.scene_manager.active_scene_mut() else {
            return;
        };
        let Some(cam_handle) = scene.active_camera else {
            return;
        };
        if let Some(cam) = scene.cameras.get_mut(cam_handle) {
            cam.set_aspect(aspect);
        }
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
        let Some(scene_handle) = self.scene_manager.active_handle() else {
            return false;
        };
        let time = self.time;
        let Some(scene) = self.scene_manager.get_scene_mut(scene_handle) else {
            return false;
        };
        let Some(camera_node) = scene.active_camera else {
            return false;
        };
        let Some(cam) = scene.cameras.get(camera_node) else {
            return false;
        };
        let render_camera = cam.extract_render_camera();
        if let Some(composer) = self
            .renderer
            .begin_frame(scene, &render_camera, &self.assets, time)
        {
            composer.render();
            true
        } else {
            false
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(RendererSettings::default())
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
