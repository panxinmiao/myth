//! Application Framework Module
//!
//! This module provides the application lifecycle management and windowing integration.
//! It bridges the engine core with platform-specific window systems.
//!
//! # Architecture
//!
//! The app module follows a trait-based design:
//!
//! - [`Window`]: Platform-independent window abstraction
//! - [`AppHandler`]: User-implemented trait for application behavior
//!
//! The framework handles window creation, event loop processing, input translation,
//! and frame timing. Users only need to implement [`AppHandler`].
//!
//! # Platform Support
//!
//! Currently supports:
//! - **Winit** (default): Cross-platform windowing via the winit crate
//!
//! # Example
//!
//! ```rust,ignore
//! use myth::app::{AppHandler, Window};
//! use myth::engine::{Engine, FrameState};
//!
//! struct MyGame;
//!
//! impl AppHandler for MyGame {
//!     fn init(engine: &mut Engine, window: &dyn Window) -> Self {
//!         window.set_title("My Game");
//!         MyGame
//!     }
//!
//!     fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
//!         // Use engine.input for input handling
//!         if engine.input.get_key_down(myth::resources::input::Key::Space) {
//!             println!("Jump!");
//!         }
//!     }
//! }
//! ```

pub mod window;

#[cfg(feature = "winit")]
pub mod winit;

pub use window::Window;

use crate::engine::{Engine, FrameState};
use crate::renderer::graph::FrameComposer;

/// Trait for defining application behavior.
///
/// Implement this trait to create your application. The framework will call
/// these methods at appropriate times during the application lifecycle.
///
/// # Lifecycle
///
/// 1. [`init`](Self::init) - Called once when the window and renderer are ready
/// 2. [`update`](Self::update) - Called each frame before rendering
/// 3. [`compose_frame`](Self::compose_frame) - Called to configure the render pipeline
///
/// # Input Handling
///
/// Use `engine.input` to query input state in [`update`](Self::update).
/// This is the standard game development paradigm and works across all backends.
///
/// For advanced use cases that require raw platform events (e.g., integrating
/// an external UI framework like egui), use [`on_event`](Self::on_event).
/// The concrete event type depends on the backend; with winit it is
/// `winit::event::WindowEvent`. Downcast via `event.downcast_ref::<T>()`.
///
/// # Example
///
/// ```rust,ignore
/// use myth::app::{AppHandler, Window};
///
/// impl AppHandler for MyApp {
///     fn init(engine: &mut Engine, window: &dyn Window) -> Self {
///         window.set_title("My App");
///         MyApp { /* ... */ }
///     }
///
///     fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
///         // Use engine.input for input queries
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
    /// * `window` - Reference to the abstract window
    fn init(engine: &mut Engine, window: &dyn Window) -> Self;

    /// Handles raw platform events (advanced use only).
    ///
    /// Most applications should use `engine.input` in [`update`](Self::update) instead.
    ///
    /// When using the winit backend, `event` is `winit::event::WindowEvent`.
    /// Access it via `event.downcast_ref::<winit::event::WindowEvent>()`.
    ///
    /// Return `true` to consume the event (preventing default input processing),
    /// or `false` to allow normal engine input handling.
    #[allow(unused_variables)]
    fn on_event(
        &mut self,
        engine: &mut Engine,
        window: &dyn Window,
        event: &dyn std::any::Any,
    ) -> bool {
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
    /// * `window` - Reference to the abstract window
    /// * `frame` - Frame timing information
    #[allow(unused_variables)]
    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {}

    /// Configures the render pipeline for this frame.
    ///
    /// Override this method to add custom render passes (UI, post-processing, etc.).
    /// The default implementation only renders the built-in forward pass.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// fn compose_frame<'a>(&'a mut self, composer: FrameComposer<'a>) {
    ///     composer
    ///         .add_node(RenderStage::UI, &mut self.ui_pass)
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
    fn init(_engine: &mut Engine, _window: &dyn Window) -> Self {
        Self
    }
}
