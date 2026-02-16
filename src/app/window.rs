//! Platform-independent window abstraction.
//!
//! Defines a [`Window`] trait that decouples application logic from
//! specific windowing backends (e.g., winit). User code interacts with
//! this trait instead of concrete window types.
//!
//! When using the `winit` backend, the concrete implementation is
//! `winit::window::Window`. Advanced users can downcast via [`Window::as_any`]
//! to access platform-specific functionality.

use glam::Vec2;

/// Platform-independent window interface.
///
/// Provides core window operations that application code typically needs:
/// setting the title, querying dimensions, requesting redraws, etc.
///
/// # Backend Access
///
/// For advanced use cases (e.g., UI framework integration), the underlying
/// platform window can be accessed via [`as_any`](Self::as_any) and downcasting:
///
/// ```rust,ignore
/// // When using the winit backend:
/// if let Some(winit_window) = window.as_any().downcast_ref::<winit::window::Window>() {
///     // Access winit-specific APIs
/// }
/// ```
pub trait Window: Send + Sync {
    /// Sets the window title.
    fn set_title(&self, title: &str);

    /// Returns the window's inner (client area) size in physical pixels.
    fn inner_size(&self) -> Vec2;

    /// Returns the display scale factor (DPI scaling).
    fn scale_factor(&self) -> f32;

    /// Requests the window to redraw.
    fn request_redraw(&self);

    /// Shows or hides the mouse cursor.
    fn set_cursor_visible(&self, visible: bool);

    /// Returns the underlying platform window as `Any` for downcasting.
    ///
    /// This enables advanced users to access platform-specific APIs
    /// without coupling the core engine to any particular windowing backend.
    fn as_any(&self) -> &dyn std::any::Any;
}
