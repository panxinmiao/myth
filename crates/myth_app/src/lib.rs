//! Myth Engine — Application Framework
//!
//! This crate provides the application lifecycle management, windowing
//! integration, and the central [`Engine`] coordinator that ties all
//! subsystems together.

pub mod app;
pub mod engine;
pub mod orbit_controls;
pub mod window;

#[cfg(feature = "winit")]
pub mod winit;

pub use app::{AppHandler, DefaultHandler};
pub use engine::{Engine, FrameState};
pub use orbit_controls::OrbitControls;
pub use window::Window;
