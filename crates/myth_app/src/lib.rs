//! Myth Engine — Application Framework
//!
//! This crate provides the application lifecycle management, windowing
//! integration, and the central [`Engine`] coordinator that ties all
//! subsystems together.

pub mod app;
pub mod engine;
pub mod window;

#[cfg(feature = "winit")]
pub mod winit;

pub use app::{AppHandler, DefaultHandler};
pub use engine::{Engine, FrameState};
pub use window::Window;
