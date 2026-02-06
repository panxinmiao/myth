//! Application Framework Module
//!
//! This module provides the application lifecycle management and windowing integration.
//! It bridges the engine core with platform-specific window systems.
//!
//! # Architecture
//!
//! The app module follows a trait-based design where users implement [`winit::AppHandler`]
//! to define their application behavior, while the framework handles:
//!
//! - Window creation and management
//! - Event loop processing
//! - Input translation
//! - Frame timing
//!
//! # Platform Support
//!
//! Currently supports:
//! - **Winit** (default): Cross-platform windowing via the winit crate
//!
//! # Example
//!
//! ```rust,ignore
//! use myth::app::winit::{App, AppHandler};
//!
//! struct MyGame;
//!
//! impl AppHandler for MyGame {
//!     fn init(engine: &mut MythEngine, window: &Arc<Window>) -> Self {
//!         MyGame
//!     }
//!
//!     fn update(&mut self, engine: &mut MythEngine, window: &Arc<Window>, frame: &FrameState) {
//!         // Game logic here
//!     }
//! }
//!
//! App::new().with_title("My Game").run::<MyGame>()?;
//! ```

#[cfg(feature = "winit")]
pub mod winit;
