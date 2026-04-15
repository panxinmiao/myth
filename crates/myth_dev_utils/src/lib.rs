//! Internal developer-facing utilities shared by Myth examples and apps.
//!
//! This crate is intentionally not published. It collects tooling that is
//! useful for demos, gallery builds, inspectors, and debug overlays without
//! polluting the stable engine runtime surface.

pub mod fps_counter;
pub mod time;
pub mod ui_pass;

pub use egui;
pub use fps_counter::FpsCounter;
pub use time::Timer;
pub use ui_pass::{UiPass, UiPassNode};

pub mod prelude {
    pub use crate::{FpsCounter, Timer, UiPass, UiPassNode};
}
