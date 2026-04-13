//! Internal developer-facing utilities shared by Myth examples and apps.
//!
//! This crate is intentionally not published. It collects tooling that is
//! useful for demos, gallery builds, inspectors, and debug overlays without
//! polluting the stable engine runtime surface.

pub mod ui_pass;

pub use myth_app::OrbitControls;
pub use myth_core::utils::FpsCounter;
pub use ui_pass::{UiPass, UiPassNode};

pub mod prelude {
    pub use crate::{FpsCounter, OrbitControls, UiPass, UiPassNode};
}