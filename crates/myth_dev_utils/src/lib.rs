//! Internal developer-facing utilities shared by Myth examples and apps.
//!
//! This crate is intentionally not published. It collects tooling that is
//! useful for demos, gallery builds, inspectors, and debug overlays without
//! polluting the stable engine runtime surface.

pub mod fps_counter;
pub mod ocean;
pub mod time;

pub use fps_counter::FpsCounter;
pub use ocean::{
    OceanCameraSource, OceanLightSource, OceanPreset, OceanQuality, OceanRenderer, OceanSettings,
};
pub use time::Timer;

pub mod prelude {
    pub use crate::{
        FpsCounter, OceanCameraSource, OceanLightSource, OceanPreset, OceanQuality, OceanRenderer,
        OceanSettings, Timer,
    };
}
