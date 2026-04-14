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

#[doc(hidden)]
pub mod __macro_support {
    pub use env_logger;
    pub use log;
    pub use pollster;

    #[cfg(target_arch = "wasm32")]
    pub use console_error_panic_hook;
    #[cfg(target_arch = "wasm32")]
    pub use console_log;
    #[cfg(target_arch = "wasm32")]
    pub use wasm_bindgen;
    #[cfg(target_arch = "wasm32")]
    pub use wasm_bindgen_futures;

    pub trait WasmMainResult {
        fn report(self);
    }

    impl WasmMainResult for () {
        fn report(self) {}
    }

    impl<T, E> WasmMainResult for Result<T, E>
    where
        E: core::fmt::Display,
    {
        fn report(self) {
            if let Err(error) = self {
                log::error!("Myth application exited with error: {error}");
            }
        }
    }

    pub fn report_wasm_result<R>(result: R)
    where
        R: WasmMainResult,
    {
        result.report();
    }
}
