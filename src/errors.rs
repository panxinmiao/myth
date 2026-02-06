//! Error Types
//!
//! This module defines the hierarchical error types used throughout the engine.
//!
//! # Overview
//!
//! Errors are organized into three main categories:
//! - **Platform errors** ([`PlatformError`]): Window system, event loop, and adapter issues
//! - **Asset errors** ([`AssetError`]): I/O, network, parsing, and data validation issues
//! - **Render errors** ([`RenderError`]): GPU device and shader compilation issues
//!
//! # Usage
//!
//! All public APIs return [`Result<T>`] which is an alias for `std::result::Result<T, Error>`.
//!
//! ```rust,ignore
//! use myth::errors::{Error, Result, AssetError};
//!
//! fn load_asset() -> Result<()> {
//!     // Errors are automatically converted via From implementations
//!     let data = std::fs::read("file.txt")?; // io::Error -> AssetError -> Error
//!     Ok(())
//! }
//! ```

use thiserror::Error;

// ============================================================================
// Top-Level Error Type
// ============================================================================

/// The top-level error type for the Myth engine.
///
/// This enum delegates to specialized sub-error types for better organization
/// and more specific error handling.
#[derive(Error, Debug)]
pub enum Error {
    /// Platform and window system related errors.
    #[error("Platform error: {0}")]
    Platform(#[from] PlatformError),

    /// Asset loading and processing related errors (I/O, Network, Parsing).
    #[error("Asset error: {0}")]
    Asset(#[from] AssetError),

    /// Rendering and GPU core related errors.
    #[error("Render error: {0}")]
    Render(#[from] RenderError),

    /// General engine error for miscellaneous cases.
    #[error("Engine error: {0}")]
    General(String),
}

// ============================================================================
// Sub-Category: AssetError
// Covers: I/O, Network, Image, glTF, Parsing
// ============================================================================

/// Errors related to asset loading and processing.
///
/// This covers file I/O, network requests, image decoding,
/// format parsing (glTF, JSON), and data validation.
#[derive(Error, Debug)]
pub enum AssetError {
    /// The requested asset was not found.
    #[error("Asset not found: {0}")]
    NotFound(String),

    /// File I/O error.
    #[error("Failed to load file: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP request error.
    #[cfg(feature = "http")]
    #[error("Network request failed: {0}")]
    Network(#[from] reqwest::Error),

    /// URL parsing error.
    #[cfg(feature = "http")]
    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    /// HTTP response error with status code.
    #[error("HTTP response error: status {status}")]
    HttpResponse {
        /// HTTP status code
        status: u16,
    },

    /// Failed to parse data format (Image, glTF, JSON, etc.)
    #[error("Failed to parse data format: {0}")]
    Format(String),

    /// Invalid asset data (cube map validation, index out of bounds, etc.)
    #[error("Invalid asset data: {0}")]
    InvalidData(String),

    /// Base64 decoding error.
    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),

    /// Task join error (when async tasks fail to complete).
    #[error("Task join error: {0}")]
    TaskJoin(String),
}

// ============================================================================
// Sub-Category: PlatformError
// Covers: Winit, WindowHandle, Adapter discovery
// ============================================================================

/// Errors related to platform and window system.
///
/// This covers window handle errors, event loop issues,
/// and GPU adapter discovery failures.
#[derive(Error, Debug)]
pub enum PlatformError {
    /// Window handle error.
    #[error("Window handle error: {0}")]
    WindowHandle(#[from] raw_window_handle::HandleError),

    /// Event loop error (winit).
    #[cfg(feature = "winit")]
    #[error("Event loop error: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),

    /// No compatible GPU adapter found.
    #[error("No compatible GPU adapter found: {0}")]
    AdapterNotFound(String),

    /// Failed to create surface.
    #[error("Failed to create surface: {0}")]
    SurfaceConfigFailed(String),

    /// WASM-specific error.
    #[cfg(target_arch = "wasm32")]
    #[error("WASM error: {0}")]
    Wasm(String),

    /// Feature not enabled.
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

// ============================================================================
// Sub-Category: RenderError
// Covers: WGPU Device, Shader Compilation, Render Graph
// ============================================================================

/// Errors related to rendering and GPU operations.
///
/// This covers GPU device creation, shader compilation,
/// and render graph construction failures.
#[derive(Error, Debug)]
pub enum RenderError {
    /// WGPU device creation error.
    #[error("WGPU device error: {0}")]
    RequestDeviceFailed(#[from] wgpu::RequestDeviceError),

    /// Shader compilation failed.
    #[error("Shader compilation failed: {0}")]
    ShaderCompile(String),

    /// Render graph error.
    #[error("Render graph error: {0}")]
    Graph(String),
}

// ============================================================================
// Convenient conversion implementations for AssetError
// ============================================================================

impl From<image::ImageError> for AssetError {
    fn from(err: image::ImageError) -> Self {
        AssetError::Format(format!("Image error: {}", err))
    }
}

#[cfg(feature = "gltf")]
impl From<gltf::Error> for AssetError {
    fn from(err: gltf::Error) -> Self {
        AssetError::Format(format!("glTF error: {}", err))
    }
}

impl From<serde_json::Error> for AssetError {
    fn from(err: serde_json::Error) -> Self {
        AssetError::Format(format!("JSON error: {}", err))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<tokio::task::JoinError> for AssetError {
    fn from(err: tokio::task::JoinError) -> Self {
        AssetError::TaskJoin(err.to_string())
    }
}

// ============================================================================
// Convenient conversion implementations for top-level Error
// These allow ? operator to work seamlessly across error types
// ============================================================================

impl From<image::ImageError> for Error {
    fn from(err: image::ImageError) -> Self {
        Error::Asset(AssetError::from(err))
    }
}

#[cfg(feature = "gltf")]
impl From<gltf::Error> for Error {
    fn from(err: gltf::Error) -> Self {
        Error::Asset(AssetError::from(err))
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Asset(AssetError::from(err))
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Asset(AssetError::from(err))
    }
}

#[cfg(feature = "http")]
impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Error::Asset(AssetError::from(err))
    }
}

#[cfg(feature = "http")]
impl From<url::ParseError> for Error {
    fn from(err: url::ParseError) -> Self {
        Error::Asset(AssetError::from(err))
    }
}

impl From<base64::DecodeError> for Error {
    fn from(err: base64::DecodeError) -> Self {
        Error::Asset(AssetError::from(err))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<tokio::task::JoinError> for Error {
    fn from(err: tokio::task::JoinError) -> Self {
        Error::Asset(AssetError::from(err))
    }
}

impl From<raw_window_handle::HandleError> for Error {
    fn from(err: raw_window_handle::HandleError) -> Self {
        Error::Platform(PlatformError::from(err))
    }
}

#[cfg(feature = "winit")]
impl From<winit::error::EventLoopError> for Error {
    fn from(err: winit::error::EventLoopError) -> Self {
        Error::Platform(PlatformError::from(err))
    }
}

impl From<wgpu::RequestDeviceError> for Error {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        Error::Render(RenderError::from(err))
    }
}

/// Alias for `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;
