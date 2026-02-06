//! Error Types
//!
//! This module defines the error types used throughout the engine.
//!
//! # Overview
//!
//! The main error type [`MythError`] covers all failure modes including:
//! - GPU initialization failures  
//! - Asset loading and decoding errors
//! - Resource management errors
//! - HTTP and network errors
//!
//! # Usage
//!
//! All public APIs return [`Result<T>`] which is an alias for `std::result::Result<T, MythError>`.
//!
//! ```rust,ignore
//! use myth::errors::{MythError, Result};
//!
//! fn load_asset() -> Result<()> {
//!     // Operations that may fail return Result
//!     Ok(())
//! }
//! ```

use thiserror::Error;

/// The main error type for the Myth engine.
///
/// This enum covers all possible error conditions that can occur
/// during engine operation. Each variant provides specific context
/// about what went wrong.
#[derive(Error, Debug)]
pub enum MythError {
    // ========================================================================
    // GPU & Rendering Errors
    // ========================================================================
    /// Failed to request a compatible GPU adapter.
    #[error("Failed to request WGPU adapter: {0}")]
    AdapterRequestFailed(String),

    /// Failed to create the GPU device.
    #[error("Failed to create WGPU device: {0}")]
    DeviceCreateFailed(#[from] wgpu::RequestDeviceError),

    /// Window system error.
    #[error("Window system error: {0}")]
    WindowError(#[from] raw_window_handle::HandleError),

    /// Event loop error (winit).
    #[cfg(feature = "winit")]
    #[error("Event loop error: {0}")]
    EventLoopError(#[from] winit::error::EventLoopError),

    // ========================================================================
    // Asset Loading Errors
    // ========================================================================
    /// The requested asset was not found.
    #[error("Asset not found: {0}")]
    AssetNotFound(String),

    /// Asset index out of bounds.
    #[error("Asset index out of bounds: {context} (index: {index})")]
    AssetIndexOutOfBounds {
        /// Description of what was being accessed
        context: String,
        /// The invalid index
        index: usize,
    },

    // ========================================================================
    // I/O Errors
    // ========================================================================
    /// File I/O error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    // ========================================================================
    // HTTP & Network Errors
    // ========================================================================
    /// HTTP request error.
    #[cfg(feature = "http")]
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// URL parsing error.
    #[cfg(feature = "http")]
    #[error("URL parse error: {0}")]
    UrlParseError(#[from] url::ParseError),

    /// HTTP response error with status code.
    #[error("HTTP response error: status {status}")]
    HttpResponseError {
        /// HTTP status code
        status: u16,
    },

    // ========================================================================
    // Image & Texture Errors
    // ========================================================================
    /// Image decoding error.
    #[error("Image decode error: {0}")]
    ImageDecodeError(String),

    /// Cube map validation error.
    #[error("Cube map error: {0}")]
    CubeMapError(String),

    // ========================================================================
    // Format & Parsing Errors
    // ========================================================================
    /// glTF parsing or loading error.
    #[cfg(feature = "gltf")]
    #[error("glTF error: {0}")]
    GltfError(String),

    /// Data URI parsing error.
    #[error("Data URI error: {0}")]
    DataUriError(String),

    /// JSON parsing error.
    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Base64 decoding error.
    #[error("Base64 decode error: {0}")]
    Base64Error(#[from] base64::DecodeError),

    // ========================================================================
    // Async & Threading Errors
    // ========================================================================
    /// Task join error (when async tasks fail to complete).
    #[error("Task join error: {0}")]
    TaskJoinError(String),

    // ========================================================================
    // Platform-Specific Errors
    // ========================================================================
    /// Feature not enabled.
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),

    /// WASM-specific error.
    #[cfg(target_arch = "wasm32")]
    #[error("WASM error: {0}")]
    WasmError(String),
}

// ============================================================================
// Convenient conversion implementations
// ============================================================================

impl From<image::ImageError> for MythError {
    fn from(err: image::ImageError) -> Self {
        MythError::ImageDecodeError(err.to_string())
    }
}

#[cfg(feature = "gltf")]
impl From<gltf::Error> for MythError {
    fn from(err: gltf::Error) -> Self {
        MythError::GltfError(err.to_string())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<tokio::task::JoinError> for MythError {
    fn from(err: tokio::task::JoinError) -> Self {
        MythError::TaskJoinError(err.to_string())
    }
}

/// Alias for `Result<T, MythError>`.
pub type Result<T> = std::result::Result<T, MythError>;
