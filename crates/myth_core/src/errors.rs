//! Error types for the Myth engine.
//!
//! # Hierarchy
//!
//! - [`Error`] — top-level enum delegating to sub-categories
//!   - [`PlatformError`] — window system and adapter errors
//!   - [`AssetError`] — I/O, network, parsing errors
//!   - [`RenderError`] — GPU device and shader errors

use thiserror::Error;

// ============================================================================
// Top-Level Error
// ============================================================================

/// Top-level error type for the Myth engine.
#[derive(Error, Debug)]
pub enum Error {
    /// Platform and window system errors.
    #[error("Platform error: {0}")]
    Platform(#[from] PlatformError),

    /// Asset loading and processing errors.
    #[error("Asset error: {0}")]
    Asset(#[from] AssetError),

    /// Rendering and GPU errors.
    #[error("Render error: {0}")]
    Render(#[from] RenderError),

    /// General engine error.
    #[error("Engine error: {0}")]
    General(String),
}

// ============================================================================
// Asset Errors
// ============================================================================

/// Errors related to asset loading and processing.
#[derive(Error, Debug)]
pub enum AssetError {
    /// Asset not found.
    #[error("Asset not found: {0}")]
    NotFound(String),

    /// File I/O error.
    #[error("Failed to load file: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP request failed.
    #[error("Network request failed: {0}")]
    Network(String),

    /// HTTP response with non-success status.
    #[error("HTTP response error: status {status}")]
    HttpResponse {
        /// HTTP status code.
        status: u16,
    },

    /// Data format parsing failure.
    #[error("Failed to parse data format: {0}")]
    Format(String),

    /// Invalid asset data.
    #[error("Invalid asset data: {0}")]
    InvalidData(String),

    /// Base64 decoding error.
    #[error("Base64 decode error: {0}")]
    Base64Decode(String),

    /// Async task join error.
    #[error("Task join error: {0}")]
    TaskJoin(String),
}

// ============================================================================
// Platform Errors
// ============================================================================

/// Errors related to platform and window system.
#[derive(Error, Debug)]
pub enum PlatformError {
    /// Window handle error.
    #[error("Window handle error: {0}")]
    WindowHandle(String),

    /// Event loop error.
    #[error("Event loop error: {0}")]
    EventLoop(String),

    /// No compatible GPU adapter found.
    #[error("No compatible GPU adapter found: {0}")]
    AdapterNotFound(String),

    /// Surface configuration failed.
    #[error("Failed to create surface: {0}")]
    SurfaceConfigFailed(String),

    /// WASM-specific error.
    #[error("WASM error: {0}")]
    Wasm(String),

    /// Feature not enabled.
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

// ============================================================================
// Render Errors
// ============================================================================

/// Errors related to rendering and GPU operations.
#[derive(Error, Debug)]
pub enum RenderError {
    /// GPU device creation error.
    #[error("WGPU device error: {0}")]
    RequestDeviceFailed(String),

    /// Shader compilation failed.
    #[error("Shader compilation failed: {0}")]
    ShaderCompile(String),

    /// Render graph error.
    #[error("Render graph error: {0}")]
    Graph(String),
}

/// Alias for `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;
