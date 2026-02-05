//! Error Types
//!
//! This module defines the error types used throughout the engine.
//!
//! The main error type [`MythError`] covers all failure modes including:
//! - GPU initialization failures
//! - Asset loading errors
//! - Resource management errors

use thiserror::Error;

/// The main error type for the Myth engine.
///
/// This enum covers all possible error conditions that can occur
/// during engine operation.
#[derive(Error, Debug)]
pub enum MythError {
    /// Failed to request a compatible GPU adapter.
    #[error("Failed to request WGPU adapter: {0}")]
    AdapterRequestFailed(String),

    /// Failed to create the GPU device.
    #[error("Failed to create WGPU device: {0}")]
    DeviceCreateFailed(#[from] wgpu::RequestDeviceError),

    /// Window system error.
    #[error("Window system error: {0}")]
    WindowError(#[from] raw_window_handle::HandleError),

    /// The requested asset was not found.
    #[error("Asset not found: {0}")]
    AssetNotFound(String),

    /// File I/O error.
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    /// Image decoding error.
    #[error("Image decode error: {0}")]
    ImageError(String),
}

/// Alias for `Result<T, MythError>`.
pub type Result<T> = std::result::Result<T, MythError>;
