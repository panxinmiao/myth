use thiserror::Error;

#[derive(Error, Debug)]
pub enum ThreeError {
    #[error("Failed to request WGPU adapter: {0}")]
    AdapterRequestFailed(String),

    #[error("Failed to create WGPU device: {0}")]
    DeviceCreateFailed(#[from] wgpu::RequestDeviceError),

    #[error("Window system error: {0}")]
    WindowError(#[from] winit::error::OsError),

    #[error("Asset not found: {0}")]
    AssetNotFound(String),
    
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image decode error: {0}")]
    ImageError(String),
}


pub type Result<T> = std::result::Result<T, ThreeError>;