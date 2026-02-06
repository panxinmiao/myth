//! Render Settings Configuration
//!
//! This module defines the configuration options for the rendering system.
//!
//! # Example
//!
//! ```rust,ignore
//! use myth::render::RenderSettings;
//!
//! let settings = RenderSettings {
//!     vsync: false,
//!     clear_color: wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
//!     power_preference: wgpu::PowerPreference::HighPerformance,
//!     ..Default::default()
//! };
//!
//! App::new()
//!     .with_settings(settings)
//!     .run::<MyApp>()?;
//! ```

/// Configuration options for the rendering system.
///
/// This struct controls fundamental rendering parameters including GPU selection,
/// required features, and common render state settings.
///
/// # Fields
///
/// | Field | Description | Default |
/// |-------|-------------|---------|
/// | `enable_hdr` | Enable HDR rendering mode | `true` |
/// | `msaa_samples` | Number of MSAA samples | `1` |
/// | `vsync` | Vertical sync enabled | `true` |
/// | `clear_color` | Background clear color | Black |
/// | `power_preference` | GPU selection preference | `HighPerformance` |
/// | `required_features` | Required wgpu features | Empty |
/// | `required_limits` | Required wgpu limits | Default |
/// | `depth_format` | Depth buffer format | `Depth32Float` |
///
/// # GPU Selection
///
/// The `power_preference` field controls which GPU adapter is selected:
///
/// - `HighPerformance`: Prefer discrete GPU (better for games/visualization)
/// - `LowPower`: Prefer integrated GPU (better for battery life)
///
/// # Example
///
/// ```rust,ignore
/// use myth::render::RenderSettings;
///
/// // High-performance settings for games
/// let game_settings = RenderSettings {
///     power_preference: wgpu::PowerPreference::HighPerformance,
///     vsync: false, // Uncapped framerate
///     ..Default::default()
/// };
///
/// // Battery-friendly settings for tools
/// let tool_settings = RenderSettings {
///     power_preference: wgpu::PowerPreference::LowPower,
///     vsync: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RenderSettings {
    /// Whether to use straightforward rendering mode.
    ///
    /// if false, the main scene will be rendered directly to the screen surface,
    /// bypassing intermediate render targets and post-processing.
    /// This can improve performance for simple scenes without effects.
    pub enable_hdr: bool,

    /// Background clear color for the main render target.
    ///
    /// This color is used to clear the framebuffer at the start of each frame.
    pub clear_color: wgpu::Color,

    /// Enable vertical synchronization (`VSync`).
    ///
    /// When `true`, the framerate is capped to the display refresh rate,
    /// reducing screen tearing and power consumption.
    /// When `false`, the framerate is uncapped, which may cause tearing
    /// but reduces input latency.
    pub vsync: bool,

    /// Number of samples for multi-sample anti-aliasing (MSAA).
    ///
    /// Set to 1 to disable MSAA. Common values are 2, 4, or 8.
    /// Higher values improve quality but increase GPU load.
    pub msaa_samples: u32,

    /// GPU adapter selection preference.
    ///
    /// - `HighPerformance`: Prefer discrete/dedicated GPU
    /// - `LowPower`: Prefer integrated GPU
    pub power_preference: wgpu::PowerPreference,

    /// Required wgpu features that must be supported by the adapter.
    ///
    /// The engine will fail to initialize if these features are not available.
    /// Use with caution on WebGPU targets where feature support varies.
    pub required_features: wgpu::Features,

    /// Required wgpu limits that must be supported by the adapter.
    ///
    /// Limits define maximum resource sizes, binding counts, etc.
    pub required_limits: wgpu::Limits,

    /// Depth buffer texture format.
    ///
    /// `Depth32Float` is recommended for reverse-Z rendering (better precision).
    /// `Depth24PlusStencil8` can be used if stencil buffer is needed.
    pub depth_format: wgpu::TextureFormat,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::HighPerformance,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            clear_color: wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            vsync: true,
            msaa_samples: 1,
            depth_format: wgpu::TextureFormat::Depth32Float,
            enable_hdr: true,
        }
    }
}
