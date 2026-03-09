//! Renderer Settings & Render Path Configuration
//!
//! This module defines the rendering pipeline configuration for the engine.
//!
//! [`RenderPath`] determines the pipeline **topology** (which passes are
//! assembled, whether HDR targets are used, etc.) while the **rasterization
//! state** (MSAA sample count) is configured independently via
//! [`RendererSettings::msaa_samples`].  This orthogonal design allows each
//! axis to be changed without affecting the other.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use myth::render::{RendererSettings, RenderPath};
//!
//! // Default: High-fidelity modern pipeline (HDR + post-processing)
//! let settings = RendererSettings::default();
//!
//! // Lightweight forward pipeline with 4× MSAA for simple scenes
//! let settings = RendererSettings {
//!     path: RenderPath::BasicForward,
//!     msaa_samples: 4,
//!     vsync: false,
//!     ..Default::default()
//! };
//!
//! App::new()
//!     .with_settings(settings)
//!     .run::<MyApp>()?;
//! ```

// ---------------------------------------------------------------------------
// RenderPath
// ---------------------------------------------------------------------------

/// Determines the pipeline **topology** — which render passes are assembled,
/// whether HDR intermediates are allocated, and which post-processing chain
/// is available.
///
/// MSAA sample count is configured **independently** via
/// [`RendererSettings::msaa_samples`], keeping rasterization state orthogonal
/// to topology selection.  Both paths support hardware MSAA when
/// `msaa_samples > 1`.
///
/// # Path Comparison
///
/// | Capability              | `BasicForward`    | `HighFidelity`          |
/// |-------------------------|-------------------|-------------------------|
/// | Hardware MSAA           | ✅ (configurable) | ✅ (configurable)       |
/// | HDR render targets      | ❌                | ✅                      |
/// | Bloom                   | ❌                | ✅                      |
/// | Tone Mapping            | ❌                | ✅                      |
/// | FXAA (post-process AA)  | ❌                | ✅                      |
/// | Depth-Normal Prepass    | ❌                | ✅ (auto-skipped w/ MSAA)|
/// | SSAO                    | ❌                | ✅                      |
/// | SSSSS                   | ❌                | ✅                      |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPath {
    /// Lightweight forward rendering pipeline.
    ///
    /// Renders the scene directly to the surface (or an LDR intermediate when
    /// MSAA is active). No HDR render targets, no bloom, no tone mapping.
    ///
    /// Best suited for:
    /// - Low-end / mobile devices
    /// - Simple 3D or 2D/UI scenes that do not need advanced lighting
    /// - Scenarios where hardware MSAA is preferred over post-process AA
    BasicForward,

    /// High-fidelity hybrid rendering pipeline.
    ///
    /// Uses HDR floating-point render targets and a full post-processing chain
    /// (Bloom → Tone Mapping → FXAA). When MSAA is enabled
    /// (`RendererSettings::msaa_samples > 1`), scene-drawing passes render
    /// into multi-sampled intermediates that are resolved at the appropriate
    /// pipeline stages.
    ///
    /// Includes Depth-Normal Prepass (auto-managed), SSAO, and SSSSS.
    ///
    /// Best suited for:
    /// - Desktop / high-end mobile with modern GPUs
    /// - PBR scenes requiring physically-correct lighting and effects
    /// - Any application that benefits from bloom, tone mapping, or SSAO
    HighFidelity,
}

impl Default for RenderPath {
    #[inline]
    fn default() -> Self {
        Self::HighFidelity
    }
}

impl RenderPath {
    /// Returns `true` when this path enables post-processing (HDR targets,
    /// bloom, tone mapping, FXAA, etc.).
    #[inline]
    #[must_use]
    pub fn supports_post_processing(&self) -> bool {
        matches!(self, Self::HighFidelity)
    }

    /// Returns `true` when this path supports a depth-normal prepass.
    ///
    /// When hardware MSAA is enabled, the prepass Early-Z benefit is lost
    /// for the main scene draw; the prepass may still be scheduled to
    /// supply depth/normals to SSAO and SSSSS.
    #[inline]
    #[must_use]
    pub fn requires_z_prepass(&self) -> bool {
        matches!(self, Self::HighFidelity)
    }

    /// Returns the main color attachment format for scene rendering.
    ///
    /// - [`HighFidelity`](Self::HighFidelity): HDR float format (`Rgba16Float`)
    /// - [`BasicForward`](Self::BasicForward): the supplied surface format (LDR)
    #[inline]
    #[must_use]
    pub fn main_color_format(&self, surface_format: wgpu::TextureFormat) -> wgpu::TextureFormat {
        match self {
            Self::HighFidelity => crate::renderer::HDR_TEXTURE_FORMAT,
            Self::BasicForward => surface_format,
        }
    }
}

// ---------------------------------------------------------------------------
// RendererSettings
// ---------------------------------------------------------------------------

/// Global configuration for renderer initialization.
///
/// Consumed once during [`Renderer::init`] to set up the GPU context and
/// allocate pipeline-level resources.  Both `path` and `msaa_samples` can
/// be changed at runtime via [`Renderer::set_render_path`] and
/// [`Renderer::set_msaa_samples`] respectively.
///
/// # Example
///
/// ```rust,ignore
/// use myth::render::{RendererSettings, RenderPath};
///
/// // High-performance gaming setup
/// let game = RendererSettings {
///     path: RenderPath::HighFidelity,
///     vsync: false,
///     ..Default::default()
/// };
///
/// // Battery-friendly mobile setup with 4× MSAA
/// let mobile = RendererSettings {
///     path: RenderPath::BasicForward,
///     msaa_samples: 4,
///     power_preference: wgpu::PowerPreference::LowPower,
///     vsync: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RendererSettings {
    // === Core Pipeline Configuration ===
    /// The rendering pipeline topology.
    ///
    /// Determines which passes are assembled into the frame graph and which
    /// post-processing effects are available.  See [`RenderPath`].
    pub path: RenderPath,

    /// Hardware MSAA sample count (1 = disabled, common values: 2, 4, 8).
    ///
    /// Orthogonal to `path` — both `BasicForward` and `HighFidelity`
    /// support hardware multi-sampling when this value is > 1.
    pub msaa_samples: u32,

    /// Enable vertical synchronization (VSync).
    ///
    /// When `true`, the frame rate is capped to the display refresh rate,
    /// reducing screen tearing and power consumption.
    /// When `false`, the frame rate is uncapped, which may cause tearing
    /// but reduces input latency.
    pub vsync: bool,

    // === GPU / Backend Configuration ===
    /// Force a specific wgpu backend (Vulkan, Metal, DX12, …).
    ///
    /// `None` lets wgpu choose the best available backend for the platform.
    /// Override this only when debugging backend-specific issues.
    pub backends: Option<wgpu::Backends>,

    /// GPU adapter selection preference.
    ///
    /// - `HighPerformance`: Prefer discrete / dedicated GPU
    /// - `LowPower`: Prefer integrated GPU (better battery life)
    pub power_preference: wgpu::PowerPreference,

    // === Rendering Defaults ===
    /// Background clear color for the main render target.
    ///
    /// Used to clear the framebuffer at the start of each frame.
    /// May be overridden at runtime by the active scene's background settings.
    pub clear_color: wgpu::Color,

    /// Required wgpu features that must be supported by the adapter.
    ///
    /// The engine will fail to initialize if these features are unavailable.
    /// Use with caution on WebGPU targets where feature support varies.
    pub required_features: wgpu::Features,

    /// Required wgpu limits (max buffer sizes, binding counts, etc.).
    pub required_limits: wgpu::Limits,

    /// Depth buffer texture format.
    ///
    /// Defaults to `Depth24PlusStencil8` which provides both depth precision
    /// and a stencil buffer (needed for SSS feature IDs, etc.).
    /// Use [`Depth32Float`](wgpu::TextureFormat::Depth32Float) if you only
    /// need depth and want maximum precision.
    pub depth_format: wgpu::TextureFormat,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            path: RenderPath::default(),
            msaa_samples: 1,
            vsync: true,
            backends: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            clear_color: wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            depth_format: wgpu::TextureFormat::Depth24PlusStencil8,
        }
    }
}

impl RendererSettings {
    /// Returns the effective MSAA sample count.
    #[inline]
    #[must_use]
    pub fn msaa_samples(&self) -> u32 {
        self.msaa_samples
    }
}

// Backward-compatible type alias — prefer `RendererSettings` in new code.
#[doc(hidden)]
#[deprecated(since = "0.2.0", note = "Renamed to `RendererSettings`")]
pub type RenderSettings = RendererSettings;
