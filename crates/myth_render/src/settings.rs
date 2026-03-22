//! Renderer Settings & Render Path Configuration
//!
//! This module defines the rendering pipeline configuration for the engine.
//!
//! [`RenderPath`] determines the pipeline **topology** (which passes are
//! assembled, whether HDR targets are used, etc.), while anti-aliasing is
//! configured via [`RendererSettings::aa_mode`] using the unified
//! [`AntiAliasingMode`] enum.  This orthogonal design allows each axis to
//! be changed without affecting the other.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use myth::render::{RendererSettings, RenderPath, AntiAliasingMode};
//!
//! // Default: HighFidelity + TAA (recommended for PBR)
//! let settings = RendererSettings::default();
//!
//! // Lightweight forward pipeline with 4× MSAA for simple scenes
//! let settings = RendererSettings {
//!     path: RenderPath::BasicForward,
//!     aa_mode: AntiAliasingMode::MSAA(4),
//!     vsync: false,
//!     ..Default::default()
//! };
//!
//! App::new()
//!     .with_settings(settings)
//!     .run::<MyApp>()?;
//! ```

// Re-export AntiAliasingMode from myth_resources so downstream code can
// reference it via `myth_render::settings::AntiAliasingMode`.
pub use myth_resources::AntiAliasingMode;

// ---------------------------------------------------------------------------
// RenderPath
// ---------------------------------------------------------------------------

/// Determines the pipeline **topology** — which render passes are assembled,
/// whether HDR intermediates are allocated, and which post-processing chain
/// is available.
///
/// Anti-aliasing is configured **independently** via
/// [`RendererSettings::aa_mode`], keeping rasterization state orthogonal
/// to topology selection.
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
/// | TAA                     | ❌                | ✅                      |
/// | Depth-Normal Prepass    | ❌                | ✅ (auto-skipped w/ MSAA)|
/// | SSAO                    | ❌                | ✅                      |
/// | SSSS                    | ❌                | ✅                      |
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
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
    /// (Bloom → Tone Mapping → FXAA / TAA).  When MSAA is enabled via
    /// [`AntiAliasingMode::MSAA`], scene-drawing passes render into
    /// multi-sampled intermediates that are resolved at the appropriate
    /// pipeline stages.
    ///
    /// Includes Depth-Normal Prepass (auto-managed), SSAO, SSSS, and TAA.
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
    /// supply depth/normals to SSAO and SSSS.
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
            Self::HighFidelity => crate::HDR_TEXTURE_FORMAT,
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
/// allocate pipeline-level resources.  `path` can be changed
/// at runtime via [`Renderer::set_render_path`].
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
/// // Battery-friendly mobile setup
/// let mobile = RendererSettings {
///     path: RenderPath::BasicForward,
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

    /// Required wgpu features that must be supported by the adapter.
    ///
    /// The engine will fail to initialize if these features are unavailable.
    /// Use with caution on WebGPU targets where feature support varies.
    pub required_features: wgpu::Features,

    /// Required wgpu limits (max buffer sizes, binding counts, etc.).
    pub required_limits: wgpu::Limits,

    /// Depth buffer texture format.
    ///
    /// Defaults to `Depth32Float` — pure 32-bit floating-point depth with
    /// maximum precision and full `COPY_SRC`/`COPY_DST` support on all
    /// backends (including WebGPU).  Screen-space feature filtering (SSS,
    /// SSR) uses the `Feature_ID` colour attachment instead of a hardware
    /// stencil channel.
    pub depth_format: wgpu::TextureFormat,

    /// Gloobal anisotropic filtering level for default textures.
    pub anisotropy_clamp: u16,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            path: RenderPath::default(),
            vsync: true,
            backends: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            depth_format: wgpu::TextureFormat::Depth32Float,
            anisotropy_clamp: 16,
        }
    }
}
