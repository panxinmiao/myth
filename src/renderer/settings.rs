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

use crate::resources::fxaa::FxaaSettings;
use crate::resources::taa::TaaSettings;

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
            Self::HighFidelity => crate::renderer::HDR_TEXTURE_FORMAT,
            Self::BasicForward => surface_format,
        }
    }
}



/// Unified anti-aliasing mode for the rendering pipeline.
///
/// Each variant carries its own configuration payload, forming an algebraic
/// data type (ADT) that eliminates "ghost state" — only the settings for
/// the currently active technique exist in memory.
///
/// The engine automatically manages MSAA render targets, FXAA post-process
/// passes, and TAA temporal state based on the selected mode.
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum AntiAliasingMode {
    /// No anti-aliasing.  Maximum performance.
    None,
    /// FXAA only.  Minimal overhead — smooths high-frequency noise but
    /// produces softer geometric edges.  Good for low-end / Web targets.
    FXAA(FxaaSettings),
    /// Hardware multi-sampling (e.g. 4×).  Crisp geometric edges but may
    /// exhibit PBR specular flickering.  Best for non-PBR / toon styles.
    MSAA(u32),
    /// MSAA + FXAA.  MSAA resolves geometric edges, FXAA removes specular
    /// shimmer.  Best static image quality with zero temporal ghosting.
    MSAA_FXAA(u32, FxaaSettings),
    /// Temporal Anti-Aliasing — the **recommended default** for PBR.
    /// Resolves all aliasing categories with slight temporal softening.
    TAA(TaaSettings),
}

impl Default for AntiAliasingMode {
    fn default() -> Self {
        Self::TAA(TaaSettings::default())
    }
}

impl AntiAliasingMode {
    /// Returns the hardware MSAA sample count implied by this AA mode.
    ///
    /// Only [`MSAA`](Self::MSAA) and [`MSAA_FXAA`](Self::MSAA_FXAA)
    /// produce a value greater than 1.  All other modes rasterize at 1×.
    #[inline]
    #[must_use]
    pub fn msaa_sample_count(&self) -> u32 {
        match self {
            Self::MSAA(s) | Self::MSAA_FXAA(s, _) => *s,
            _ => 1,
        }
    }

    /// Returns `true` when TAA is the active technique.
    #[inline]
    #[must_use]
    pub fn is_taa(&self) -> bool {
        matches!(self, Self::TAA(_))
    }

    /// Returns `true` when FXAA is active (standalone or combined with MSAA).
    #[inline]
    #[must_use]
    pub fn is_fxaa(&self) -> bool {
        matches!(self, Self::FXAA(_) | Self::MSAA_FXAA(..)) 
    }

    /// Returns the [`FxaaSettings`] if the current mode uses FXAA.
    #[inline]
    #[must_use]
    pub fn fxaa_settings(&self) -> Option<&FxaaSettings> {
        match self {
            Self::FXAA(s) | Self::MSAA_FXAA(_, s) => Some(s),
            _ => Option::None,
        }
    }

    /// Returns the [`TaaSettings`] if the current mode is TAA.
    #[inline]
    #[must_use]
    pub fn taa_settings(&self) -> Option<&TaaSettings> {
        match self {
            Self::TAA(s) => Some(s),
            _ => Option::None,
        }
    }
}


// ---------------------------------------------------------------------------
// RendererSettings
// ---------------------------------------------------------------------------

/// Global configuration for renderer initialization.
///
/// Consumed once during [`Renderer::init`] to set up the GPU context and
/// allocate pipeline-level resources.  `path` and `aa_mode` can be changed
/// at runtime via [`Renderer::set_render_path`] and
/// [`Renderer::set_aa_mode`] respectively.
///
/// # Example
///
/// ```rust,ignore
/// use myth::render::{RendererSettings, RenderPath, AntiAliasingMode};
///
/// // High-performance gaming setup (TAA is the default)
/// let game = RendererSettings {
///     path: RenderPath::HighFidelity,
///     vsync: false,
///     ..Default::default()
/// };
///
/// // Battery-friendly mobile setup with 4× MSAA
/// let mobile = RendererSettings {
///     path: RenderPath::BasicForward,
///     aa_mode: AntiAliasingMode::MSAA(4),
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

    /// Anti-aliasing mode.  Controls MSAA, FXAA and TAA in a unified way.
    /// See [`AntiAliasingMode`] for available options.
    pub aa_mode: AntiAliasingMode,

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
            aa_mode: AntiAliasingMode::default(),
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
    /// Returns the hardware MSAA sample count implied by the current AA mode.
    #[inline]
    #[must_use]
    pub fn msaa_sample_count(&self) -> u32 {
        self.aa_mode.msaa_sample_count()
    }

    /// Returns `true` when TAA is the active anti-aliasing mode.
    #[inline]
    #[must_use]
    pub fn is_taa_enabled(&self) -> bool {
        self.aa_mode.is_taa()
    }

    /// Returns `true` when FXAA is active (either standalone or combined with MSAA).
    #[inline]
    #[must_use]
    pub fn is_fxaa_enabled(&self) -> bool {
        self.aa_mode.is_fxaa()
    }
}


