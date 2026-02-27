//! Renderer Settings & Render Path Configuration
//!
//! This module defines the rendering pipeline configuration for the engine.
//!
//! The core abstraction is [`RenderPath`], which determines whether the engine
//! operates in a lightweight forward-only mode or a full high-fidelity pipeline
//! with HDR, post-processing, and (future) depth-normal prepass / SSAO support.
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
//!     path: RenderPath::BasicForward { msaa_samples: 4 },
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

/// Defines the core rendering path and pipeline topology.
///
/// The engine supports two fundamentally different rendering paths that control
/// which passes are assembled into the frame graph, how anti-aliasing is handled,
/// and which post-processing effects are available.
///
/// # Path Comparison
///
/// | Capability              | `BasicForward`    | `HighFidelity`          |
/// |-------------------------|-------------------|-------------------------|
/// | Hardware MSAA           | ✅ (configurable) | ❌ (forced to 1)        |
/// | HDR render targets      | ❌                | ✅                      |
/// | Bloom                   | ❌                | ✅                      |
/// | Tone Mapping            | ❌                | ✅                      |
/// | FXAA (post-process AA)  | ❌                | ✅                      |
/// | Depth-Normal Prepass¹   | ❌                | ✅                      |
/// | SSAO¹                   | ❌                | ✅                      |
/// | SSSSS¹                  | ❌                | ✅                      |
///
/// ¹ Planned features — the `HighFidelity` path is designed to accommodate
///   these without further API changes.
///
/// # Design Rationale
///
/// Hardware MSAA and HDR post-processing are fundamentally at odds in a modern
/// rendering pipeline: MSAA requires multi-sampled render targets that are
/// expensive to resolve and incompatible with most screen-space effects. By
/// making the choice explicit at the path level, the engine can allocate
/// resources optimally and avoid hidden performance cliffs.
#[derive(Debug, Clone, PartialEq, Copy, Eq)]
pub enum RenderPath {
    /// Lightweight forward rendering pipeline.
    ///
    /// Renders the scene directly to the surface (or an LDR intermediate when
    /// MSAA is active). No HDR render targets, no bloom, no tone mapping.
    ///
    /// Supports hardware multi-sample anti-aliasing (MSAA) with configurable
    /// sample count.
    ///
    /// Best suited for:
    /// - Low-end / mobile devices
    /// - Simple 3D or 2D/UI scenes that do not need advanced lighting
    /// - Scenarios where hardware MSAA is preferred over post-process AA
    BasicForward {
        /// MSAA sample count. Common values: 1 (off), 2, 4, 8.
        msaa_samples: u32,
    },

    /// High-fidelity hybrid rendering pipeline.
    ///
    /// Uses HDR floating-point render targets and a full post-processing chain
    /// (Bloom → Tone Mapping → FXAA). Hardware MSAA is **forced to 1** in this
    /// mode; anti-aliasing should be achieved through screen-space techniques
    /// such as FXAA (or TAA in the future).
    ///
    /// This path is designed to host a Depth-Normal Prepass and SSAO in future
    /// releases without requiring API-level changes.
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
        // Modern engines default to the high-fidelity pipeline.
        Self::HighFidelity
    }
}

impl RenderPath {
    /// Returns `true` when this path enables post-processing (HDR targets,
    /// bloom, tone mapping, FXAA, etc.).
    ///
    /// Currently only [`HighFidelity`](Self::HighFidelity) supports post-processing.
    #[inline]
    #[must_use]
    pub fn supports_post_processing(&self) -> bool {
        matches!(self, Self::HighFidelity)
    }

    /// Returns `true` when this path will use a depth-normal prepass.
    ///
    /// Reserved for future use — always returns `false` for
    /// [`BasicForward`](Self::BasicForward) and `true` for
    /// [`HighFidelity`](Self::HighFidelity) once SSAO / depth-normal
    /// pre-pass is implemented.
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
            Self::BasicForward { .. } => surface_format,
        }
    }

    /// Returns the effective MSAA sample count for this path.
    ///
    /// [`HighFidelity`](Self::HighFidelity) always returns **1**.
    #[inline]
    #[must_use]
    pub fn msaa_samples(&self) -> u32 {
        match self {
            Self::BasicForward { msaa_samples } => *msaa_samples,
            Self::HighFidelity => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// RendererSettings
// ---------------------------------------------------------------------------

/// Global configuration for renderer initialization.
///
/// This struct is consumed once during [`Renderer::init`] to set up the GPU
/// context and allocate pipeline-level resources. Runtime changes to the
/// render path are possible via [`Renderer::set_render_path`].
///
/// # Fields
///
/// | Field              | Description                              | Default            |
/// |--------------------|------------------------------------------|--------------------|
/// | `path`             | Render pipeline path                     | `HighFidelity`     |
/// | `vsync`            | Vertical sync enabled                    | `true`             |
/// | `backends`         | Forced wgpu backend (or auto)            | `None`             |
/// | `power_preference` | GPU adapter selection strategy            | `HighPerformance`  |
/// | `clear_color`      | Default framebuffer clear color          | Black (0,0,0,1)    |
/// | `required_features`| Required wgpu features                   | Empty              |
/// | `required_limits`  | Required wgpu limits                     | Default            |
/// | `depth_format`     | Depth buffer texture format              | `Depth32Float`     |
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
///     path: RenderPath::BasicForward { msaa_samples: 4 },
///     power_preference: wgpu::PowerPreference::LowPower,
///     vsync: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RendererSettings {
    // === Core Pipeline Configuration ===
    /// The rendering pipeline path.
    ///
    /// Determines the overall pipeline topology, available post-processing
    /// effects, and MSAA allocation strategy. See [`RenderPath`] for details.
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
    /// [`Depth32Float`](wgpu::TextureFormat::Depth32Float) is recommended for
    /// reverse-Z rendering (better precision). Use `Depth24PlusStencil8` when
    /// a stencil buffer is needed.
    pub depth_format: wgpu::TextureFormat,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            path: RenderPath::default(),
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
    /// Returns the effective MSAA sample count for the current render path.
    ///
    /// - [`BasicForward`](RenderPath::BasicForward): returns the configured `msaa_samples`.
    /// - [`HighFidelity`](RenderPath::HighFidelity): always returns **1** (hardware MSAA
    ///   is disabled; use post-process AA such as FXAA instead).
    #[inline]
    #[must_use]
    pub fn msaa_samples(&self) -> u32 {
        self.path.msaa_samples()
    }
}

// Backward-compatible type alias — prefer `RendererSettings` in new code.
#[doc(hidden)]
#[deprecated(since = "0.2.0", note = "Renamed to `RendererSettings`")]
pub type RenderSettings = RendererSettings;
