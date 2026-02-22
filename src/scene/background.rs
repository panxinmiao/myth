//! Background Mode & Settings
//!
//! Defines the background rendering mode for a scene and the settings wrapper
//! that owns the GPU uniform buffer for skybox parameters.
//!
//! # Architecture
//!
//! [`BackgroundMode`] is the lightweight enum that describes *what* to draw
//! (solid color, gradient, or texture). [`BackgroundSettings`] wraps the mode
//! together with a `CpuBuffer<SkyboxParamsUniforms>` whose version is
//! automatically bumped only when setter methods actually change a value.
//! The render pass calls `ensure_buffer_id()` in its `prepare()` step and
//! never writes to the buffer — GPU sync happens only when needed.
//!
//! # Supported Modes
//!
//! - [`BackgroundMode::Color`]: Solid color clear (most efficient - uses hardware clear)
//! - [`BackgroundMode::Gradient`]: Vertical gradient (top → bottom)
//! - [`BackgroundMode::Texture`]: Texture-based background (cubemap, equirectangular, planar)

use glam::Vec4;

use crate::define_gpu_data_struct;
use crate::resources::WgslType;
use crate::resources::buffer::CpuBuffer;
use crate::resources::texture::TextureSource;

// ============================================================================
// GPU Uniform Struct
// ============================================================================

define_gpu_data_struct!(
    /// Skybox per-draw parameters uploaded to the GPU.
    ///
    /// Camera data (view_projection_inverse, camera_position) is obtained from
    /// the global bind group's `RenderStateUniforms`, so only skybox-specific
    /// values live here.
    struct SkyboxParamsUniforms {
        pub color_top: Vec4,
        pub color_bottom: Vec4,
        pub rotation: f32 = 0.0,
        pub intensity: f32 = 1.0,
        pub(crate) __pad0: f32,
        pub(crate) __pad1: f32,
    }
);

// ============================================================================
// BackgroundMode (lightweight enum — describes *what* to draw)
// ============================================================================

/// Background rendering mode.
///
/// Determines how the scene background is rendered. Each variant maps
/// to a different shader pipeline variant for optimal performance.
///
/// # Performance
///
/// - `Color`: Uses GPU hardware clear — zero draw calls, maximum throughput.
/// - `Gradient`: Renders a fullscreen triangle with per-vertex interpolation.
/// - `Texture`: Renders a fullscreen triangle with texture sampling + ray reconstruction.
#[derive(Clone, Debug)]
pub enum BackgroundMode {
    /// Solid color clear (most efficient).
    ///
    /// Uses the GPU's hardware clear operation — no draw calls needed.
    /// Alpha channel can be used for post-composition (typically 1.0).
    Color(Vec4),

    /// Vertical gradient (sky color → ground color).
    ///
    /// Renders a fullscreen triangle with smooth interpolation based
    /// on view direction's Y component.
    Gradient {
        /// Color at the top of the sky (Y = +1)
        top: Vec4,
        /// Color at the bottom/ground (Y = -1)
        bottom: Vec4,
    },

    /// Texture-based background (skybox / panorama / planar).
    ///
    /// Renders a fullscreen triangle with ray reconstruction from the
    /// depth buffer, sampling the background texture along the view direction.
    Texture {
        /// Texture source (asset handle or attachment)
        source: TextureSource,
        /// Y-axis rotation in radians
        rotation: f32,
        /// Brightness/exposure multiplier
        intensity: f32,
        /// Texture mapping method
        mapping: BackgroundMapping,
    },
}

/// Texture mapping method for background rendering.
///
/// Each variant produces a different shader pipeline variant
/// (via `ShaderDefines`) to avoid dynamic branching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackgroundMapping {
    /// Standard cubemap sampling.
    ///
    /// The view direction is used directly as a cubemap lookup vector.
    /// Best for pre-processed environment cubemaps.
    Cube,

    /// Equirectangular (latitude-longitude) projection.
    ///
    /// Maps the view direction to UV coordinates using `atan2` / `asin`.
    /// Best for 360° panoramic HDR images.
    Equirectangular,

    /// Planar screen-space mapping.
    ///
    /// The texture is mapped directly to screen space (UV = NDC).
    /// Not affected by camera rotation or zoom — acts as a fixed backdrop.
    Planar,
}

impl Default for BackgroundMode {
    fn default() -> Self {
        // Default: dark grey, matching common 3D editor conventions
        Self::Color(Vec4::new(0.0, 0.0, 0.0, 1.0))
    }
}

impl BackgroundMode {
    /// Creates a solid color background.
    #[inline]
    #[must_use]
    pub fn color(r: f32, g: f32, b: f32) -> Self {
        Self::Color(Vec4::new(r, g, b, 1.0))
    }

    /// Creates a solid color background with alpha.
    #[inline]
    #[must_use]
    pub fn color_with_alpha(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Color(Vec4::new(r, g, b, a))
    }

    /// Creates a vertical gradient background.
    #[inline]
    #[must_use]
    pub fn gradient(top: Vec4, bottom: Vec4) -> Self {
        Self::Gradient { top, bottom }
    }

    /// Creates a cubemap skybox background.
    #[inline]
    #[must_use]
    pub fn cubemap(source: impl Into<TextureSource>, intensity: f32) -> Self {
        Self::Texture {
            source: source.into(),
            rotation: 0.0,
            intensity,
            mapping: BackgroundMapping::Cube,
        }
    }

    /// Creates an equirectangular panorama background.
    #[inline]
    #[must_use]
    pub fn equirectangular(source: impl Into<TextureSource>, intensity: f32) -> Self {
        Self::Texture {
            source: source.into(),
            rotation: 0.0,
            intensity,
            mapping: BackgroundMapping::Equirectangular,
        }
    }

    /// Creates a planar (screen-space) background.
    #[inline]
    #[must_use]
    pub fn planar(source: impl Into<TextureSource>, intensity: f32) -> Self {
        Self::Texture {
            source: source.into(),
            rotation: 0.0,
            intensity,
            mapping: BackgroundMapping::Planar,
        }
    }

    /// Returns the clear color for the RenderPass.
    ///
    /// - `Color` mode: returns the specified color (hardware clear).
    /// - Other modes: returns black (skybox pass will overdraw).
    #[must_use]
    pub fn clear_color(&self) -> wgpu::Color {
        match self {
            Self::Color(c) => wgpu::Color {
                r: f64::from(c.x),
                g: f64::from(c.y),
                b: f64::from(c.z),
                a: f64::from(c.w),
            },
            // For gradient/texture modes, clear to black.
            // The SkyboxPass will fill uncovered pixels.
            Self::Gradient { .. } | Self::Texture { .. } => wgpu::Color::BLACK,
        }
    }

    /// Returns `true` if this mode requires a skybox draw call.
    ///
    /// `Color` mode uses hardware clear and needs no draw call.
    #[inline]
    #[must_use]
    pub fn needs_skybox_pass(&self) -> bool {
        !matches!(self, Self::Color(_))
    }
}

// ============================================================================
// BackgroundSettings (wraps mode + CpuBuffer — follows ToneMappingSettings pattern)
// ============================================================================

/// Background rendering configuration (mode + automatic uniform version control).
///
/// Wraps [`BackgroundMode`] together with a `CpuBuffer<SkyboxParamsUniforms>`
/// whose version is automatically bumped only when setter methods write new
/// values. The render pass only calls `ensure_buffer_id()` — no per-frame
/// buffer writes occur in the render pipeline.
///
/// # Usage
///
/// ```rust,ignore
/// // Set mode (automatically syncs uniform values)
/// scene.background.set_mode(BackgroundMode::equirectangular(tex, 1.0));
///
/// // Fine-tune individual parameters
/// scene.background.set_rotation(0.5);
/// scene.background.set_intensity(2.0);
/// ```
#[derive(Debug, Clone)]
pub struct BackgroundSettings {
    /// The current background rendering mode.
    pub mode: BackgroundMode,

    /// Skybox parameters uniform buffer (version-tracked).
    /// Updated via setter methods; render pass only reads.
    pub uniforms: CpuBuffer<SkyboxParamsUniforms>,
}

impl Default for BackgroundSettings {
    fn default() -> Self {
        let mode = BackgroundMode::default();
        let uniforms = CpuBuffer::new(
            SkyboxParamsUniforms::default(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("Skybox Params Uniforms"),
        );
        let mut settings = Self { mode, uniforms };
        settings.sync_uniforms_from_mode();
        settings
    }
}

impl BackgroundSettings {
    /// Sets the background mode and syncs uniform values accordingly.
    ///
    /// The `CpuBuffer` version is bumped only if the derived uniform values
    /// actually differ from the current state.
    pub fn set_mode(&mut self, mode: BackgroundMode) {
        self.mode = mode;
        self.sync_uniforms_from_mode();
    }

    /// Sets the Y-axis rotation (radians) for texture-based backgrounds.
    ///
    /// Also updates the `rotation` field inside `BackgroundMode::Texture`
    /// to keep the enum and buffer in sync.
    pub fn set_rotation(&mut self, rotation: f32) {
        if let BackgroundMode::Texture { rotation: r, .. } = &mut self.mode {
            *r = rotation;
        }
        self.uniforms.write().rotation = rotation;
    }

    /// Sets the brightness/exposure multiplier for texture-based backgrounds.
    ///
    /// Also updates the `intensity` field inside `BackgroundMode::Texture`
    /// to keep the enum and buffer in sync.
    pub fn set_intensity(&mut self, intensity: f32) {
        if let BackgroundMode::Texture { intensity: i, .. } = &mut self.mode {
            *i = intensity;
        }
        self.uniforms.write().intensity = intensity;
    }

    /// Sets gradient colors (top and bottom).
    ///
    /// Switches the mode to `Gradient` if it isn't already.
    pub fn set_gradient_colors(&mut self, top: Vec4, bottom: Vec4) {
        self.mode = BackgroundMode::Gradient { top, bottom };
        let mut p = self.uniforms.write();
        p.color_top = top;
        p.color_bottom = bottom;
        p.rotation = 0.0;
        p.intensity = 1.0;
    }

    // === Delegate methods from BackgroundMode ===

    /// Returns the clear color for the RenderPass.
    #[inline]
    #[must_use]
    pub fn clear_color(&self) -> wgpu::Color {
        self.mode.clear_color()
    }

    /// Returns `true` if the current mode requires a skybox draw call.
    #[inline]
    #[must_use]
    pub fn needs_skybox_pass(&self) -> bool {
        self.mode.needs_skybox_pass()
    }

    // === Internal ===

    /// Derives uniform values from the current `BackgroundMode` and writes
    /// them into the `CpuBuffer`.
    fn sync_uniforms_from_mode(&mut self) {
        let mut p = self.uniforms.write();
        match &self.mode {
            BackgroundMode::Color(c) => {
                p.color_top = *c;
                p.color_bottom = *c;
                p.rotation = 0.0;
                p.intensity = 1.0;
            }
            BackgroundMode::Gradient { top, bottom } => {
                p.color_top = *top;
                p.color_bottom = *bottom;
                p.rotation = 0.0;
                p.intensity = 1.0;
            }
            BackgroundMode::Texture {
                rotation,
                intensity,
                ..
            } => {
                p.color_top = Vec4::ZERO;
                p.color_bottom = Vec4::ZERO;
                p.rotation = *rotation;
                p.intensity = *intensity;
            }
        }
    }
}
