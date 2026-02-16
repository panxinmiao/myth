//! Background Mode Definition
//!
//! Defines the background rendering mode for a scene.
//! This is the data layer ("what to draw"), not the rendering layer ("how to draw").
//!
//! # Supported Modes
//!
//! - [`BackgroundMode::Color`]: Solid color clear (most efficient - uses hardware clear)
//! - [`BackgroundMode::Gradient`]: Vertical gradient (top → bottom)
//! - [`BackgroundMode::Texture`]: Texture-based background (cubemap, equirectangular, planar)

use glam::Vec4;

use crate::resources::texture::TextureSource;

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
