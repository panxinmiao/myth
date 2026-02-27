//! Tone Mapping Configuration
//!
//! This module defines tone mapping modes and settings as pure data structures.
//! These are placed in the resources layer to avoid circular dependencies between
//! Scene and Renderer modules.
//!
//! In addition to tone mapping algorithm selection and exposure control, this module
//! also supports:
//! - **Vignette**: Edge darkening effect controlled by intensity and smoothness
//! - **Color Grading (LUT)**: 3D lookup table for color manipulation

use glam::Vec4;

use crate::assets::TextureHandle;
use crate::resources::WgslType;
use crate::resources::buffer::CpuBuffer;
use crate::{ShaderDefines, define_gpu_data_struct};

/// Tone mapping algorithm selection.
///
/// Different algorithms provide different looks and performance characteristics:
///
/// - [`Linear`](ToneMappingMode::Linear): No tone mapping (for debugging or LDR workflows)
/// - [`Neutral`](ToneMappingMode::Neutral): Balanced, film-like response (recommended default)
/// - [`Reinhard`](ToneMappingMode::Reinhard): Classic operator, soft highlight rolloff
/// - [`Cineon`](ToneMappingMode::Cineon): Film emulation with extended range
/// - [`ACESFilmic`](ToneMappingMode::ACESFilmic): Industry standard filmic curve
/// - [`AgX`](ToneMappingMode::AgX): Modern filmic tonemapper with excellent color handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ToneMappingMode {
    /// No tone mapping (linear passthrough)
    Linear,
    /// Neutral tone mapping (balanced, film-like)
    #[default]
    Neutral,
    /// Reinhard operator (classic, soft highlights)
    Reinhard,
    /// Cineon film emulation
    Cineon,
    /// ACES Filmic (industry standard)
    ACESFilmic,
    /// `AgX` tonemapper (modern, excellent color handling)
    AgX,
}

impl ToneMappingMode {
    /// Applies the tone mapping mode to shader defines.
    ///
    /// This sets the `TONE_MAPPING_MODE` macro to the appropriate value
    /// for shader compilation.
    pub fn apply_to_defines(&self, defines: &mut ShaderDefines) {
        let mode_str = match self {
            Self::Linear => "LINEAR",
            Self::Reinhard => "REINHARD",
            Self::Cineon => "CINEON",
            Self::ACESFilmic => "ACES_FILMIC",
            Self::Neutral => "NEUTRAL",
            Self::AgX => "AGXX",
        };
        defines.set("TONE_MAPPING_MODE", mode_str);
    }

    /// Returns a human-readable name for the mode.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Neutral => "Neutral",
            Self::Reinhard => "Reinhard",
            Self::Cineon => "Cineon",
            Self::ACESFilmic => "ACES Filmic",
            Self::AgX => "AgX",
        }
    }

    /// Returns all available tone mapping modes.
    #[must_use]
    pub fn all() -> &'static [ToneMappingMode] {
        &[
            Self::Linear,
            Self::Neutral,
            Self::Reinhard,
            Self::Cineon,
            Self::ACESFilmic,
            Self::AgX,
        ]
    }
}

define_gpu_data_struct!(
    struct ToneMappingUniforms {
        pub exposure: f32 = 1.0,
        pub contrast: f32 = 1.0,
        pub saturation: f32 = 1.0,
        pub chromatic_aberration: f32 = 0.0,

        pub film_grain: f32 = 0.0,
        pub vignette_intensity: f32 = 0.0,
        pub vignette_smoothness: f32 = 0.5,
        pub lut_contribution: f32 = 1.0,

        pub vignette_color: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1.0),
    }
);

/// Tone mapping configuration.
///
/// This struct holds all parameters for the tone mapping post-processing pass.
/// Changes to uniforms are tracked via `CpuBuffer`'s internal version for efficient GPU sync.
///
/// # Usage
///
/// ```rust,ignore
/// // Get settings from scene
/// let settings = &mut scene.tone_mapping;
///
/// // Modify parameters (automatically updates version)
/// settings.set_exposure(1.5);
/// settings.set_mode(ToneMappingMode::ACESFilmic);
/// ```
#[derive(Debug, Clone)]
pub struct ToneMappingSettings {
    /// Selected tone mapping algorithm
    pub mode: ToneMappingMode,

    pub uniforms: CpuBuffer<ToneMappingUniforms>,
    /// Exposure multiplier (default: 1.0)
    // pub exposure: f32,

    // /// === Color Adjustments ===
    // pub contrast: f32,    // Default: 1.0 (no change)
    // pub saturation: f32,    // Default: 1.0 (no change)

    // /// === Film Effects ===
    // pub chromatic_aberration: f32, // Default: 0.0 (disabled)
    // pub film_grain: f32,    // Default: 0.0 (disabled)

    // // === Vignette (edge darkening) ===
    // /// Vignette intensity: 0.0 = disabled, higher = stronger darkening (default: 0.0)
    // pub vignette_intensity: f32,
    // /// Vignette smoothness: controls the falloff curve (default: 0.5, range 0.1~1.0)
    // pub vignette_smoothness: f32,
    // /// Vignette color: RGBA values for the vignette effect (default: [0.0, 0.0, 0.0, 1.0])
    // pub vignette_color: Vec4,

    // === Color Grading (3D LUT) ===
    /// Optional 3D LUT texture handle. When `Some`, the `USE_LUT` shader macro is enabled
    /// and the pipeline is recompiled. When `None`, no LUT is applied.
    pub lut_texture: Option<TextureHandle>,
    // /// LUT contribution weight: 0.0 = original color, 1.0 = fully LUT-graded (default: 1.0)
    // // pub lut_contribution: f32,

    // /// Internal version number (for change tracking)
    // version: u64,
}

impl Default for ToneMappingSettings {
    fn default() -> Self {
        Self {
            mode: ToneMappingMode::default(),
            uniforms: CpuBuffer::new(
                ToneMappingUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("ToneMappingUniforms"),
            ),
            lut_texture: None,
        }
    }
}

impl ToneMappingSettings {
    /// Creates new tone mapping settings with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the tone mapping mode.
    ///
    /// Only updates the mode if the value actually changed.
    pub fn set_mode(&mut self, mode: ToneMappingMode) {
        if self.mode != mode {
            self.mode = mode;
        }
    }

    /// Sets the exposure value.
    pub fn set_exposure(&mut self, exposure: f32) {
        self.uniforms.write().exposure = exposure;
    }

    /// Sets the contrast adjustment.
    pub fn set_contrast(&mut self, contrast: f32) {
        self.uniforms.write().contrast = contrast;
    }

    /// Sets the saturation adjustment.
    pub fn set_saturation(&mut self, saturation: f32) {
        self.uniforms.write().saturation = saturation;
    }

    /// Sets the chromatic aberration intensity.
    pub fn set_chromatic_aberration(&mut self, intensity: f32) {
        self.uniforms.write().chromatic_aberration = intensity;
    }

    /// Sets the film grain intensity.
    pub fn set_film_grain(&mut self, intensity: f32) {
        self.uniforms.write().film_grain = intensity;
    }

    /// Sets the vignette intensity.
    ///
    /// A value of 0.0 disables the vignette effect.
    pub fn set_vignette_intensity(&mut self, intensity: f32) {
        self.uniforms.write().vignette_intensity = intensity;
    }

    /// Sets the vignette smoothness.
    ///
    /// Controls how quickly the darkening falls off from the edges.
    /// Recommended range: 0.1 ~ 1.0. Default is 0.5 for a balanced look.
    pub fn set_vignette_smoothness(&mut self, smoothness: f32) {
        self.uniforms.write().vignette_smoothness = smoothness;
    }

    /// Sets the vignette color.
    /// This is the color used for the vignette effect (default: black with full alpha).
    pub fn set_vignette_color(&mut self, color: Vec4) {
        self.uniforms.write().vignette_color = color;
    }

    /// Sets the 3D LUT texture for color grading.
    ///
    /// Pass `Some(handle)` to enable LUT-based color grading, or `None` to disable it.
    /// Changing this triggers a pipeline rebuild (shader macro change).
    pub fn set_lut_texture(&mut self, lut: Option<TextureHandle>) {
        if self.lut_texture != lut {
            self.lut_texture = lut;
        }
    }

    /// Sets the LUT contribution weight.
    ///
    /// 0.0 = original color, 1.0 = fully LUT-graded.
    pub fn set_lut_contribution(&mut self, contribution: f32) {
        self.uniforms.write().lut_contribution = contribution;
    }

    /// Returns whether a LUT texture is currently set.
    #[inline]
    #[must_use]
    pub fn has_lut(&self) -> bool {
        self.lut_texture.is_some()
    }
}
