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

use crate::ShaderDefines;
use crate::assets::TextureHandle;

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

/// Tone mapping configuration (pure data + version control).
///
/// This struct holds all parameters for the tone mapping post-processing pass.
/// Changes are tracked via an internal version number for efficient GPU sync.
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
    /// Exposure multiplier (default: 1.0)
    pub exposure: f32,

    // === Vignette (edge darkening) ===
    /// Vignette intensity: 0.0 = disabled, higher = stronger darkening (default: 0.0)
    pub vignette_intensity: f32,
    /// Vignette smoothness: controls the falloff curve (default: 0.5, range 0.1~1.0)
    pub vignette_smoothness: f32,

    // === Color Grading (3D LUT) ===
    /// Optional 3D LUT texture handle. When `Some`, the `USE_LUT` shader macro is enabled
    /// and the pipeline is recompiled. When `None`, no LUT is applied.
    pub lut_texture: Option<TextureHandle>,
    /// LUT contribution weight: 0.0 = original color, 1.0 = fully LUT-graded (default: 1.0)
    pub lut_contribution: f32,

    /// Internal version number (for change tracking)
    version: u64,
}

impl Default for ToneMappingSettings {
    fn default() -> Self {
        Self {
            mode: ToneMappingMode::default(),
            exposure: 1.0,
            vignette_intensity: 0.0,
            vignette_smoothness: 0.5,
            lut_texture: None,
            lut_contribution: 1.0,
            version: 0,
        }
    }
}

impl ToneMappingSettings {
    /// Creates new tone mapping settings with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets the current version number.
    ///
    /// The version is incremented whenever any setting changes,
    /// allowing render passes to detect when updates are needed.
    #[inline]
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Sets the exposure value.
    ///
    /// Only updates the version if the value actually changed.
    pub fn set_exposure(&mut self, exposure: f32) {
        if (self.exposure - exposure).abs() > 1e-5 {
            self.exposure = exposure;
            self.bump_version();
        }
    }

    /// Sets the tone mapping mode.
    ///
    /// Only updates the version if the mode actually changed.
    pub fn set_mode(&mut self, mode: ToneMappingMode) {
        if self.mode != mode {
            self.mode = mode;
            self.bump_version();
        }
    }

    /// Sets the vignette intensity.
    ///
    /// A value of 0.0 disables the vignette effect.
    /// Only updates the version if the value actually changed.
    pub fn set_vignette_intensity(&mut self, intensity: f32) {
        if (self.vignette_intensity - intensity).abs() > 1e-5 {
            self.vignette_intensity = intensity;
            self.bump_version();
        }
    }

    /// Sets the vignette smoothness.
    ///
    /// Controls how quickly the darkening falls off from the edges.
    /// Recommended range: 0.1 ~ 1.0. Only updates the version if the value changed.
    pub fn set_vignette_smoothness(&mut self, smoothness: f32) {
        if (self.vignette_smoothness - smoothness).abs() > 1e-5 {
            self.vignette_smoothness = smoothness;
            self.bump_version();
        }
    }

    /// Sets the 3D LUT texture for color grading.
    ///
    /// Pass `Some(handle)` to enable LUT-based color grading, or `None` to disable it.
    /// Changing this triggers a pipeline rebuild (shader macro change).
    pub fn set_lut_texture(&mut self, lut: Option<TextureHandle>) {
        if self.lut_texture != lut {
            self.lut_texture = lut;
            self.bump_version();
        }
    }

    /// Sets the LUT contribution weight.
    ///
    /// 0.0 = original color, 1.0 = fully LUT-graded.
    /// Only updates the version if the value actually changed.
    pub fn set_lut_contribution(&mut self, contribution: f32) {
        if (self.lut_contribution - contribution).abs() > 1e-5 {
            self.lut_contribution = contribution;
            self.bump_version();
        }
    }

    /// Returns whether a LUT texture is currently set.
    #[inline]
    #[must_use]
    pub fn has_lut(&self) -> bool {
        self.lut_texture.is_some()
    }

    /// Manually bumps the version number.
    ///
    /// Call this to force a GPU update even if parameters haven't changed.
    #[inline]
    pub fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
}
