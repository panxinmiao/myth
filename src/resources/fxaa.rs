//! FXAA (Fast Approximate Anti-Aliasing) Configuration
//!
//! This module defines FXAA settings as a pure data structure, following the same
//! pattern as [`BloomSettings`](super::bloom::BloomSettings) and
//! [`ToneMappingSettings`](super::tone_mapping::ToneMappingSettings).
//!
//! FXAA is a screen-space anti-aliasing technique that identifies aliased edges
//! via luma contrast detection and applies sub-pixel smoothing. It operates on
//! LDR (post-tone-mapped) images and is therefore placed **after** tone mapping
//! in the HDR pipeline.
//!
//! # Quality Presets
//!
//! | Preset   | Iterations | Best for              |
//! |----------|------------|----------------------|
//! | `Low`    | 4          | Mobile / low-end GPU |
//! | `Medium` | 8          | Default balance      |
//! | `High`   | 12         | Maximum quality      |
//!
//! # Usage
//!
//! ```rust,ignore
//! // Access via scene
//! scene.fxaa.enabled = true;
//! scene.fxaa.set_quality(FxaaQuality::High);
//! ```

/// FXAA quality preset.
///
/// Controls the number of edge exploration iterations in the FXAA shader.
/// Higher quality means more iterations and better edge detection at the
/// cost of additional texture samples per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FxaaQuality {
    /// 4 iterations — suitable for mobile and low-end GPUs.
    Low,
    /// 8 iterations — good balance of quality and performance (default).
    #[default]
    Medium,
    /// 12 iterations — maximum edge exploration quality.
    High,
}

impl FxaaQuality {
    /// Returns a human-readable name for the quality preset.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }

    /// Returns all available quality presets.
    #[must_use]
    pub const fn all() -> &'static [FxaaQuality] {
        &[Self::Low, Self::Medium, Self::High]
    }

    /// Returns the shader define key for this quality preset.
    ///
    /// Used by `FxaaPass` to select the correct shader variant.
    #[must_use]
    pub(crate) const fn define_key(self) -> &'static str {
        match self {
            Self::Low => "FXAA_QUALITY_LOW",
            Self::Medium => "FXAA_QUALITY_MEDIUM",
            Self::High => "FXAA_QUALITY_HIGH",
        }
    }
}

/// FXAA post-processing configuration.
///
/// This is a lightweight settings struct — FXAA has no per-frame GPU uniforms,
/// only a quality preset that affects shader compilation.
///
/// # Usage
///
/// ```rust,ignore
/// let fxaa = &mut scene.fxaa;
/// fxaa.enabled = true;
/// fxaa.set_quality(FxaaQuality::High);
/// ```
#[derive(Debug, Clone)]
pub struct FxaaSettings {
    /// Whether FXAA is enabled.
    pub enabled: bool,

    /// Quality preset controlling edge exploration iterations.
    quality: FxaaQuality,
}

impl Default for FxaaSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            quality: FxaaQuality::default(),
        }
    }
}

impl FxaaSettings {
    /// Creates new FXAA settings with default values (enabled, medium quality).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current quality preset.
    #[inline]
    #[must_use]
    pub fn quality(&self) -> FxaaQuality {
        self.quality
    }

    /// Sets the quality preset.
    ///
    /// Changing the quality will trigger shader recompilation on the next frame.
    pub fn set_quality(&mut self, quality: FxaaQuality) {
        self.quality = quality;
    }

    /// Sets whether FXAA is enabled.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}
