//! Bloom Post-Processing Configuration
//!
//! This module defines bloom settings as pure data structures, following the same
//! pattern as [`ToneMappingSettings`](super::tone_mapping::ToneMappingSettings).
//!
//! The bloom implementation is based on the physically-based bloom technique from
//! *Call of Duty: Advanced Warfare*, which uses progressive downsampling with a
//! 13-tap filter and upsampling with a 3×3 tent filter. Unlike threshold-based
//! bloom, this approach naturally preserves energy and produces realistic results
//! in HDR pipelines.
//!
//! # Reference
//!
//! - [Physically Based Bloom (LearnOpenGL)](https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom)
//! - *Next Generation Post Processing in Call of Duty: Advanced Warfare* (SIGGRAPH 2014)

/// Bloom post-processing configuration (pure data + version control).
///
/// This struct holds all parameters for the physically-based bloom pass.
/// Changes are tracked via an internal version number for efficient GPU sync.
///
/// # Usage
///
/// ```rust,ignore
/// // Access via scene
/// let bloom = &mut scene.bloom;
///
/// // Enable bloom and configure
/// bloom.set_enabled(true);
/// bloom.set_strength(0.04);
/// bloom.set_radius(0.005);
/// ```
#[derive(Debug, Clone)]
pub struct BloomSettings {
    /// Whether bloom is enabled.
    pub enabled: bool,

    /// Bloom intensity multiplier applied during final composition.
    ///
    /// Controls how much the bloom contributes to the final image.
    /// A value of 0.0 disables bloom; typical values are 0.01–0.1.
    ///
    /// Default: `0.04`
    pub strength: f32,

    /// Maximum number of mip levels in the downsample/upsample chain.
    ///
    /// More mip levels produce wider bloom at the cost of additional passes.
    /// The actual count is clamped to the maximum possible for the render
    /// target size.
    ///
    /// Default: `8`
    pub max_mip_levels: u32,

    /// Filter radius for the upsampling tent filter, in UV-space coordinates.
    ///
    /// Larger values produce softer, wider bloom. Very small values
    /// can produce aliasing artifacts.
    ///
    /// Default: `0.005`
    pub radius: f32,

    /// Whether to apply Karis average on the first downsample pass.
    ///
    /// Karis averaging weights samples by inverse luminance, which
    /// suppresses firefly artifacts from extremely bright pixels.
    /// Recommended for scenes with high-contrast specular highlights.
    ///
    /// Default: `true`
    pub karis_average: bool,

    /// Internal version number (for change tracking).
    version: u64,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            strength: 0.04,
            max_mip_levels: 6,
            radius: 0.005,
            karis_average: true,
            version: 0,
        }
    }
}

impl BloomSettings {
    /// Creates new bloom settings with default values (disabled).
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

    /// Sets whether bloom is enabled.
    pub fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.bump_version();
        }
    }

    /// Sets the bloom strength.
    pub fn set_strength(&mut self, strength: f32) {
        let strength = strength.max(0.0);
        if (self.strength - strength).abs() > 1e-6 {
            self.strength = strength;
            self.bump_version();
        }
    }

    /// Sets the maximum number of mip levels.
    pub fn set_max_mip_levels(&mut self, levels: u32) {
        let levels = levels.clamp(1, 16);
        if self.max_mip_levels != levels {
            self.max_mip_levels = levels;
            self.bump_version();
        }
    }

    /// Sets the upsampling filter radius.
    pub fn set_radius(&mut self, radius: f32) {
        let radius = radius.max(0.0);
        if (self.radius - radius).abs() > 1e-6 {
            self.radius = radius;
            self.bump_version();
        }
    }

    /// Sets whether Karis average is used on the first downsample.
    pub fn set_karis_average(&mut self, enabled: bool) {
        if self.karis_average != enabled {
            self.karis_average = enabled;
            self.bump_version();
        }
    }

    /// Manually bumps the version number.
    #[inline]
    pub fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
}
