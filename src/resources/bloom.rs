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
//! # GPU Uniform Structs
//!
//! - [`UpsampleUniforms`]: Controls the tent filter radius during upsampling.
//! - [`CompositeUniforms`]: Controls bloom strength during final composition.
//!
//! These structs are defined here (rather than in the render pass) so that
//! `BloomSettings` can own the `CpuBuffer<T>` instances. User-facing setters
//! write directly into the buffers via `CpuBuffer::write()`, which automatically
//! tracks the data version. The render pass only calls `ensure_buffer_id()` and
//! never writes to the buffers itself; GPU sync happens only when the version
//! has actually changed.
//!
//! # Reference
//!
//! - [Physically Based Bloom (LearnOpenGL)](https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom)
//! - *Next Generation Post Processing in Call of Duty: Advanced Warfare* (SIGGRAPH 2014)

use crate::define_gpu_data_struct;
use crate::resources::WgslType;
use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::UniformArray;

// ============================================================================
// GPU Uniform Structs
// ============================================================================

define_gpu_data_struct!(
    /// GPU uniform data for the upsample shader.
    ///
    /// Controls the tent filter radius used during the upsampling phase.
    struct UpsampleUniforms {
        pub filter_radius: f32,
        pub(crate) __pad: UniformArray<u32, 3>,
    }
);

define_gpu_data_struct!(
    /// GPU uniform data for the composite shader.
    ///
    /// Controls how much bloom contributes to the final image.
    struct CompositeUniforms {
        pub bloom_strength: f32,
        pub(crate) __pad: UniformArray<u32, 3>,
    }
);

// ============================================================================
// BloomSettings
// ============================================================================

/// Bloom post-processing configuration (pure data + automatic version control).
///
/// This struct holds all parameters for the physically-based bloom pass.
/// Dynamic GPU uniform data lives in `CpuBuffer<T>` fields; the internal
/// version is automatically bumped when setter methods modify values via
/// `CpuBuffer::write()`. The render pass calls `ensure_buffer_id()` which
/// only performs a GPU upload when the version has changed.
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

    /// Maximum number of mip levels in the downsample/upsample chain.
    ///
    /// More mip levels produce wider bloom at the cost of additional passes.
    /// The actual count is clamped to the maximum possible for the render
    /// target size.
    ///
    /// Default: `6`
    max_mip_levels: u32,

    /// Whether to apply Karis average on the first downsample pass.
    ///
    /// Karis averaging weights samples by inverse luminance, which
    /// suppresses firefly artifacts from extremely bright pixels.
    /// Recommended for scenes with high-contrast specular highlights.
    ///
    /// Default: `true`
    pub karis_average: bool,

    /// Upsample filter uniforms (`filter_radius`).
    /// Updated via `set_radius()` — version tracking is automatic.
    pub(crate) upsample_uniforms: CpuBuffer<UpsampleUniforms>,

    /// Composite blend uniforms (`bloom_strength`).
    /// Updated via `set_strength()` — version tracking is automatic.
    pub(crate) composite_uniforms: CpuBuffer<CompositeUniforms>,
}

impl Default for BloomSettings {
    fn default() -> Self {
        let upsample = UpsampleUniforms {
            filter_radius: 0.005,
            ..Default::default()
        };

        let composite = CompositeUniforms {
            bloom_strength: 0.04,
            ..Default::default()
        };

        Self {
            enabled: false,
            max_mip_levels: 6,
            karis_average: true,
            upsample_uniforms: CpuBuffer::new(
                upsample,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("Bloom Upsample Uniforms"),
            ),
            composite_uniforms: CpuBuffer::new(
                composite,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("Bloom Composite Uniforms"),
            ),
        }
    }
}

impl BloomSettings {
    /// Creates new bloom settings with default values (disabled).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current bloom strength.
    #[inline]
    #[must_use]
    pub fn strength(&self) -> f32 {
        self.composite_uniforms.read().bloom_strength
    }

    /// Returns the current upsample filter radius.
    #[inline]
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.upsample_uniforms.read().filter_radius
    }

    /// Returns the maximum number of mip levels.
    #[inline]
    #[must_use]
    pub fn max_mip_levels(&self) -> u32 {
        self.max_mip_levels
    }

    /// Sets whether bloom is enabled.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Sets the bloom strength.
    ///
    /// Controls how much the bloom contributes to the final image.
    /// A value of 0.0 effectively disables bloom; typical values are 0.01–0.1.
    pub fn set_strength(&mut self, strength: f32) {
        self.composite_uniforms.write().bloom_strength = strength.max(0.0);
    }

    /// Sets the maximum number of mip levels.
    pub fn set_max_mip_levels(&mut self, levels: u32) {
        self.max_mip_levels = levels.clamp(1, 16);
    }

    /// Sets the upsampling filter radius.
    ///
    /// Larger values produce softer, wider bloom. Very small values
    /// can produce aliasing artifacts.
    pub fn set_radius(&mut self, radius: f32) {
        self.upsample_uniforms.write().filter_radius = radius.max(0.0);
    }

    /// Sets whether Karis average is used on the first downsample.
    pub fn set_karis_average(&mut self, enabled: bool) {
        self.karis_average = enabled;
    }
}
